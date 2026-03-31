from pathlib import Path

from langchain_core.documents import Document

from app.vectorstores.milvus_vector_store import MilvusVectorStore
from core.utils.logger_utils import logger
from core.config.milvus_config import MILVUS_DOCS_COLLECTION
from ingestion.indexing.milvus_document_comparator import (
    MilvusDocumentComparator,
    MilvusDocumentSyncPlan,
)


class KnowledgeChunkIndexer:
    def __init__(self):
        """
        Initialize the knowledge chunk indexer.

        Behavior:
            - Reuses the LangChain Milvus vector store wrapper.
            - Stores processed documents directly through `add_documents`.
            - Lets the vector store handle embedding and insertion together.
        """
        self.vector_store = MilvusVectorStore().vector_store
        self.comparator = MilvusDocumentComparator()

    @staticmethod
    def is_image_chunk(doc: Document) -> bool:
        """
        Check whether a document is an image chunk.

        Args:
            doc (Document): Input document.

        Returns:
            bool: True when the chunk type is image.
        """
        return str(doc.metadata.get("type", "")) == "image"

    def select_documents(self, documents: list[Document], skip_image: bool = False) -> list[Document]:
        """
        Select documents that should be inserted into the vector store.

        Args:
            documents (list[Document]): Input document list.
            skip_image (bool): Whether to skip image chunks.

        Returns:
            list[Document]: Filtered document list.
        """
        if not skip_image:
            return documents
        return [doc for doc in documents if not self.is_image_chunk(doc)]

    @staticmethod
    def get_raw_data(doc: Document) -> str | None:
        """
        Build the raw data field stored alongside vector data.

        Mapping:
            - image -> raw image path
            - table -> raw table text
            - code -> raw code text
            - text  -> empty string

        Args:
            doc (Document): Input document.

        Returns:
            str: Raw source data for the current chunk type.
        """
        metadata = doc.metadata
        doc_type = str(metadata.get("type", "text"))

        if doc_type == "image":
            return str(metadata.get("image_path", "")).strip()

        if doc_type == "table":
            return str(metadata.get("table", "")).strip()

        if doc_type == "code":
            return str(metadata.get("code", "")).strip()

        return ""

    def prepare_documents_for_insert(self, documents: list[Document], skip_image: bool = False) -> list[Document]:
        """
        Prepare documents for Milvus insertion.

        Behavior:
            - Optionally filters image chunks.
            - Adds the `raw_data` field into document metadata.

        Args:
            documents (list[Document]): Input document list.
            skip_image (bool): Whether to skip image chunks.

        Returns:
            list[Document]: Documents ready for vector-store insertion.
        """
        selected_documents = self.select_documents(documents, skip_image=skip_image)
        prepared_documents: list[Document] = []

        for doc in selected_documents:
            prepared_documents.append(
                Document(
                    page_content=doc.page_content,
                    metadata=doc.metadata | {"raw_data": self.get_raw_data(doc)},
                )
            )

        return prepared_documents

    def insert_documents(self, documents: list[Document], skip_image: bool = False) -> dict:
        """
        Insert processed documents into the configured vector store.

        Args:
            documents (list[Document]): Documents produced by the ingestion pipeline.
            skip_image (bool): Whether to skip image chunks before insertion.

        Returns:
            dict: Insert summary containing inserted ids and count.
        """
        prepared_documents = self.prepare_documents_for_insert(documents, skip_image=skip_image)
        if not prepared_documents:
            return {"insert_count": 0, "ids": []}

        logger.info(
            f"Insert documents into vector store: document_count={len(prepared_documents)}")
        ids = self.vector_store.add_documents(prepared_documents)
        self.comparator.get_client().flush(MILVUS_DOCS_COLLECTION)
        result = {
            "insert_count": len(ids),
            "ids": ids,
        }
        return result

    def inspect_document(self, file_path: str | Path) -> MilvusDocumentSyncPlan:
        return self.comparator.inspect_document(file_path)

    def sync_documents(
            self,
            documents: list[Document],
            sync_plan: MilvusDocumentSyncPlan,
            skip_image: bool = False,
    ) -> dict:
        """
        Synchronize one source document with Milvus according to comparator rules.

        Args:
            documents (list[Document]): Processed source documents.
            sync_plan (MilvusDocumentSyncPlan): Upfront document sync decision.
            skip_image (bool): Whether to skip image chunks before insertion.

        Returns:
            dict: Sync result containing action, delete result, and insert result.
        """
        if sync_plan.should_skip:
            result = sync_plan.to_result()
            logger.info(f"Skip Milvus sync: {result}")
            return result

        delete_result = None
        if sync_plan.action == "replace":
            delete_result = self.comparator.delete_document_chunks(sync_plan.doc_id)

        insert_result = self.insert_documents(documents, skip_image=skip_image)
        result = sync_plan.to_result(
            delete_result=delete_result,
            insert_result=insert_result,
        )
        logger.info(f"Milvus sync completed: {result}")
        return result
