import base64
import mimetypes
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from ingestion.processors.special_chunk_prompts import (
    CODE_EXPLANATION_PROMPT,
    IMAGE_EXPLANATION_PROMPT,
    TABLE_EXPLANATION_PROMPT,
)
from models import llm
from models import vlm


class SpecialChunkExplainer:
    def __init__(
            self,
            window_size: int = 1,
            code_model: BaseChatModel = llm,
            table_model: BaseChatModel = llm,
            image_model: BaseChatModel = vlm,
    ):
        """
        Initialize the special chunk explainer.

        Behavior:
            - Uses a text model for code explanation.
            - Uses a text model for table explanation.
            - Uses a multimodal model for image explanation.
            - Builds dedicated LangChain chains for each chunk type.

        Args:
            window_size (int): Number of neighbor chunks to include on each side.
            code_model: Runnable model used for code explanation.
            table_model: Runnable model used for table explanation.
            image_model: Runnable multimodal model used for image explanation.
        """
        self.window_size = window_size
        self.str_output_parser = StrOutputParser()
        self.code_chain = CODE_EXPLANATION_PROMPT | code_model | self.str_output_parser
        self.table_chain = TABLE_EXPLANATION_PROMPT | table_model | self.str_output_parser
        self.image_chain = IMAGE_EXPLANATION_PROMPT | image_model | self.str_output_parser

    def explain_documents(self, docs: list[Document]) -> list[Document]:
        """
        Explain all special chunks in an ordered document list.

        Behavior:
            - Leaves plain text chunks unchanged.
            - Replaces page content of code, table, and image chunks with model explanations.
            - Preserves metadata and original document order.

        Args:
            docs (list[Document]): Ordered documents produced by the Markdown processor.

        Returns:
            list[Document]: Documents with explained special chunks.
        """
        processed_docs: list[Document] = []

        for index, doc in enumerate(docs):
            if not self._is_special_chunk(doc):
                processed_docs.append(doc)
                continue

            explanation = self._explain_special_chunk(docs=docs, index=index)
            processed_docs.append(
                Document(
                    page_content=explanation,
                    metadata=doc.metadata,
                )
            )

        return processed_docs

    @staticmethod
    def _is_special_chunk(doc: Document) -> bool:
        """
        Check whether a document requires special explanation.

        Args:
            doc (Document): Input document.

        Returns:
            bool: True for code, table, or image chunks.
        """
        return str(doc.metadata.get("type", "")) in {"code", "table", "image"}

    def _explain_special_chunk(self, docs: list[Document], index: int) -> str:
        """
        Route a target chunk to its dedicated explanation chain.

        Args:
            docs (list[Document]): Full ordered document list.
            index (int): Index of the target chunk.

        Returns:
            str: Explanation text generated for the target chunk.
        """
        doc = docs[index]
        doc_type = str(doc.metadata.get("type", ""))

        if doc_type == "code":
            return self._explain_code_chunk(docs=docs, index=index)
        if doc_type == "table":
            return self._explain_table_chunk(docs=docs, index=index)
        if doc_type == "image":
            return self._explain_image_chunk(docs=docs, index=index)

        return doc.page_content

    def _explain_code_chunk(self, docs: list[Document], index: int) -> str:
        """
        Explain a code chunk using the code chain.

        Args:
            docs (list[Document]): Full ordered document list.
            index (int): Index of the target code chunk.

        Returns:
            str: Code explanation text.
        """
        doc = docs[index]
        payload = {
            "chunk_index": doc.metadata.get("chunk_index", ""),
            "heading_hierarchy": doc.metadata.get("heading_hierarchy", ""),
            "code": doc.metadata.get("code", ""),
            "window_context": self._build_window_context(docs, index),
        }
        result = self.code_chain.invoke(payload).strip()
        return result

    def _explain_table_chunk(self, docs: list[Document], index: int) -> str:
        """
        Explain a table chunk using the table chain.

        Args:
            docs (list[Document]): Full ordered document list.
            index (int): Index of the target table chunk.

        Returns:
            str: Table explanation text.
        """
        doc = docs[index]
        payload = {
            "chunk_index": doc.metadata.get("chunk_index", ""),
            "heading_hierarchy": doc.metadata.get("heading_hierarchy", ""),
            "table": doc.metadata.get("table", ""),
            "window_context": self._build_window_context(docs, index),
        }
        result = self.table_chain.invoke(payload).strip()
        return result

    def _explain_image_chunk(self, docs: list[Document], index: int) -> str:
        """
        Explain an image chunk using the multimodal chain.

        Args:
            docs (list[Document]): Full ordered document list.
            index (int): Index of the target image chunk.

        Returns:
            str: Image explanation text.
        """
        doc = docs[index]
        payload = {
            "chunk_index": doc.metadata.get("chunk_index", ""),
            "heading_hierarchy": doc.metadata.get("heading_hierarchy", ""),
            "window_context": self._build_window_context(docs, index),
            "image_data_url": self._to_image_data_url(str(doc.metadata.get("image_path", ""))),
        }
        result = self.image_chain.invoke(payload).strip()
        return result

    def _build_window_context(self, docs: list[Document], target_index: int) -> str:
        """
        Build sliding-window context text around a target chunk.

        Args:
            docs (list[Document]): Full ordered document list.
            target_index (int): Target chunk index.

        Returns:
            str: Concatenated context text for prompt input.
        """
        context_blocks = []

        for index, doc in self._get_window_docs(docs, target_index):
            metadata = doc.metadata
            context_blocks.append(
                "\n".join([
                    f"[chunk_index] {metadata.get('chunk_index', '')}",
                    f"[type] {metadata.get('type', '')}",
                    f"[heading_hierarchy] {metadata.get('heading_hierarchy', '')}",
                    f"[is_target] {'yes' if index == target_index else 'no'}",
                    f"[content]\n{self._get_chunk_source_text(doc)}",
                ])
            )

        context = "\n\n".join(context_blocks)
        return context

    def _get_window_docs(self, docs: list[Document], target_index: int) -> list[tuple[int, Document]]:
        """
        Collect chunks inside the configured sliding window.

        Args:
            docs (list[Document]): Full ordered document list.
            target_index (int): Target chunk index.

        Returns:
            list[tuple[int, Document]]: Window documents with their original indices.
        """
        start = max(0, target_index - self.window_size)
        end = min(len(docs), target_index + self.window_size + 1)
        return [(index, docs[index]) for index in range(start, end)]

    @staticmethod
    def _get_chunk_source_text(doc: Document) -> str:
        """
        Extract the raw source content used for prompt context.

        Args:
            doc (Document): Input chunk document.

        Returns:
            str: Raw code, table, image path, or page content depending on chunk type.
        """
        metadata = doc.metadata
        doc_type = str(metadata.get("type", ""))

        if doc_type == "code":
            return str(metadata.get("code", ""))
        if doc_type == "table":
            return str(metadata.get("table", ""))
        if doc_type == "image":
            return str(metadata.get("image_path", ""))
        return doc.page_content

    @staticmethod
    def _to_image_data_url(image_path: str) -> str:
        """
        Convert a local image file into a base64 data URL for multimodal input.

        Args:
            image_path (str): Local image path.

        Returns:
            str: Base64 data URL.

        Raises:
            FileNotFoundError: If the image path does not exist.
        """
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        mime_type, _ = mimetypes.guess_type(path.name)
        normalized_mime_type = mime_type or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{normalized_mime_type};base64,{encoded}"
