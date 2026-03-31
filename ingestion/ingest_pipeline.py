import os
from pathlib import Path

from langchain_core.documents import Document

from core.utils.logger_utils import logger
from core.utils.path_utils import get_abs_path
from ingestion.indexing import KnowledgeChunkIndexer, MilvusDocumentSyncPlan
from ingestion.processors import DEFAULT_IMAGES_OUTPUT_DIR
from ingestion.processors import MarkdownProcessor, SpecialChunkExplainer
from ingestion.processors import PDFToMarkdownProcessor

DEFAULT_PDF_API_URL = "https://5d0ce1b5kaiea6u7.aistudio-app.com/layout-parsing"
DEFAULT_PDF_TOKEN = "68aead6db9f471c9a52684188382fc92f661d0d7"


class IngestPipeline:
    def __init__(
            self,
            chunk_threshold: int = 1000,
            images_output_dir: str | Path = DEFAULT_IMAGES_OUTPUT_DIR,
            explain_special: bool = True,
            window_size: int = 1,
            skip_image: bool = False,
            pdf_api_url: str | None = None,
            pdf_token: str | None = None,
    ):
        """
        Initialize the ingestion pipeline.

        Behavior:
            - Supports markdown files directly.
            - Supports PDF files by converting them to markdown first.
            - Optionally explains code, table, and image chunks.
            - Produces final documents for insertion.
            - Uses the indexer to insert documents directly into Milvus.

        Args:
            chunk_threshold (int): Semantic split threshold for Markdown text chunks.
            images_output_dir (str | Path): Directory used to store copied or downloaded images.
            explain_special (bool): Whether to explain code, table, and image chunks before embedding.
            window_size (int): Sliding window size for special chunk explanation.
            skip_image (bool): Whether to skip image chunks before final insertion.
            pdf_api_url (str | None): PDF parsing API endpoint.
            pdf_token (str | None): PDF parsing API token.
        """
        self.chunk_threshold = chunk_threshold
        self.images_output_dir = Path(images_output_dir)
        self.explain_special = explain_special
        self.window_size = window_size
        self.skip_image = skip_image
        self.pdf_api_url = pdf_api_url
        self.pdf_token = pdf_token

        self.markdown_processor = MarkdownProcessor(
            images_output_dir=self.images_output_dir,
            chunk_threshold=self.chunk_threshold,
        )
        self.special_chunk_explainer = SpecialChunkExplainer(window_size=self.window_size)
        self.knowledge_chunk_indexer = KnowledgeChunkIndexer()

    def inspect_document(self, file_path: str | Path) -> MilvusDocumentSyncPlan:
        """
        Inspect the document at task start and decide whether it needs work.

        Workflow:
            1. Validate the source file exists.
            2. Compare source path and content hash against Milvus.
            3. Return the sync plan for the current task.

        Args:
            file_path (str | Path): Input document path.

        Returns:
            MilvusDocumentSyncPlan: Upfront task decision for the current file.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {path}")

        return self.knowledge_chunk_indexer.inspect_document(path)

    def process_document(
            self,
            file_path: str | Path,
            sync_plan: MilvusDocumentSyncPlan | None = None,
    ) -> list[Document]:
        """
        Process a document file into final ordered documents.

        Workflow:
            1. Use the upfront sync plan from task inspection.
            2. Skip processing immediately when the file does not need work.
            3. Detect file type.
            4. Convert PDF to markdown when needed.
            5. Process markdown into ordered documents.
            6. Optionally explain special chunks.
            7. Return final documents for downstream embedding or storage.

        Args:
            file_path (str | Path): Input document path.
            sync_plan (MilvusDocumentSyncPlan | None): Optional precomputed task decision.

        Returns:
            list[Document]: Final documents ready for vector-store insertion.
        """
        path = Path(file_path)
        current_sync_plan = sync_plan or self.inspect_document(path)
        if current_sync_plan.should_skip:
            return []

        logger.info(f"Start processing document: {path}")
        docs = self._build_documents(path)
        logger.info(f"Document chunks generated: {len(docs)}")

        if self.explain_special:
            docs = self.special_chunk_explainer.explain_documents(docs)
            logger.info("Special chunk explanation completed")

        final_docs = self.knowledge_chunk_indexer.select_documents(docs, skip_image=self.skip_image)
        logger.info(f"Final document count: {len(final_docs)}")
        return final_docs

    def store_document(self, file_path: str | Path) -> dict:
        """
        Process a document and store the final documents into Milvus.

        Args:
            file_path (str | Path): Input document path.

        Returns:
            dict: Insert result returned by the Milvus client.
        """
        sync_plan = self.inspect_document(file_path)
        documents = self.process_document(file_path, sync_plan=sync_plan)
        result = self.knowledge_chunk_indexer.sync_documents(
            documents=documents,
            sync_plan=sync_plan,
            skip_image=False,
        )
        return result

    def _build_documents(self, file_path: Path):
        """
        Build ordered documents from an input file.

        Args:
            file_path (Path): Input file path.

        Returns:
            list[Document]: Ordered documents produced from the file.

        Raises:
            ValueError: If the file type is unsupported or PDF configuration is missing.
        """
        suffix = file_path.suffix.lower()
        if suffix in {".md", ".markdown"}:
            logger.info("Detected markdown input")
            return self.markdown_processor.process_file(file_path)

        if suffix == ".pdf":
            logger.info("Detected PDF input")
            if not self.pdf_api_url or not self.pdf_token:
                raise ValueError("PDF processing requires both pdf_api_url and pdf_token")

            pdf_processor = PDFToMarkdownProcessor(
                api_url=self.pdf_api_url,
                token=self.pdf_token,
                images_output_dir=self.images_output_dir,
            )
            markdown_content = pdf_processor.parse_pdf(file_path)
            return self.markdown_processor.process_content(markdown_content, str(file_path))

        raise ValueError(f"Unsupported document type: {file_path.suffix}")


def build_knowledge_base():
    """
    Run the document ingestion pipeline locally and insert the final documents into Milvus.
    """
    import argparse

    default_markdown_dir = Path(get_abs_path(os.path.join("data", "docs", "markdown")))

    parser = argparse.ArgumentParser(description="Run the ingestion pipeline and print final records.")
    parser.add_argument(
        "file_path",
        nargs="?",
        default=str(default_markdown_dir),
        help="Absolute path to an input markdown/PDF file, or a directory containing markdown files.",
    )
    parser.add_argument(
        "--chunk-threshold",
        type=int,
        default=1000,
        help="Semantic split threshold for text documents.",
    )
    parser.add_argument(
        "--images-output-dir",
        default=DEFAULT_IMAGES_OUTPUT_DIR,
        help="Directory used to copy or download images.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=1,
        help="Sliding window size used when explaining special chunks.",
    )
    parser.add_argument(
        "--skip-image",
        action="store_true",
        help="Skip image chunks before final insertion.",
    )
    parser.add_argument(
        "--disable-explain-special",
        action="store_true",
        help="Disable special chunk explanation before embedding.",
    )
    parser.add_argument(
        "--pdf-api-url",
        default=DEFAULT_PDF_API_URL,
        help="PDF parsing API endpoint.",
    )
    parser.add_argument(
        "--pdf-token",
        default=DEFAULT_PDF_TOKEN,
        help="PDF parsing API token.",
    )
    args = parser.parse_args()

    pipeline = IngestPipeline(
        chunk_threshold=args.chunk_threshold,
        images_output_dir=args.images_output_dir,
        explain_special=not args.disable_explain_special,
        window_size=args.window_size,
        skip_image=args.skip_image,
        pdf_api_url=args.pdf_api_url,
        pdf_token=args.pdf_token,
    )

    input_path = Path(args.file_path)
    if input_path.is_dir():
        markdown_files = sorted(input_path.glob("*.md"))
        if not markdown_files:
            raise FileNotFoundError(f"No markdown files found in directory: {input_path}")

        for markdown_file in markdown_files:
            pipeline.store_document(markdown_file)
        return

    pipeline.store_document(input_path)


if __name__ == "__main__":
    build_knowledge_base()
