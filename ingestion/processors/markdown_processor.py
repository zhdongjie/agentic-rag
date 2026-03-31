import hashlib
import os
import re
import shutil
import time
from pathlib import Path
from typing import Pattern
from urllib.parse import urlparse
from urllib.request import urlopen

from langchain_core.documents import Document

from core.utils.file_utils import (
    list_files_with_allowed_extensions,
)
from core.utils.logger_utils import logger
from core.utils.path_utils import get_abs_path
from ingestion.indexing import MilvusDocumentComparator
from ingestion.splitters import MarkdownSplitter

DEFAULT_IMAGES_OUTPUT_DIR = get_abs_path(os.path.join("data", "docs", "images"))


class MarkdownProcessor:
    def __init__(
            self,
            images_output_dir: str | Path = DEFAULT_IMAGES_OUTPUT_DIR,
            chunk_threshold: int = 1000,
    ):
        """
        Initialize the Markdown processor.

        Behavior:
            - Splits Markdown documents by heading hierarchy first.
            - Further separates each section into text, code, table, and image chunks.
            - Copies or downloads markdown images into a local directory.
            - Preserves chunk order and attaches metadata required by downstream indexing.

        Args:
            images_output_dir (str | Path): Directory used to save local copies of markdown images.
            chunk_threshold (int): Length threshold beyond which text chunks will be semantically split.
        """
        self.images_output_dir = Path(images_output_dir)
        self.chunk_threshold = chunk_threshold
        self.splitter = MarkdownSplitter()
        self.header_keys = [header_key for _, header_key in self.splitter.headers_to_split_on]
        self.code_block_pattern: Pattern[str] = re.compile(
            r"(?P<code>^[ \t]*```(?P<lang>[^\n`]*)\n.*?^[ \t]*```[ \t]*$)",
            re.MULTILINE | re.DOTALL,
        )
        self.table_pattern: Pattern[str] = re.compile(
            r"(?P<table>"
            r"^\|(?:[^\n|]*\|){2,}\s*\n"
            r"^\|(?:\s*:?-+:?\s*\|)+\s*$\n"
            r"(?:^\|(?:[^\n|]*\|){2,}\s*$\n?)*)",
            re.MULTILINE,
        )
        self.image_pattern: Pattern[str] = re.compile(
            r'(?P<image>!\[(?P<alt>[^]]*)]\((?P<target>[^)\n]+)\))'
        )

        self._ensure_dir(self.images_output_dir)

    @staticmethod
    def _ensure_dir(path: str | Path):
        """
        Create a directory if it does not already exist.

        Args:
            path (str | Path): Directory path to create.

        Raises:
            OSError: If the directory cannot be created.
        """
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error(f"Failed to create directory {path}: {exc}")
            raise

    def process_content(self, markdown_content: str, original_file_path: str) -> list[Document]:
        """
        Convert a markdown string into an ordered list of LangChain documents.

        Workflow:
            1. Build document-level metadata.
            2. Split the markdown by headings.
            3. Split each heading section into ordered text, code, table, and image blocks.
            4. Normalize heading metadata and finalize chunk metadata.

        Args:
            markdown_content (str): Raw Markdown text.
            original_file_path (str): Source file path of the Markdown document.

        Returns:
            list[Document]: Ordered chunk documents ready for later explanation and embedding.
        """
        if not markdown_content or not markdown_content.strip():
            return []

        normalized_source_path = MilvusDocumentComparator.normalize_source_path(original_file_path)
        base_metadata = {
            "doc_id": MilvusDocumentComparator.build_doc_id(normalized_source_path),
            "doc_hash": MilvusDocumentComparator.build_doc_hash(normalized_source_path),
            "source": normalized_source_path,
            "created_at": int(time.time()),
        }

        markdown_docs = self.splitter.md_header_splitter.split_text(markdown_content)

        ordered_docs: list[Document] = []
        for section_index, markdown_doc in enumerate(markdown_docs, start=1):
            ordered_docs.extend(
                self._split_section_blocks(
                    content=markdown_doc.page_content,
                    section_metadata=markdown_doc.metadata,
                    base_metadata=base_metadata,
                    markdown_path=original_file_path,
                )
            )

        final_docs = self._finalize_documents(self._add_heading_hierarchy(ordered_docs))
        return final_docs

    def _split_section_blocks(
            self,
            content: str,
            section_metadata: dict,
            base_metadata: dict,
            markdown_path: str,
    ) -> list[Document]:
        """
        Split a heading section into ordered block documents.

        Behavior:
            - Scans the section from left to right.
            - Recognizes code blocks, Markdown tables, and markdown images.
            - Emits text between special blocks as plain text documents.
            - Preserves the original order of all generated chunks.

        Args:
            content (str): Markdown text inside a heading section.
            section_metadata (dict): Heading metadata generated by the header splitter.
            base_metadata (dict): File-level metadata shared by all chunks.
            markdown_path (str): Source markdown file path.

        Returns:
            list[Document]: Ordered documents belonging to the section.
        """
        if not content or not content.strip():
            return []

        docs: list[Document] = []
        text_start = 0
        cursor = 0

        while cursor < len(content):
            block = self._match_block_at(content, cursor)
            if block is None:
                cursor += 1
                continue

            block_type, match = block
            start, end = match.span()

            if start > text_start:
                docs.extend(
                    self._build_text_docs(
                        text=content[text_start:start],
                        metadata=section_metadata | base_metadata,
                    )
                )

            if block_type == "code":
                docs.append(
                    Document(
                        page_content=self._get_match_group(match, "code").strip(),
                        metadata=section_metadata | base_metadata | {
                            "type": "code",
                            "code": self._get_match_group(match, "code").strip(),
                            "language": self._get_match_group(match, "lang").strip(),
                        },
                    )
                )
            elif block_type == "table":
                docs.append(
                    Document(
                        page_content=self._get_match_group(match, "table").strip(),
                        metadata=section_metadata | base_metadata | {
                            "type": "table",
                            "table": self._get_match_group(match, "table").strip(),
                        },
                    )
                )
            elif block_type == "image":
                image_doc = self._build_image_doc(
                    match=match,
                    metadata=section_metadata | base_metadata,
                    markdown_path=markdown_path,
                )
                if image_doc is not None:
                    docs.append(image_doc)

            cursor = end
            text_start = cursor

        if text_start < len(content):
            docs.extend(
                self._build_text_docs(
                    text=content[text_start:],
                    metadata=section_metadata | base_metadata,
                )
            )

        return docs

    def _match_block_at(self, content: str, cursor: int) -> tuple[str, re.Match[str]] | None:
        """
        Try to match a special markdown block at the current cursor position.

        Matching order:
            1. Code block
            2. Markdown table
            3. Markdown image

        Args:
            content (str): Source markdown content.
            cursor (int): Current scan position.

        Returns:
            tuple[str, re.Match[str]] | None: Matched block type and regex match, or None.
        """
        patterns = [
            ("code", self.code_block_pattern),
            ("table", self.table_pattern),
            ("image", self.image_pattern),
        ]

        for block_type, pattern in patterns:
            match = pattern.match(content, cursor)
            if match is not None:
                return block_type, match

        return None

    def _build_text_docs(self, text: str, metadata: dict) -> list[Document]:
        """
        Build one or more text documents from a plain text segment.

        Behavior:
            - Ignores empty or whitespace-only segments.
            - Creates a single text document for short content.
            - Applies semantic splitting when content exceeds the chunk threshold.

        Args:
            text (str): Raw text segment.
            metadata (dict): Metadata shared by the generated text chunk(s).

        Returns:
            list[Document]: One or more text documents.
        """
        cleaned_text = text.strip()
        if not cleaned_text:
            return []

        text_doc = Document(
            page_content=cleaned_text,
            metadata=metadata | {"type": "text"},
        )
        if len(cleaned_text) > self.chunk_threshold:
            return self.splitter.semantic_splitter.split_documents([text_doc])
        return [text_doc]

    @staticmethod
    def _get_match_group(match: re.Match[str], group_name: str) -> str:
        """
        Safely retrieve a named regex group as a string.

        Args:
            match (re.Match[str]): Regex match object.
            group_name (str): Group name to extract.

        Returns:
            str: Extracted group value, or an empty string when unavailable.
        """
        value = match.group(group_name)
        return value if isinstance(value, str) else ""

    def _build_image_doc(self, match: re.Match[str], metadata: dict, markdown_path: str) -> Document | None:
        """
        Build an image document from a markdown image match.

        Args:
            match (re.Match[str]): Regex match for a markdown image block.
            metadata (dict): Metadata shared by the image chunk.
            markdown_path (str): Source markdown file path.

        Returns:
            Document | None: Image document if the image can be materialized locally, otherwise None.
        """
        image_src, image_title = self._parse_image_target(self._get_match_group(match, "target"))
        image_path = self._materialize_image(image_src, markdown_path)
        if image_path is None:
            logger.warning(f"Skip image block because materialization failed: src={image_src}")
            return None

        return Document(
            page_content=str(image_path),
            metadata=metadata | {
                "type": "image",
                "image_path": str(image_path),
                "image_url": image_src if self._is_url(image_src) else "",
                "image_alt": self._get_match_group(match, "alt"),
                "image_title": image_title,
            },
        )

    @staticmethod
    def _parse_image_target(target: str) -> tuple[str, str]:
        """
        Parse the target portion of a markdown image syntax.

        Behavior:
            - Extracts the actual image source.
            - Extracts the optional image title.
            - Normalizes angle-bracket-wrapped sources.

        Args:
            target (str): Raw content inside markdown image parentheses.

        Returns:
            tuple[str, str]: Parsed image source and optional title.
        """
        cleaned_target = target.strip()
        match = re.match(r'^(?P<src><[^>]+>|.+?)(?:\s+"(?P<title>[^"]*)")?$', cleaned_target)
        if match is None:
            return cleaned_target, ""

        image_src = match.group("src") if isinstance(match.group("src"), str) else cleaned_target
        image_title = match.group("title") if isinstance(match.group("title"), str) else ""
        if image_src.startswith("<") and image_src.endswith(">"):
            image_src = image_src[1:-1].strip()

        return image_src.strip(), image_title

    def _materialize_image(self, image_src: str, markdown_path: str) -> Path | None:
        """
        Convert an image source into a local file path.

        Behavior:
            - Downloads remote images.
            - Resolves local relative image paths against the markdown file path.
            - Copies local absolute-path images.
            - Skips unsupported or invalid paths.

        Args:
            image_src (str): Image source path or URL.
            markdown_path (str): Source markdown file path, used for logging context.

        Returns:
            Path | None: Local image path if successful, otherwise None.
        """
        try:
            if self._is_url(image_src):
                return self._download_image(image_src)

            source_path = self._resolve_image_path(image_src=image_src, markdown_path=markdown_path)
            if source_path is None:
                logger.warning(f"Skip invalid image path: {image_src}")
                return None

            return self._copy_image(source_path)
        except Exception as exc:
            logger.warning(f"Failed to process image '{image_src}' in '{markdown_path}': {exc}")
            return None

    @staticmethod
    def _resolve_image_path(image_src: str, markdown_path: str) -> Path | None:
        """
        Resolve a markdown image path into an absolute local filesystem path.

        Behavior:
            - Returns absolute paths unchanged.
            - Resolves relative paths against the markdown file's parent directory.

        Args:
            image_src (str): Image path extracted from markdown.
            markdown_path (str): Source markdown file path.

        Returns:
            Path | None: Absolute image path, or None when the input is empty.
        """
        cleaned_src = image_src.strip()
        if not cleaned_src:
            return None

        source_path = Path(cleaned_src)
        if source_path.is_absolute():
            return source_path

        return (Path(markdown_path).resolve().parent / source_path).resolve()

    @staticmethod
    def _is_url(value: str) -> bool:
        """
        Determine whether a string is an HTTP or HTTPS URL.

        Args:
            value (str): Input value to check.

        Returns:
            bool: True when the input is a valid HTTP(S) URL.
        """
        parsed = urlparse(value)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def _copy_image(self, source_path: str | Path) -> Path | None:
        """
        Copy a local image into the processor image directory.

        Args:
            source_path (str | Path): Source image file path.

        Returns:
            Path | None: Copied image path, or None when the source does not exist.
        """
        source = Path(source_path)
        if not source.exists() or not source.is_file():
            logger.warning(f"Image file not found: {source}")
            return None

        target = self.images_output_dir / self._build_image_name(source.read_bytes(), source.suffix)
        if not target.exists():
            shutil.copy2(source, target)
        return target

    def _download_image(self, image_url: str) -> Path:
        """
        Download an image from a URL into the processor image directory.

        Args:
            image_url (str): Remote image URL.

        Returns:
            Path: Local path of the downloaded image.
        """
        with urlopen(image_url, timeout=30) as response:
            image_bytes = response.read()

        suffix = Path(urlparse(image_url).path).suffix or ".img"
        target = self.images_output_dir / self._build_image_name(image_bytes, suffix)
        if not target.exists():
            target.write_bytes(image_bytes)
        return target

    @staticmethod
    def _build_image_name(content: bytes, suffix: str) -> str:
        """
        Build a stable local filename for an image.

        Args:
            content (bytes): Raw image bytes.
            suffix (str): Original or inferred file suffix.

        Returns:
            str: Normalized image filename based on content hash.
        """
        normalized_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        return f"{hashlib.md5(content).hexdigest()}{normalized_suffix.lower()}"

    def _add_heading_hierarchy(self, docs: list[Document]) -> list[Document]:
        """
        Propagate heading hierarchy metadata across ordered documents.

        Behavior:
            - Tracks the latest non-empty heading value for each configured level.
            - Clears lower levels when a higher-level heading changes.
            - Ensures every document carries a complete heading context.

        Args:
            docs (list[Document]): Documents produced before hierarchy normalization.

        Returns:
            list[Document]: Documents with normalized heading metadata.
        """
        processed_docs: list[Document] = []
        current_titles = {header_key: "" for header_key in self.header_keys}

        for doc in docs:
            new_metadata = doc.metadata.copy()

            updated_level = None
            for level, header_key in enumerate(self.header_keys, start=1):
                header_value = new_metadata.get(header_key)
                if header_value:
                    current_titles[header_key] = header_value
                    updated_level = level

            if updated_level is not None:
                for header_key in self.header_keys[updated_level:]:
                    current_titles[header_key] = ""

            for header_key in self.header_keys:
                new_metadata[header_key] = current_titles[header_key]

            processed_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata=new_metadata,
                )
            )

        return processed_docs

    def _finalize_documents(self, docs: list[Document]) -> list[Document]:
        """
        Finalize chunk metadata and display text for downstream use.

        Behavior:
            - Assigns sequential chunk indices and chunk IDs.
            - Builds heading hierarchy text.
            - Normalizes code, table, and image metadata fields.
            - Rewrites page content into the retrieval-ready text form.

        Args:
            docs (list[Document]): Ordered documents after heading normalization.

        Returns:
            list[Document]: Fully finalized documents.
        """
        processed_docs: list[Document] = []
        for chunk_index, doc in enumerate(docs, start=1):
            new_metadata = doc.metadata.copy()
            heading_hierarchy = self._build_heading_hierarchy_text(new_metadata)
            doc_type = str(new_metadata.get("type", "text"))
            code = str(new_metadata.get("code", "")) if doc_type == "code" else ""
            table = str(new_metadata.get("table", "")) if doc_type == "table" else ""
            formatted_page_content = self._build_text_content(
                page_content=doc.page_content,
                heading_hierarchy=heading_hierarchy,
                doc_type=doc_type,
                image_path=str(new_metadata.get("image_path", "")),
                code=code,
                table=table,
            )

            new_metadata.update({
                "chunk_index": chunk_index,
                "chunk_id": f"chunk_{chunk_index:05d}",
                "heading_hierarchy": heading_hierarchy,
                "image_path": str(new_metadata.get("image_path", "")) if doc_type == "image" else "",
                "code": code,
                "table": table,
            })

            processed_docs.append(
                Document(
                    page_content=formatted_page_content,
                    metadata=new_metadata,
                )
            )

        return processed_docs

    def _build_heading_hierarchy_text(self, metadata: dict) -> str:
        """
        Build a heading hierarchy string from document metadata.

        Args:
            metadata (dict): Document metadata containing heading levels.

        Returns:
            str: Heading hierarchy joined with arrows.
        """
        headers = []
        for header_key in self.header_keys:
            header_value = str(metadata.get(header_key, "")).strip()
            if header_value:
                headers.append(header_value)
        return " -> ".join(headers)

    @staticmethod
    def _build_text_content(
            page_content: str,
            heading_hierarchy: str,
            doc_type: str,
            image_path: str,
            code: str,
            table: str,
    ) -> str:
        """
        Build the final page content used for retrieval and later explanation.

        Args:
            page_content (str): Original document content.
            heading_hierarchy (str): Resolved heading hierarchy string.
            doc_type (str): Chunk type.
            image_path (str): Local image path for image chunks.
            code (str): Raw code content for code chunks.
            table (str): Raw table content for table chunks.

        Returns:
            str: Final formatted chunk text.
        """
        prefix = f"{heading_hierarchy}\n" if heading_hierarchy else ""
        if doc_type == "image":
            return f"{prefix}{image_path}".strip()
        if doc_type == "code":
            return f"{prefix}{code}".strip()
        if doc_type == "table":
            return f"{prefix}{table}".strip()

        content = page_content.strip()
        return f"{prefix}{content}" if prefix else content

    def process_file(self, markdown_path: str | Path) -> list[Document]:
        """
        Process a single markdown file into ordered chunk documents.

        Args:
            markdown_path (str | Path): Input markdown file path.

        Returns:
            list[Document]: Processed chunk documents.
        """
        markdown_path = Path(markdown_path)
        markdown_content = markdown_path.read_text(encoding="utf-8")
        return self.process_content(markdown_content, str(markdown_path))

    def process_dir(self, markdown_dir: str | Path) -> list[Document]:
        """
        Process all markdown files in a directory.

        Args:
            markdown_dir (str | Path): Directory containing markdown files.

        Returns:
            list[Document]: Concatenated documents from all processed files.
        """
        markdown_paths = list_files_with_allowed_extensions(
            data_path=markdown_dir,
            allowed_types=(".md", ".markdown"),
            recursive=False,
        )

        docs: list[Document] = []
        for markdown_path in sorted(markdown_paths):
            docs.extend(self.process_file(markdown_path))

        return docs
