from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter

from models import embedding


class MarkdownSplitter:
    def __init__(
            self,
            semantic_splitter_model: Embeddings = embedding
    ):
        """
        Initialize the MarkdownSplitter with predefined header rules and semantic splitting strategy.

        Behavior:
            - Defines markdown header levels from h1 to h6.
            - Initializes a header-based splitter to segment text by structure.
            - Initializes a semantic splitter using embedding-based similarity.

        Semantic Splitter Configuration:
            - breakpoint_threshold_type="percentile":
                Determines chunk boundaries based on percentile thresholds of similarity scores.
            - breakpoint_threshold_amount=95:
                Uses the 95th percentile as the threshold for splitting, meaning only
                significant semantic changes trigger a split.

        Dependencies:
            - `MarkdownHeaderTextSplitter` for structural splitting.
            - `SemanticChunker` for semantic segmentation.
            - `embedding` object must be defined and provide embedding vectors.
        """
        self.headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
            ("#####", "h5"),
            ("######", "h6"),
        ]
        self.md_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )
        self.semantic_splitter = SemanticChunker(
            embeddings=semantic_splitter_model,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
        )
