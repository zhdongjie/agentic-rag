from typing import Any

from langchain_core.documents import Document
from langgraph.graph.message import MessagesState


class RagAgentState(MessagesState):
    rewritten_query: str
    sub_queries: list[str]
    query_transform_result: dict[str, Any]
    retrieved_docs: list[Document]
    document_relevance_evaluation: dict[str, Any]
