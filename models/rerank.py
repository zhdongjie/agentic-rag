from langchain_community.document_compressors import DashScopeRerank
from langchain_core.documents import BaseDocumentCompressor

from core.config.models_config import RERANK_API_KEY, RERANK_MODEL


def create_rerank(top_n: int = 3) -> BaseDocumentCompressor:
    return DashScopeRerank(
        model=RERANK_MODEL,
        dashscope_api_key=RERANK_API_KEY,
        top_n=top_n,
    )
