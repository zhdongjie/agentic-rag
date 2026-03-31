from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings

from core.config.models_config import EMBEDDING_MODEL, EMBEDDING_API_KEY


def create_embedding() -> Embeddings:
    return DashScopeEmbeddings(
        model=EMBEDDING_MODEL,
        dashscope_api_key=EMBEDDING_API_KEY,
    )
