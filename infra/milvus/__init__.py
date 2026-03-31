from core.config.milvus_config import MILVUS_DB, MILVUS_DOCS_COLLECTION, MILVUS_EMBEDDING_DIM
from infra.milvus.manager import create_collection, create_database
from infra.milvus.schema import knowledge_chunk_index_params, knowledge_chunk_schema


def init_milvus():
    create_database(MILVUS_DB)

    # Create document chunks collection
    create_collection(
        db_name=MILVUS_DB,
        collection_name=MILVUS_DOCS_COLLECTION,
        schema=knowledge_chunk_schema(MILVUS_EMBEDDING_DIM),
        index_params=knowledge_chunk_index_params(),
    )
    # TODO: Create chat context collection


__all__ = ["init_milvus"]
