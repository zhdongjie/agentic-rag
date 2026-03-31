from langchain_milvus import BM25BuiltInFunction, Milvus

from core.config.milvus_config import (
    MILVUS_DB,
    MILVUS_DOCS_COLLECTION,
    MILVUS_TOKEN,
    MILVUS_URI,
)
from models import embedding


class MilvusVectorStore:
    def __init__(self):
        self.vector_store = Milvus(
            embedding_function=embedding,
            builtin_function=BM25BuiltInFunction(
                input_field_names="text",
                output_field_names="sparse_vector",
                enable_match=True,
            ),
            connection_args={
                "uri": MILVUS_URI,
                "token": MILVUS_TOKEN,
                "db_name": MILVUS_DB,
            },
            collection_name=MILVUS_DOCS_COLLECTION,
            vector_field=["dense_vector", "sparse_vector"],
            text_field="text",
            search_params=[
                {
                    "metric_type": "COSINE",
                    "params": {},
                },
                {
                    "metric_type": "BM25",
                    "params": {},
                },
            ],
            auto_id=True,
        )
