from pymilvus import MilvusClient

from core.config.milvus_config import MILVUS_URI, MILVUS_TOKEN


def get_client() -> MilvusClient:
    return MilvusClient(
        uri=MILVUS_URI,
        token=MILVUS_TOKEN,
    )
