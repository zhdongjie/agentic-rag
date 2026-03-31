from pymilvus import CollectionSchema, DataType, FieldSchema, Function, FunctionType

from infra.milvus.client import get_client


def knowledge_chunk_schema(dim: int) -> CollectionSchema:
    """
    Build the hybrid-search collection schema for knowledge chunks.

    Fields:
        - dense_vector: semantic embedding vector
        - sparse_vector: BM25 sparse vector generated from text

    Args:
        dim (int): Dense vector dimension.

    Returns:
        CollectionSchema: Milvus collection schema with BM25 function attached.
    """
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="doc_hash", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="heading_hierarchy", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="raw_data", dtype=DataType.VARCHAR, max_length=65535, nullable=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True, enable_match=True, ),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="created_at", dtype=DataType.INT64),
    ]

    functions = [
        Function(
            name="text_bm25_fn",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse_vector"],
        )
    ]

    return CollectionSchema(
        fields=fields,
        functions=functions,
    )


def knowledge_chunk_index_params():
    """
    Build index params for hybrid retrieval.

    Indexes:
        - dense_vector: AUTOINDEX + COSINE
        - sparse_vector: SPARSE_INVERTED_INDEX + BM25

    Returns:
        IndexParams: Milvus index parameter object.
    """
    client = get_client()
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_vector_idx",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_vector_idx",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={
            "inverted_index_algo": "DAAT_MAXSCORE",
            "bm25_k1": 1.2,
            "bm25_b": 0.75
        }
    )
    return index_params
