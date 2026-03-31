from pymilvus import CollectionSchema

from core.utils.logger_utils import logger
from infra.milvus.client import get_client


def create_database(db_name: str) -> None:
    """
    Create the Milvus database; create it if it does not exist.

    Args:
        db_name: Database name
    """
    client = get_client()

    existing_dbs = client.list_databases()
    if db_name not in existing_dbs:
        client.create_database(db_name)
        logger.info(f"Database '{db_name}' created.")
    else:
        logger.info(f"Database '{db_name}' already exists.")


def create_collection(
        db_name: str,
        collection_name: str,
        schema: CollectionSchema,
        index_params=None,
) -> None:
    """
    Create a Milvus Collection if not exists.

    Args:
        db_name: Database name
        collection_name: Collection name
        schema: Collection schema
        index_params: Collection index params
    """
    client = get_client()

    # check service
    existing_dbs = client.list_databases()
    if db_name not in existing_dbs:
        logger.error(f"Database '{db_name}' not exists.")
        return

    client.using_database(db_name)

    # check collection
    if client.has_collection(collection_name):
        logger.info(f"Collection '{collection_name}' already exists.")
        return

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
    )

    logger.info(f"Collection '{collection_name}' created successfully.")
