from core.utils.path_utils import get_abs_path
from core.utils.yaml_utils import load_yaml

config = load_yaml(get_abs_path("config.yaml"))

MILVUS_URI = config["milvus"]["uri"]
MILVUS_TOKEN = config["milvus"]["token"]
MILVUS_DB = config["milvus"]["database"]
MILVUS_EMBEDDING_DIM = config["milvus"]["embedding-dim"]
MILVUS_DOCS_COLLECTION = config["milvus"]["collections"]["docs"]
MILVUS_CONTEXT_COLLECTION = config["milvus"]["collections"]["context"]
