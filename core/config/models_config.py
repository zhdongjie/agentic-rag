from core.utils.path_utils import get_abs_path
from core.utils.yaml_utils import load_yaml

config = load_yaml(get_abs_path("config.yaml"))

LLM_MODEL = config["model"]["llm"]["model"]
LLM_BASE_URL = config["model"]["llm"]["base-url"]
LLM_API_KEY = config["model"]["llm"]["api-key"]

EMBEDDING_BASE_URL = config["model"]["embedding"]["provider"]
EMBEDDING_MODEL = config["model"]["embedding"]["model"]
EMBEDDING_API_KEY = config["model"]["embedding"]["api-key"]

VLM_MODEL = config["model"]["vlm"]["model"]
VLM_BASE_URL = config["model"]["vlm"]["base-url"]
VLM_API_KEY = config["model"]["vlm"]["api-key"]

RERANK_MODEL = config["model"]["rerank"]["model"]
RERANK_API_KEY = config["model"]["rerank"]["api-key"]
