from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from core.config.models_config import VLM_API_KEY, VLM_BASE_URL, VLM_MODEL


def create_vlm(*, enable_thinking: bool | None = None) -> BaseChatModel:
    extra_body = None
    if enable_thinking is not None:
        extra_body = {"enable_thinking": enable_thinking}
    return ChatOpenAI(
        model=VLM_MODEL,
        base_url=VLM_BASE_URL,
        api_key=VLM_API_KEY,
        extra_body=extra_body,
    )
