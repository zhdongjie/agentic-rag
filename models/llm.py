from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from core.config.models_config import LLM_MODEL, LLM_BASE_URL, LLM_API_KEY


def create_llm(
        *,
        enable_thinking: bool | None = None,
) -> BaseChatModel:
    extra_body = None
    if enable_thinking is not None:
        extra_body = {"enable_thinking": enable_thinking}

    return ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        extra_body=extra_body,
    )
