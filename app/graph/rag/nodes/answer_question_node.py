import json

from app.graph.rag.async_recording import (
    submit_async_answer_record,
)
from app.graph.rag.prompts import ANSWER_PROMPT
from app.graph.rag.state import RagAgentState
from app.graph.rag.utils import format_retrieved_docs
from core.utils.logger_utils import logger
from models import llm

answer_question_chain = ANSWER_PROMPT | llm


def answer_question_node(state: RagAgentState):
    query = state["messages"][-1].content

    query_transform_result = state.get("query_transform_result", {})
    retrieved_docs = state.get("retrieved_docs", [])
    document_relevance_evaluation = state.get("document_relevance_evaluation", {})

    message = answer_question_chain.invoke({
        "query": query,
        "retrieved_docs": format_retrieved_docs(retrieved_docs),
    })

    questions_list = []
    seen_urls = set()

    for doc in retrieved_docs:
        q_meta = doc.metadata.get("questions")
        if q_meta and q_meta != "[]":
            try:
                parsed_qs = json.loads(q_meta)
                for q in parsed_qs:
                    if q.get("url") not in seen_urls:
                        questions_list.append(q)
                        seen_urls.add(q.get("url"))
            except Exception as e:
                doc_id = doc.metadata.get("doc_id", "Unknown")
                logger.exception(f"解析试题元数据失败: {e}, doc_id: {doc_id}, 异常数据: {q_meta}")
                continue
    message.additional_kwargs["questions"] = questions_list

    submit_async_answer_record(
        query=query,
        query_transform_result=query_transform_result,
        retrieved_docs=retrieved_docs,
        document_relevance_evaluation=document_relevance_evaluation,
        answer=message.content,
    )

    return {
        "messages": [message],
        "questions": questions_list
    }
