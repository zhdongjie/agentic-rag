from langgraph.config import get_stream_writer

from app.graph.rag.async_recording import submit_async_retrieval_record
from app.graph.rag.state import RagAgentState


def record_retrieval_issue_node(state: RagAgentState):
    query = state["messages"][-1].content

    query_transform_result = state.get("query_transform_result", {})
    retrieved_docs = state.get("retrieved_docs", [])
    document_relevance_evaluation = state.get("document_relevance_evaluation", {})

    submit_async_retrieval_record(
        query=query,
        query_transform_result=query_transform_result,
        retrieved_docs=retrieved_docs,
        document_relevance_evaluation=document_relevance_evaluation,
    )

    answer = "当前没有检索到足够相关的资料来支持回答。这个问题已记录，后续会人工排查处理。"
    stream_writer = get_stream_writer()
    for char in answer:
        stream_writer({
            "event": "answer_token",
            "content": char,
        })

    stream_writer({
        "event": "answer_done",
        "content": answer,
    })

    return {}
