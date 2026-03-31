from app.graph.rag.async_recording import (
    submit_async_answer_record,
    submit_async_retrieval_record,
)
from app.graph.rag.prompts import ANSWER_PROMPT
from app.graph.rag.state import RagAgentState
from app.graph.rag.utils import format_retrieved_docs
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

    submit_async_answer_record(
        query=query,
        query_transform_result=query_transform_result,
        retrieved_docs=retrieved_docs,
        document_relevance_evaluation=document_relevance_evaluation,
        answer=message.content,
    )

    return {"messages": [message]}
