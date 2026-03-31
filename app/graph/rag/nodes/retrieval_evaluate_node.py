import json

from app.graph.rag.prompts import RETRIEVAL_RELEVANCE_PROMPT
from app.graph.rag.schemas import RetrievalRelevanceEvaluation
from app.graph.rag.state import RagAgentState
from app.graph.rag.utils import format_retrieved_docs
from models import llm

retrieval_evaluate_chain = RETRIEVAL_RELEVANCE_PROMPT | llm.with_structured_output(
    RetrievalRelevanceEvaluation,
    method="function_calling",
)


def retrieval_evaluate_node(state: RagAgentState):
    query = state["messages"][-1].content

    retrieved_docs = state.get("retrieved_docs", [])
    query_transform_result = state.get("query_transform_result", {})

    result = retrieval_evaluate_chain.invoke({
        "query": query,
        "query_transform_result": json.dumps(query_transform_result, ensure_ascii=False),
        "retrieved_docs": format_retrieved_docs(retrieved_docs),
    })

    return {"document_relevance_evaluation": result.model_dump()}
