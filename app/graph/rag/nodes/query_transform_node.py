from app.graph.rag.prompts import QUERY_TRANSFORM_PROMPT
from app.graph.rag.schemas import QueryTransformResult
from app.graph.rag.state import RagAgentState
from models import llm

query_transform_chain = QUERY_TRANSFORM_PROMPT | llm.with_structured_output(
    QueryTransformResult,
    method="function_calling",
)


def query_transform_node(state: RagAgentState):
    query = state["messages"][-1].content

    result = query_transform_chain.invoke({
        "query": query,
    })
    result_dict = result.model_dump()

    rewritten_query = result_dict.get("rewritten_query", query)

    sub_queries = [
        q for q in result_dict.get("sub_queries", [])
        if q not in {query, rewritten_query}
    ]

    normalized_result = result_dict | {
        "rewritten_query": rewritten_query,
        "sub_queries": sub_queries,
    }

    return {
        "rewritten_query": rewritten_query,
        "sub_queries": sub_queries,
        "query_transform_result": normalized_result,
    }
