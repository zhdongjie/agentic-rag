from app.graph.rag.state import RagAgentState
from models import rerank


def rerank_node(state: RagAgentState):
    query = state["messages"][-1].content
    retrieved_docs = state.get("retrieved_docs", [])

    if not retrieved_docs:
        return {"retrieved_docs": []}

    reranked_docs = rerank.compress_documents(
        documents=retrieved_docs,
        query=query,
    )

    return {"retrieved_docs": list(reranked_docs)}
