from app.graph.rag.state import RagAgentState
from app.vectorstores.milvus_vector_store import MilvusVectorStore


def retrieval_node(state: RagAgentState):
    vector_store = MilvusVectorStore().vector_store
    queries = _build_queries(state)

    merged_docs: dict[str, tuple] = {}

    for query in queries:
        docs_with_scores = vector_store.similarity_search_with_score(
            query=query,
            k=20,
            fetch_k=50,
            ranker_type="rrf",
            ranker_params={"k": 60},
        )

        for doc, score in docs_with_scores:
            chunk_id = doc.metadata.get("chunk_id", "")
            existing = merged_docs.get(chunk_id)
            if existing is None or score > existing[1]:
                doc.metadata["score"] = score
                doc.metadata["matched_query"] = query
                merged_docs[chunk_id] = (doc, score)

    sorted_docs = [
        doc for doc, _ in sorted(merged_docs.values(), key=lambda item: item[1], reverse=True)
    ]

    return {"retrieved_docs": sorted_docs}


def _build_queries(state: RagAgentState) -> list[str]:
    query = state["messages"][-1].content
    rewritten_query = state.get("rewritten_query")
    sub_queries = state.get("sub_queries", [])

    merged_queries = [query, rewritten_query, *sub_queries]
    deduplicated_queries = []
    seen = set()

    for item in merged_queries:
        if not item or item in seen:
            continue
        seen.add(item)
        deduplicated_queries.append(item)

    return deduplicated_queries or [query]
