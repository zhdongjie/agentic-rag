from pathlib import Path

from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.constants import END
from langgraph.graph import START, StateGraph

from app.graph.rag.nodes.answer_question_node import answer_question_node
from app.graph.rag.nodes.query_transform_node import query_transform_node
from app.graph.rag.nodes.record_retrieval_issue_node import record_retrieval_issue_node
from app.graph.rag.nodes.rerank_node import rerank_node
from app.graph.rag.nodes.retrieval_evaluate_node import retrieval_evaluate_node
from app.graph.rag.nodes.retrieval_node import retrieval_node
from app.graph.rag.state import RagAgentState


def route_retrieval_evaluation(state: RagAgentState):
    if state.get("document_relevance_evaluation", {}).get("passed", False):
        return "answer_question_node"
    return "record_retrieval_issue_node"


rag_graph = (
    StateGraph(RagAgentState)
    .add_node(query_transform_node)
    .add_node(retrieval_node)
    .add_node(rerank_node)
    .add_node(retrieval_evaluate_node)
    .add_node(answer_question_node)
    .add_node(record_retrieval_issue_node)
    .add_edge(START, "query_transform_node")
    .add_edge("query_transform_node", "retrieval_node")
    .add_edge("retrieval_node", "rerank_node")
    .add_edge("rerank_node", "retrieval_evaluate_node")
    .add_conditional_edges(
        "retrieval_evaluate_node",
        route_retrieval_evaluation,
        {
            "answer_question_node": "answer_question_node",
            "record_retrieval_issue_node": "record_retrieval_issue_node",
        },
    )
    .add_edge("record_retrieval_issue_node", END)
    .add_edge("answer_question_node", END)
    .compile()
)


def export_rag_graph_image(
        output_path: str | Path | None = None,
        *,
        draw_method: MermaidDrawMethod = MermaidDrawMethod.API,
) -> Path:
    target_path = Path(output_path) if output_path else Path(__file__).with_name("rag_graph.png")
    target_path.parent.mkdir(parents=True, exist_ok=True)

    rag_graph.get_graph().draw_mermaid_png(
        output_file_path=str(target_path),
        draw_method=draw_method,
    )
    return target_path.resolve()


def mian():
    result = rag_graph.stream(
        input={
            "messages": [
                ("human", "什么是多态？"),
            ]
        },
        stream_mode=["values", "updates", "messages", "custom", "checkpoints", "tasks", "debug"],
        version="v2",
    )

    for chunk in result:
        if chunk["type"] == "custom":
            print(chunk["data"]["content"], end="", flush=True)
            continue

        if chunk["type"] == "messages":
            print(chunk["data"][0].content, end="", flush=True)


if __name__ == "__main__":
    # image_path = export_rag_graph_image()
    # print(f"Graph image saved to: {image_path}")

    mian()
