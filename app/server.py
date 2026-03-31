from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langgraph.graph.message import MessagesState

from app.graph.rag.graph import rag_graph

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs() -> RedirectResponse:
    return RedirectResponse("/docs")


add_routes(
    app,
    rag_graph,
    path="/chat",
    input_type=MessagesState,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
