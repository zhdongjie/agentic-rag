import threading

from app.graph.rag.prompts import ANSWER_EVALUATION_PROMPT
from app.graph.rag.schemas import AnswerEvaluation
from app.graph.rag.utils import format_retrieved_docs
from core.utils.logger_utils import logger
from models import llm

answer_evaluation_chain = ANSWER_EVALUATION_PROMPT | llm.with_structured_output(
    AnswerEvaluation,
    method="function_calling",
)


def submit_async_retrieval_record(
        query: str,
        query_transform_result: dict,
        retrieved_docs: list,
        document_relevance_evaluation: dict,
) -> None:
    payload = {
        "query": query,
        "query_transform_result": query_transform_result,
        "retrieved_doc": retrieved_docs,
        "retrieved_doc_count": len(retrieved_docs),
        "document_relevance_evaluation": document_relevance_evaluation,
    }
    _start_background_task(_record_retrieval_evaluation, payload)


def submit_async_answer_record(
        query: str,
        query_transform_result: dict,
        retrieved_docs: list,
        document_relevance_evaluation: dict,
        answer: str,
) -> None:
    payload = {
        "query": query,
        "query_transform_result": query_transform_result,
        "retrieved_docs": retrieved_docs,
        "retrieved_doc_count": len(retrieved_docs),
        "document_relevance_evaluation": document_relevance_evaluation,
        "answer": answer,
    }
    _start_background_task(_evaluate_and_record_answer, payload)


def _start_background_task(target, payload: dict) -> None:
    thread = threading.Thread(
        target=target,
        args=(payload,),
        daemon=True,
    )
    thread.start()


def _record_retrieval_evaluation(payload: dict) -> None:
    try:
        evaluation = payload.get("document_relevance_evaluation", {})
        passed = bool(evaluation.get("passed", False))
        stage = "retrieval_relevance_passed" if passed else "retrieval_relevance_failed"
        record = payload | {"stage": stage}

        if passed:
            logger.info(f"[async_retrieval_evaluation] record={record}")
        else:
            logger.warning(f"[async_retrieval_evaluation] record={record}")
    except Exception as exc:
        logger.exception(f"[async_retrieval_evaluation] failed: {exc}")


def _evaluate_and_record_answer(payload: dict) -> None:
    try:
        result = answer_evaluation_chain.invoke({
            "query": payload["query"],
            "retrieved_docs": format_retrieved_docs(payload.get("retrieved_docs", [])),
            "answer": payload["answer"],
        })

        evaluation = result.model_dump()
        passed = bool(evaluation.get("passed", False))
        stage = "answer_evaluation_passed" if passed else "answer_evaluation_failed"
        record = payload | {"stage": stage, "answer_evaluation": evaluation}

        if passed:
            logger.info(f"[async_answer_evaluation] record={record}")
        else:
            logger.warning(f"[async_answer_evaluation] record={record}")
    except Exception as exc:
        logger.exception(f"[async_answer_evaluation] failed: {exc}")
