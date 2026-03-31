from langchain_core.documents import Document


def format_retrieved_docs(retrieved_docs: list[Document] | None) -> str:
    if not retrieved_docs:
        return "No retrieved documents."

    parts = []
    for index, doc in enumerate(retrieved_docs, start=1):
        parts.append(_format_single_doc(index, doc))
    return "\n\n".join(parts)


def _format_single_doc(index: int, doc: Document) -> str:
    metadata = doc.metadata
    doc_type = str(metadata.get("type", "text"))
    heading_hierarchy = str(metadata.get("heading_hierarchy", ""))
    chunk_id = str(metadata.get("chunk_id", ""))
    score = metadata.get("score", "")

    lines = [
        f"[doc_{index}]",
        f"chunk_id={chunk_id}",
        f"type={doc_type}",
        f"score={score}",
        f"heading_hierarchy={heading_hierarchy}",
    ]

    if doc_type in {"code", "table", "image"}:
        lines.append("description:")
        lines.append(doc.page_content)
        lines.append(_get_raw_content(doc))
    else:
        lines.append("content:")
        lines.append(doc.page_content)

    return "\n".join(lines)


def _get_raw_content(doc: Document) -> str:
    metadata = doc.metadata
    raw_data = metadata.get("raw_data", "")
    return str(raw_data)
