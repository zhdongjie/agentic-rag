from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document


def text_loader(file_path) -> list[Document]:
    return TextLoader(file_path=file_path, encoding="utf-8", ).load()
