from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from src.config import settings
from src.data.pdf_preprocessor import (
    export_processed_chunks,
    load_pdf_documents,
    load_web_json_documents,
    split_documents,
)


class IndexBuilder:
    def __init__(self) -> None:
        self.embeddings = OpenAIEmbeddings(model=settings.embedding_model)
        self.persist_dir = settings.vector_db_dir

    def build(self) -> tuple[int, str]:
        raw_docs = load_pdf_documents(settings.raw_docs_dir) + load_web_json_documents(settings.raw_docs_dir)
        if not raw_docs:
            raise ValueError("data/raw 폴더에 PDF 또는 JSON 데이터가 없습니다.")

        chunks = split_documents(raw_docs)
        preview_path = export_processed_chunks(chunks, settings.processed_docs_dir)

        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        _ = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name="startup_support_docs",
        )
        return len(chunks), preview_path

    def load_vectorstore(self) -> Chroma:
        return Chroma(
            collection_name="startup_support_docs",
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )


def docs_to_tokenized_corpus(docs: List[Document]) -> List[List[str]]:
    return [doc.page_content.lower().split() for doc in docs]
