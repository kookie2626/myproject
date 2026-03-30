from __future__ import annotations

import json
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf_documents(raw_docs_dir: str) -> List[Document]:
    raw_path = Path(raw_docs_dir)
    if not raw_path.exists():
        return []

    docs: List[Document] = []
    for pdf_path in raw_path.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()
        for page in pages:
            page.metadata["source_file"] = pdf_path.name
            page.metadata["page_number"] = page.metadata.get("page", 0) + 1
        docs.extend(pages)
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(docs)


def export_processed_chunks(chunks: List[Document], processed_docs_dir: str) -> str:
    processed_dir = Path(processed_docs_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    export_path = processed_dir / "chunks_preview.json"

    serialized = []
    for idx, chunk in enumerate(chunks):
        serialized.append(
            {
                "id": idx,
                "text": chunk.page_content,
                "metadata": chunk.metadata,
            }
        )

    export_path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(export_path)
