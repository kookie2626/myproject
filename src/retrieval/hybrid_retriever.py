from __future__ import annotations

from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.config import settings


@dataclass
class RetrievalResult:
    documents: List[Document]


class HybridRetriever:
    def __init__(self, vectorstore, base_docs: List[Document]) -> None:
        self.vectorstore = vectorstore
        self.base_docs = base_docs
        self.corpus = [d.page_content.lower().split() for d in base_docs]
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None

    def _bm25_search(self, query: str, top_k: int) -> List[Document]:
        if not self.bm25:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.base_docs[idx] for idx in top_indices]

    def _vector_search(self, query: str, top_k: int) -> List[Document]:
        return self.vectorstore.similarity_search(query=query, k=top_k)

    def retrieve(self, query: str) -> RetrievalResult:
        bm25_docs = self._bm25_search(query, settings.top_k_bm25)
        vector_docs = self._vector_search(query, settings.top_k_vector)

        dedup = {}
        for d in bm25_docs + vector_docs:
            key = (
                d.metadata.get("source_file", "unknown"),
                d.metadata.get("page_number", -1),
                d.page_content[:80],
            )
            dedup[key] = d

        return RetrievalResult(documents=list(dedup.values()))
