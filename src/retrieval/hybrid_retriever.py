from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.config import settings


@dataclass
class RetrievalResult:
    documents: List[Document]
    applied_filters: dict


@dataclass
class QueryProfile:
    regions: List[str]
    stages: List[str]
    age: int | None


def _parse_query_profile(query: str) -> QueryProfile:
    regions = [r for r in ["서울", "경기", "인천", "부산", "대구", "광주", "대전", "울산", "세종"] if r in query]
    stages = [s for s in ["예비", "초기", "도약", "재도전"] if s in query]

    age_match = re.search(r"만\s*(\d{1,2})\s*세", query)
    age = int(age_match.group(1)) if age_match else None
    return QueryProfile(regions=regions, stages=stages, age=age)


def _age_to_bucket(age: int | None) -> str | None:
    if age is None:
        return None
    if age <= 39:
        return "youth"
    return "all"


def _is_doc_match(profile: QueryProfile, doc: Document) -> bool:
    regions_meta = doc.metadata.get("regions", "")
    stages_meta = doc.metadata.get("stages", "")
    age_bucket_meta = doc.metadata.get("age_bucket", "all")
    content = doc.page_content

    if profile.regions:
        if not any(region in regions_meta or region in content for region in profile.regions):
            return False

    if profile.stages:
        if not any(stage in stages_meta or stage in content for stage in profile.stages):
            return False

    age_bucket = _age_to_bucket(profile.age)
    if age_bucket == "youth" and age_bucket_meta == "senior":
        return False

    return True


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
        profile = _parse_query_profile(query)
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

        merged_docs = list(dedup.values())
        filtered_docs = [d for d in merged_docs if _is_doc_match(profile, d)]

        return RetrievalResult(
            documents=filtered_docs if filtered_docs else merged_docs,
            applied_filters={
                "regions": profile.regions,
                "stages": profile.stages,
                "age": profile.age,
                "filter_hit": bool(filtered_docs),
            },
        )
