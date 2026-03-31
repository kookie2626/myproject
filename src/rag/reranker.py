from __future__ import annotations

import re
from typing import List, Tuple

from langchain_core.documents import Document


def _tokenize_ko_en(text: str) -> set[str]:
    # 한글/영문/숫자 토큰만 남겨 리랭킹 점수 잡음을 줄인다.
    tokens = re.findall(r"[가-힣A-Za-z0-9]{2,}", text.lower())
    return set(tokens)


def _score_doc(query: str, doc: Document) -> float:
    q = _tokenize_ko_en(query)
    d = _tokenize_ko_en(doc.page_content)
    if not q or not d:
        return 0.0

    overlap = len(q & d) / max(len(q), 1)

    # 정책 정보성 문서를 우선: 출처 URL/공고 ID가 있으면 약한 가중치 부여
    meta = getattr(doc, "metadata", {}) or {}
    has_url = 1.0 if str(meta.get("source_url", "")).strip() else 0.0
    has_notice = 1.0 if str(meta.get("notice_id", "")).strip() else 0.0
    meta_boost = 0.03 * has_url + 0.02 * has_notice
    return float(overlap + meta_boost)


def rerank_documents(
    query: str,
    documents: List[Document],
    top_n: int = 6,
    threshold: float = 0.08,
) -> Tuple[List[Document], bool, float]:
    """
    경량 리랭커: 토큰 중첩 기반 점수 + 메타데이터 보정으로 재정렬.
    반환: (문서목록, 임계값 통과여부, 최고점)
    """
    if not documents:
        return [], False, 0.0

    scored = sorted(
        [(_score_doc(query, doc), doc) for doc in documents],
        key=lambda x: x[0],
        reverse=True,
    )

    top_score = float(scored[0][0]) if scored else 0.0
    passed = [(score, doc) for score, doc in scored if score >= threshold]

    if not passed:
        return [], False, top_score

    top_docs = [doc for _, doc in passed[:top_n]]
    return top_docs, True, top_score
