from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.ingest.build_index import IndexBuilder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.rag.qa_chain import answer_with_citations


@dataclass
class EvalCase:
    question: str
    must_include: List[str]


EVAL_CASES = [
    EvalCase(
        question="만 39세 서울 거주 예비 창업자가 받을 수 있는 지원금은?",
        must_include=["출처", "서울", "예비"],
    ),
    EvalCase(
        question="예비창업패키지와 청년창업사관학교 지원 항목 차이점을 알려줘.",
        must_include=["출처", "차이", "지원"],
    ),
]


def run_eval() -> list[dict]:
    builder = IndexBuilder()
    vectorstore = builder.load_vectorstore()
    base_docs = vectorstore.get(include=["documents", "metadatas"])

    docs = []
    for text, meta in zip(base_docs["documents"], base_docs["metadatas"]):
        from langchain_core.documents import Document

        docs.append(Document(page_content=text, metadata=meta))

    retriever = HybridRetriever(vectorstore=vectorstore, base_docs=docs)

    report = []
    for case in EVAL_CASES:
        retrieved = retriever.retrieve(case.question).documents
        answer = answer_with_citations(case.question, retrieved)
        passed = all(keyword in answer for keyword in case.must_include)
        report.append(
            {
                "question": case.question,
                "passed": passed,
                "must_include": case.must_include,
            }
        )
    return report
