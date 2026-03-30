from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.config import settings


SYSTEM_PROMPT = """
당신은 창업지원사업 전문 QA 어시스턴트입니다.
반드시 제공된 문서 컨텍스트 내부 정보만 사용해 답변하세요.
모르면 모른다고 답변하고 추측하지 마세요.
답변 마지막에 반드시 출처를 아래 형식으로 제시하세요.
- 출처: 파일명 p.페이지번호
""".strip()


def _format_context(docs: List[Document]) -> str:
    lines = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page_number", "?")
        lines.append(f"[문서{i}] ({source} p.{page})\\n{doc.page_content}")
    return "\n\n".join(lines)


def answer_with_citations(query: str, docs: List[Document]) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "질문: {query}\n\n아래 컨텍스트만 사용해서 답변하세요:\n{context}\n\n"
                "정책/지원금 조건은 항목별로 정리하고, 마지막에 출처를 나열하세요.",
            ),
        ]
    )

    llm = ChatOpenAI(model=settings.openai_model, temperature=0)
    chain = prompt | llm
    result = chain.invoke({"query": query, "context": _format_context(docs)})
    return result.content
