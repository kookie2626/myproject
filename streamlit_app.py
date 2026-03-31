from __future__ import annotations

import json

import streamlit as st
from langchain_core.documents import Document

from src.eval.run_eval import run_eval
from src.ingest.build_index import IndexBuilder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.rag.qa_chain import answer_with_citations
from src.rag.reranker import rerank_documents
from src.config import settings
from src.data.web_collectors import collect_and_save_default


st.set_page_config(page_title="창업지원 RAG 데모", page_icon="📄", layout="wide")
st.title("창업지원 문서 RAG 데모")
st.caption("웹 수집 → 인덱스 생성 → 질의응답까지 한 화면에서 실행")


def _load_base_docs(vectorstore) -> list[Document]:
    fetched = vectorstore.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(fetched["documents"], fetched["metadatas"])
    ]


with st.sidebar:
    st.header("파이프라인 실행")

    if st.button("1) 웹 데이터 수집", use_container_width=True):
        try:
            output_path = f"{settings.raw_docs_dir}/web_seed.json"
            count, saved_path = collect_and_save_default(output_path)
            st.success(f"수집 완료: {count}건")
            st.write(saved_path)
        except Exception as exc:
            st.error(f"수집 실패: {exc}")

    if st.button("2) 인덱스 생성", use_container_width=True):
        try:
            builder = IndexBuilder()
            chunk_count, preview_path = builder.build()
            st.success(f"인덱스 생성 완료: {chunk_count} chunks")
            st.write(preview_path)
        except Exception as exc:
            st.error(f"인덱스 생성 실패: {exc}")

    if st.button("3) 평가 실행", use_container_width=True):
        try:
            report = run_eval()
            st.success("평가 완료")
            st.json(report)
        except Exception as exc:
            st.error(f"평가 실패: {exc}")

st.subheader("질의응답")
question = st.text_input(
    "질문 입력",
    value="만 39세 서울 거주 예비 창업자가 받을 수 있는 지원금은?",
)

if st.button("질문 실행", type="primary"):
    if not question.strip():
        st.warning("질문을 입력해주세요.")
    else:
        try:
            builder = IndexBuilder()
            vectorstore = builder.load_vectorstore()
            docs = _load_base_docs(vectorstore)

            retriever = HybridRetriever(vectorstore=vectorstore, base_docs=docs)
            retrieval_result = retriever.retrieve(question)
            reranked_docs, rerank_ok, rerank_top_score = rerank_documents(
                query=question,
                documents=retrieval_result.documents,
                top_n=settings.rerank_top_n,
                threshold=settings.rerank_threshold,
            )
            final_docs = reranked_docs if rerank_ok else retrieval_result.documents
            answer = answer_with_citations(question, final_docs)

            if not final_docs:
                st.warning("검색된 근거 문서가 없어 안전 모드로 답변을 제한했습니다.")

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("### 답변")
                st.write(answer)
            with col2:
                st.markdown("### 적용 필터")
                st.code(json.dumps(retrieval_result.applied_filters, ensure_ascii=False, indent=2), language="json")
                st.markdown("### 리랭커")
                st.code(
                    json.dumps(
                        {
                            "rerank_ok": rerank_ok,
                            "rerank_top_score": round(rerank_top_score, 4),
                            "threshold": settings.rerank_threshold,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    language="json",
                )
                st.markdown("### 검색 문서 수")
                st.write(len(final_docs))

            st.markdown("### 출처 미리보기")
            for idx, doc in enumerate(final_docs[:6], start=1):
                meta = doc.metadata
                source_file = meta.get("source_file", "unknown")
                source_url = meta.get("source_url", "")
                notice_id = meta.get("notice_id", "")
                page_number = meta.get("page_number", "?")
                st.markdown(
                    f"{idx}. {source_file} p.{page_number}"
                    + (f" | notice_id={notice_id}" if notice_id else "")
                    + (f" | {source_url}" if source_url else "")
                )
        except Exception as exc:
            st.error(f"질의 실행 실패: {exc}")
