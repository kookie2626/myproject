from __future__ import annotations

import json

import streamlit as st
from langchain_core.documents import Document

from src.eval.run_eval import run_eval
from src.ingest.build_index import IndexBuilder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.rag.qa_chain import answer_with_citations
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
            answer = answer_with_citations(question, retrieval_result.documents)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("### 답변")
                st.write(answer)
            with col2:
                st.markdown("### 적용 필터")
                st.code(json.dumps(retrieval_result.applied_filters, ensure_ascii=False, indent=2), language="json")
                st.markdown("### 검색 문서 수")
                st.write(len(retrieval_result.documents))
        except Exception as exc:
            st.error(f"질의 실행 실패: {exc}")
