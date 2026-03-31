from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_core.documents import Document

from src.eval.run_eval import run_eval
from src.config import settings
from src.data.web_collectors import collect_and_save_default
from src.ingest.build_index import IndexBuilder
from src.retrieval.hybrid_retriever import HybridRetriever
from src.rag.qa_chain import answer_with_citations


def _load_base_docs(vectorstore) -> list[Document]:
    fetched = vectorstore.get(include=["documents", "metadatas"])
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(fetched["documents"], fetched["metadatas"])
    ]


def command_build_index() -> None:
    builder = IndexBuilder()
    chunk_count, preview_path = builder.build()
    print(f"[완료] 인덱스 생성: {chunk_count} chunks")
    print(f"[완료] 전처리 미리보기: {preview_path}")


def command_collect_web() -> None:
    output_path = str(Path(settings.raw_docs_dir) / "web_seed.json")
    count, saved_path = collect_and_save_default(output_path)
    print(f"[완료] 웹 데이터 수집: {count}건")
    print(f"[완료] 저장 파일: {saved_path}")


def command_ask(question: str) -> None:
    builder = IndexBuilder()
    vectorstore = builder.load_vectorstore()
    docs = _load_base_docs(vectorstore)

    retriever = HybridRetriever(vectorstore=vectorstore, base_docs=docs)
    retrieval_result = retriever.retrieve(question)
    retrieved_docs = retrieval_result.documents
    answer = answer_with_citations(question, retrieved_docs)

    print("\n=== 질문 ===")
    print(question)
    print("\n=== 적용 필터 ===")
    print(json.dumps(retrieval_result.applied_filters, ensure_ascii=False, indent=2))
    print("\n=== 답변 ===")
    print(answer)


def command_eval() -> None:
    report = run_eval()
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="창업지원 문서 RAG 시스템")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("build-index", help="PDF 문서를 전처리하고 벡터 인덱스를 생성")
    subparsers.add_parser("collect-web", help="기본 정책 사이트 웹 데이터를 수집해 JSON으로 저장")

    ask_parser = subparsers.add_parser("ask", help="질문에 대한 답변 생성")
    ask_parser.add_argument("question", type=str, help="사용자 질문")

    subparsers.add_parser("eval", help="기본 평가 시나리오 실행")

    args = parser.parse_args()

    if args.command == "build-index":
        command_build_index()
    elif args.command == "collect-web":
        command_collect_web()
    elif args.command == "ask":
        command_ask(args.question)
    elif args.command == "eval":
        command_eval()


if __name__ == "__main__":
    main()
