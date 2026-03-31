from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


REGION_KEYWORDS = ["서울", "경기", "인천", "부산", "대구", "광주", "대전", "울산", "세종"]
STAGE_KEYWORDS = ["예비", "초기", "도약", "재도전"]


def _extract_regions(text: str) -> str:
    found = [region for region in REGION_KEYWORDS if region in text]
    return "|".join(found)


def _extract_stages(text: str) -> str:
    found = [stage for stage in STAGE_KEYWORDS if stage in text]
    return "|".join(found)


def _extract_age_bucket(text: str) -> str:
    lowered = text.replace(" ", "")
    if "청년" in text or "만39세" in lowered or re.search(r"만\s*3[0-9]세", text):
        return "youth"
    if "중장년" in text or "시니어" in text:
        return "senior"
    return "all"


def _annotate_metadata(doc: Document) -> None:
    text = doc.page_content
    doc.metadata["regions"] = _extract_regions(text)
    doc.metadata["stages"] = _extract_stages(text)
    doc.metadata["age_bucket"] = _extract_age_bucket(text)


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
            _annotate_metadata(page)
        docs.extend(pages)
    return docs


def load_web_json_documents(raw_docs_dir: str) -> List[Document]:
    raw_path = Path(raw_docs_dir)
    if not raw_path.exists():
        return []

    docs: List[Document] = []
    for json_path in raw_path.glob("*.json"):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        if not isinstance(payload, list):
            continue

        for item in payload:
            if not isinstance(item, dict):
                continue
            body = str(item.get("body", "")).strip()
            if not body:
                continue

            meta = {
                "source_file": json_path.name,
                "page_number": item.get("page_number", 1),
                "source_site": item.get("source_site", "web"),
                "source_url": item.get("url", ""),
                "title": item.get("title", ""),
                "notice_id": item.get("notice_id", ""),
                "parent_url": item.get("parent_url", ""),
            }
            doc = Document(page_content=body, metadata=meta)
            _annotate_metadata(doc)
            docs.append(doc)

            # 링크 목록을 별도 문서로 만들어 검색 시 공고/상세 페이지를 더 잘 찾게 한다.
            links = item.get("links", [])
            if isinstance(links, list):
                for link in links:
                    if not isinstance(link, dict):
                        continue

                    link_title = str(link.get("title", "")).strip()
                    link_url = str(link.get("url", "")).strip()
                    notice_id = str(link.get("notice_id", "")).strip()
                    if not link_title and not link_url and not notice_id:
                        continue

                    link_text = "\n".join(
                        [
                            f"링크 제목: {link_title}",
                            f"링크 URL: {link_url}",
                            f"공고 ID: {notice_id}",
                            f"출처 사이트: {meta['source_site']}",
                        ]
                    ).strip()

                    link_meta = dict(meta)
                    link_meta["doc_type"] = "web_link"
                    link_meta["link_title"] = link_title
                    link_meta["link_url"] = link_url
                    link_meta["notice_id"] = notice_id

                    link_doc = Document(page_content=link_text, metadata=link_meta)
                    _annotate_metadata(link_doc)
                    docs.append(link_doc)

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
