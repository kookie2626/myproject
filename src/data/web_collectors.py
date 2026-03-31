from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import httpx
from bs4 import BeautifulSoup


DEFAULT_TARGETS = [
    "https://www.k-startup.go.kr/web/contents/bizpbanc-ongoing.do",
    "https://www.kised.or.kr/menu.es?mid=a10205010000",
    "https://www.kosmes.or.kr/nsh/map/main.do#none",
]


@dataclass
class WebRecord:
    source_site: str
    url: str
    title: str
    body: str


def _site_name(url: str) -> str:
    if "k-startup.go.kr" in url:
        return "k-startup"
    if "kised.or.kr" in url:
        return "kised"
    if "kosmes.or.kr" in url:
        return "kosmes"
    return "unknown"


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_main_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")

    for bad in soup.select("script, style, noscript, iframe"):
        bad.extract()

    title = _clean_text(soup.title.get_text(" ", strip=True)) if soup.title else "untitled"

    main = soup.select_one("main") or soup.select_one("#contents") or soup.select_one(".contents")
    if main:
        body = _clean_text(main.get_text(" ", strip=True))
    else:
        body = _clean_text(soup.get_text(" ", strip=True))

    return title, body


def collect_web_records(urls: List[str] | None = None, timeout_sec: int = 20) -> List[WebRecord]:
    target_urls = urls or DEFAULT_TARGETS
    records: List[WebRecord] = []

    with httpx.Client(follow_redirects=True, timeout=timeout_sec) as client:
        for url in target_urls:
            response = client.get(url)
            response.raise_for_status()
            title, body = _extract_main_text(response.text)
            if not body:
                continue
            records.append(
                WebRecord(
                    source_site=_site_name(url),
                    url=url,
                    title=title,
                    body=body,
                )
            )

    return records


def save_records_as_json(records: List[WebRecord], output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serialized = [
        {
            "source_site": r.source_site,
            "url": r.url,
            "title": r.title,
            "body": r.body,
        }
        for r in records
    ]
    path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def collect_and_save_default(output_path: str) -> tuple[int, str]:
    records = collect_web_records()
    saved_path = save_records_as_json(records, output_path)
    return len(records), saved_path
