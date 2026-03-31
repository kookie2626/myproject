from __future__ import annotations

import json
from urllib.parse import urljoin
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

NOISE_TITLE_KEYWORDS = [
    "로그인",
    "회원가입",
    "개인정보",
    "이용약관",
    "저작권",
    "사이트맵",
    "페이스북",
    "인스타그램",
    "유튜브",
    "블로그",
    "TOP",
]

NOISE_URL_KEYWORDS = [
    "facebook.com",
    "instagram.com",
    "youtube.com",
    "blog.naver.com",
    "javascript:",
    "/login",
    "/join",
    "privacy",
    "terms",
]


@dataclass
class WebRecord:
    source_site: str
    url: str
    title: str
    body: str
    links: List[dict]
    notice_id: str = ""
    parent_url: str = ""


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


def _extract_relevant_links(url: str, html: str, max_links: int = 80) -> List[dict]:
    soup = BeautifulSoup(html, "lxml")
    links: List[dict] = []

    for a_tag in soup.select("a"):
        href = (a_tag.get("href") or "").strip()
        text = _clean_text(a_tag.get_text(" ", strip=True))
        if not href or not text:
            continue

        if href.startswith("#") or href.lower().startswith("javascript:"):
            # K-Startup 목록의 go_view(숫자) 패턴은 공고 식별자로 보존한다.
            go_view_match = re.search(r"go_view\((\d+)\)", href)
            if go_view_match:
                links.append({"title": text, "url": "", "notice_id": go_view_match.group(1)})
            continue

        absolute = urljoin(url, href)
        if absolute.startswith("http"):
            links.append({"title": text, "url": absolute, "notice_id": ""})

    dedup = {}
    for link in links:
        key = (link.get("title", ""), link.get("url", ""), link.get("notice_id", ""))
        dedup[key] = link

    deduped = list(dedup.values())
    notice_links = [link for link in deduped if link.get("notice_id")]
    normal_links = [link for link in deduped if not link.get("notice_id")]
    return (notice_links + normal_links)[:max_links]


def _is_noise_link(link: dict) -> bool:
    title = str(link.get("title", "")).lower()
    url = str(link.get("url", "")).lower()

    if len(title) <= 1:
        return True
    if any(keyword.lower() in title for keyword in NOISE_TITLE_KEYWORDS):
        return True
    if any(keyword in url for keyword in NOISE_URL_KEYWORDS):
        return True
    return False


def collect_web_records(urls: List[str] | None = None, timeout_sec: int = 20) -> List[WebRecord]:
    target_urls = urls or DEFAULT_TARGETS
    records: List[WebRecord] = []

    with httpx.Client(follow_redirects=True, timeout=timeout_sec) as client:
        for url in target_urls:
            response = client.get(url)
            response.raise_for_status()
            title, body = _extract_main_text(response.text)
            links = [
                link for link in _extract_relevant_links(url, response.text) if not _is_noise_link(link)
            ]
            if not body:
                continue
            records.append(
                WebRecord(
                    source_site=_site_name(url),
                    url=url,
                    title=title,
                    body=body,
                    links=links,
                )
            )

            # K-Startup 목록 페이지는 notice_id 기반 상세 페이지를 추가 수집한다.
            if _site_name(url) == "k-startup":
                records.extend(_collect_kstartup_details(client=client, list_url=url, links=links))

    return records


def _collect_kstartup_details(client: httpx.Client, list_url: str, links: List[dict], max_details: int = 20) -> List[WebRecord]:
    details: List[WebRecord] = []
    seen_notice_ids = set()

    for link in links:
        notice_id = str(link.get("notice_id", "")).strip()
        if not notice_id or not notice_id.isdigit() or notice_id in seen_notice_ids:
            continue
        seen_notice_ids.add(notice_id)

        detail_url = f"{list_url.split('?')[0]}?schM=view&pbancSn={notice_id}"
        try:
            response = client.get(detail_url)
            response.raise_for_status()
        except httpx.HTTPError:
            continue

        title, body = _extract_main_text(response.text)
        if not body:
            continue

        details.append(
            WebRecord(
                source_site="k-startup-detail",
                url=detail_url,
                title=title,
                body=body,
                links=[],
                notice_id=notice_id,
                parent_url=list_url,
            )
        )

        if len(details) >= max_details:
            break

    return details


def save_records_as_json(records: List[WebRecord], output_path: str) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serialized = [
        {
            "source_site": r.source_site,
            "url": r.url,
            "title": r.title,
            "body": r.body,
            "links": r.links,
            "notice_id": r.notice_id,
            "parent_url": r.parent_url,
        }
        for r in records
    ]
    path.write_text(json.dumps(serialized, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def collect_and_save_default(output_path: str) -> tuple[int, str]:
    records = collect_web_records()
    saved_path = save_records_as_json(records, output_path)
    return len(records), saved_path
