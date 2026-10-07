"""KATO의 공개 동호인 대회 일정 페이지를 수집한다."""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://kato.kr/"
DATA_FILE = Path(__file__).parent / "data" / "kato_tournaments.json"
DATE_PATTERN = re.compile(
    r"(?P<year>\d{4})년\s*(?P<month>\d{1,2})월\s*(?P<day>\d{1,2})일"
    r"(?:\s*\([^)]*\))?\s*(?P<hour>\d{1,2}:\d{2})?"
)
DETAIL_PATH_PATTERN = re.compile(r"['\"](/openGame/[\w-]+)['\"]")
SECTIONS = {"경기중인 대회", "접수중인 대회", "접수예정 대회"}


def parse_date(raw: str) -> str | None:
    """사이트의 한국어 날짜를 ISO 날짜/시간 문자열로 바꾼다."""
    match = DATE_PATTERN.search(raw)
    if not match:
        return None
    parts = match.groupdict()
    date = f"{int(parts['year']):04d}-{int(parts['month']):02d}-{int(parts['day']):02d}"
    return f"{date}T{parts['hour']}" if parts["hour"] else date


def parse_index(html: str) -> list[dict[str, str]]:
    """홈페이지의 대회 링크와 접수 구분을 추출한다."""
    soup = BeautifulSoup(html, "html.parser")
    found: dict[str, dict[str, str]] = {}
    for heading in soup.find_all("h2"):
        section = heading.get_text(" ", strip=True)
        if section not in SECTIONS:
            continue
        container = heading.find_parent("div", class_="gtco-container")
        if container is None:
            continue
        for card in container.select("div.service[onclick]"):
            match = DETAIL_PATH_PATTERN.search(card["onclick"])
            if not match:
                continue
            url = urljoin(BASE_URL, match.group(1))
            found[url] = {"url": url, "listing_status": section}
    return list(found.values())


def parse_detail(html: str, url: str, listing_status: str) -> list[dict[str, str]]:
    """대회 상세 페이지의 부서별 경기 일정을 한 행씩 추출한다."""
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.select_one(".group-title")
    if title_tag is None:
        raise ValueError(f"대회 제목을 찾지 못했습니다: {url}")
    title = title_tag.get_text(" ", strip=True)

    tables = soup.select("table.table-bordered")
    if not tables:
        raise ValueError(f"대회 일정 표를 찾지 못했습니다: {url}")

    # 접수 전에는 부서별 신청 표가 없으므로 안내 표의 '일 시' 행을 사용한다.
    general_venue = "미기재"
    for tr in tables[0].select("tr"):
        cells = tr.find_all("td", recursive=False)
        if len(cells) >= 2 and cells[0].get_text(" ", strip=True) == "장 소":
            general_venue = cells[1].get_text(" ", strip=True)
            break

    schedule_table = next((table for table in tables if table.select_one(".place")), None)
    rows = []
    if schedule_table is not None:
        for tr in schedule_table.select("tr"):
            cells = tr.find_all("td", recursive=False)
            if len(cells) < 2:
                continue
            date_tag = cells[1].find("div", recursive=False)
            starts_at = parse_date(date_tag.get_text(" ", strip=True)) if date_tag else None
            if starts_at is None:
                continue
            venue_tag = cells[1].select_one(".place")
            status_tag = cells[2].select_one(".takeready") if len(cells) > 2 else None
            rows.append(
                {
                    "tournament": title,
                    "division": cells[0].get_text(" ", strip=True),
                    "starts_at": starts_at,
                    "venue": venue_tag.get_text(" ", strip=True) if venue_tag else general_venue,
                    "registration_status": status_tag.get_text(" ", strip=True) if status_tag else listing_status,
                    "listing_status": listing_status,
                    "source_url": url,
                }
            )
    else:
        for tr in tables[0].select("tr"):
            cells = tr.find_all("td", recursive=False)
            if len(cells) < 2 or not cells[-2].get("class"):
                continue
            if not ({"first-comp", "rowcell"} & set(cells[-2].get("class", []))):
                continue
            starts_at = parse_date(cells[-1].get_text(" ", strip=True))
            if starts_at is None:
                continue
            rows.append(
                {
                    "tournament": title,
                    "division": cells[-2].get_text(" ", strip=True),
                    "starts_at": starts_at,
                    "venue": general_venue,
                    "registration_status": "미기재",
                    "listing_status": listing_status,
                    "source_url": url,
                }
            )
    return rows


def crawl(delay: float = 0.4) -> dict:
    """KATO 홈페이지에서 연결된 대회 상세 페이지를 순서대로 수집한다."""
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; HateslopTennisFAQ/1.0)"})

    def fetch(url: str) -> str:
        response = session.get(url, timeout=15)
        response.raise_for_status()
        return response.content.decode("utf-8", errors="replace")

    listings = parse_index(fetch(BASE_URL))
    if not listings:
        raise RuntimeError("대회 링크를 찾지 못했습니다. KATO 홈페이지 구조를 확인하세요.")

    events: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    for listing in listings:
        time.sleep(delay)
        try:
            rows = parse_detail(fetch(listing["url"]), **listing)
            if not rows:
                raise ValueError("부서별 일정 행이 없습니다")
            events.extend(rows)
        except (requests.RequestException, ValueError) as exc:
            errors.append({"url": listing["url"], "error": str(exc)})

    if not events:
        raise RuntimeError("수집된 경기 일정이 없습니다. 원천 데이터를 덮어쓰지 않았습니다.")
    return {
        "source": BASE_URL,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "listing_count": len(listings),
        "events": events,
        "errors": errors,
    }


def main() -> None:
    snapshot = crawl()
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    DATA_FILE.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"대회 {snapshot['listing_count']}개 페이지에서 경기 일정 {len(snapshot['events'])}건 저장: "
        f"{DATA_FILE} (오류 {len(snapshot['errors'])}건)"
    )


if __name__ == "__main__":
    main()
