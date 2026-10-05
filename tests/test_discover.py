"""Discovery tests run against fake pages, no network."""
from __future__ import annotations

import httpx
import pytest

from app import db as db_module
from app.scrape import discover as d
from tests.conftest import FIXTURES

BASE = "https://www.example-health.test/policies/index.html"
PAGES = {
    BASE: "index.html",
    "https://www.example-health.test/policies/knee-brace": "knee-brace.html",
    "https://www.example-health.test/policies/sleep-study": "sleep-study.html",
}


def _handler(request: httpx.Request) -> httpx.Response:
    name = PAGES.get(str(request.url))
    if name is None:
        return httpx.Response(404)
    return httpx.Response(200, text=(FIXTURES / name).read_text())


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(d.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        d, "_make_client", lambda: httpx.Client(transport=httpx.MockTransport(_handler), follow_redirects=True)
    )
    monkeypatch.setattr(d, "PDF_HOST_ALLOWLIST", [])
    monkeypatch.setattr(d, "DETAIL_PATH_PREFIX", "")


def test_listing_keeps_only_same_site_links_under_prefix():
    rows = d._discover_listing((FIXTURES / "index.html").read_text(), BASE)
    assert [u for _, u in rows] == [
        "https://www.example-health.test/policies/knee-brace",
        "https://www.example-health.test/policies/sleep-study",
    ]


def test_pdf_extraction_same_site_cdn_and_dedupe():
    html = (FIXTURES / "knee-brace.html").read_text()
    urls = d._extract_pdf_urls(html, "https://www.example-health.test/policies/knee-brace")
    # CDN host shares the site, the bare filename is dropped, evil.test is rejected.
    assert urls == ["https://cdn.example-health.test/files/KB001_Knee_Brace.pdf"]


def test_allowlist_restricts_hosts(monkeypatch):
    monkeypatch.setattr(d, "PDF_HOST_ALLOWLIST", ["evil.test"])
    html = (FIXTURES / "knee-brace.html").read_text()
    assert d._extract_pdf_urls(html, "https://www.example-health.test/x") == ["https://evil.test/KB001.pdf"]


def test_discover_end_to_end_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "t.db")
    first = d.discover(BASE)
    assert first == {"discovered": 2, "inserted": 2, "skipped": 0}
    second = d.discover(BASE)
    assert second == {"discovered": 2, "inserted": 0, "skipped": 2}
    with db_module.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM policies").fetchone()[0] == 2


def test_missing_source_url_exits(monkeypatch):
    monkeypatch.setattr(d, "SOURCE_URL", "")
    with pytest.raises(SystemExit):
        d.discover()
