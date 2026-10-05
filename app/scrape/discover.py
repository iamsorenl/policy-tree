"""Discover policy PDFs.

Fetch the listing page at SOURCE_URL, follow each same-site policy detail
link, harvest the `.pdf` link(s) found there, and upsert into the `policies`
table keyed by `pdf_url`.

Run: `python -m app.scrape.discover [--source-url URL]` from the repo root.
"""
from __future__ import annotations

import argparse
import os
import logging
import re
import time
from typing import Iterable
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from dotenv import load_dotenv

from app.db import connect, init_db, upsert_policy

load_dotenv()

LOG = logging.getLogger("scrape.discover")

SOURCE_URL = os.getenv("SOURCE_URL", "")
# Optional comma separated host list. When set, only PDFs on these hosts are kept.
PDF_HOST_ALLOWLIST = [h.strip().lower() for h in os.getenv("PDF_HOST_ALLOWLIST", "").split(",") if h.strip()]
# Optional path prefix for detail pages. Default is the listing page's parent folder.
DETAIL_PATH_PREFIX = os.getenv("DETAIL_PATH_PREFIX", "")
USER_AGENT = "policy-tree/0.1"
REQUEST_TIMEOUT = 30.0
POLITE_DELAY_SEC = 0.5

# Match anything that looks like a `.pdf` URL inside the rendered HTML/JSON.
# We capture quoted strings ending in `.pdf` to avoid eating closing punctuation.
_PDF_URL_RE = re.compile(r"""['"]([^'"\s<>]+?\.pdf)['"]""", re.IGNORECASE)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    return isinstance(exc, (httpx.TransportError, httpx.TimeoutException))


def _make_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
        follow_redirects=True,
        timeout=REQUEST_TIMEOUT,
    )


@retry(
    reraise=True,
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1.0, min=1.0, max=8.0),
    retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException, httpx.HTTPStatusError)),
)
def _fetch(client: httpx.Client, url: str) -> str:
    resp = client.get(url)
    if resp.status_code >= 500:
        # Trigger tenacity retry only on 5xx; let 4xx bubble up immediately.
        resp.raise_for_status()
    resp.raise_for_status()
    return resp.text


def _normalize_url(url: str, base: str) -> str:
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("/"):
        return urljoin(base, url)
    # Bare relative paths (e.g. "CG013_Acupuncture.pdf") and full URLs both
    # resolve correctly via urljoin: absolute URLs pass through unchanged,
    # relative paths get joined against the base detail-page URL.
    return urljoin(base, url)


def _site(host: str) -> str:
    """Last two host labels, so www.example.com and cdn.example.com match."""
    return ".".join(host.lower().split(".")[-2:])


def _is_acceptable_pdf_candidate(full: str, base: str) -> bool:
    """Keep absolute http(s) `.pdf` URLs that pass the host rule.

    With PDF_HOST_ALLOWLIST set, the host must be on the list. Otherwise the
    PDF must live on the same site as the page it was found on.
    """
    parsed = urlparse(full)
    if parsed.scheme not in ("http", "https") or not parsed.path.lower().endswith(".pdf"):
        return False
    host = (parsed.hostname or "").lower()
    if PDF_HOST_ALLOWLIST:
        return host in PDF_HOST_ALLOWLIST
    return _site(host) == _site(urlparse(base).hostname or "")


def _extract_pdf_urls(html: str, base_url: str) -> list[str]:
    """Find every PDF URL inside the page HTML (anchor hrefs + quoted strings)."""
    candidates = [a["href"].strip() for a in BeautifulSoup(html, "html.parser").find_all("a", href=True)]
    # Quoted strings catch PDFs that only appear inside embedded JSON.
    candidates += [m.group(1) for m in _PDF_URL_RE.finditer(html)]

    # A bare filename in JSON metadata joins to a made-up URL on the page's own
    # host. Skip it when a path-bearing candidate with the same filename exists.
    full_names = {c.rsplit("/", 1)[-1] for c in candidates if "/" in c}
    found: list[str] = []
    for raw in candidates:
        if ".pdf" not in raw.lower() or ("/" not in raw and raw in full_names):
            continue
        full = _normalize_url(raw, base_url)
        if _is_acceptable_pdf_candidate(full, base_url) and full not in found:
            found.append(full)
    return found


def _detail_prefix(source_url: str) -> str:
    if DETAIL_PATH_PREFIX:
        return DETAIL_PATH_PREFIX
    path = urlparse(source_url).path
    return path.rsplit("/", 1)[0] + "/"


def _discover_listing(html: str, source_url: str) -> list[tuple[str, str]]:
    """Return (title, detail_url) pairs for same-site links under the detail prefix."""
    soup = BeautifulSoup(html, "html.parser")
    prefix = _detail_prefix(source_url)
    site = _site(urlparse(source_url).hostname or "")
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()

    for a in soup.find_all("a", href=True):
        url = urljoin(source_url, a["href"].strip()).split("#", 1)[0]
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or _site(parsed.hostname or "") != site:
            continue
        if not parsed.path.startswith(prefix) or parsed.path.lower().endswith(".pdf"):
            continue
        if url == source_url or url in seen:
            continue
        seen.add(url)
        rows.append((a.get_text(strip=True) or url, url))
    return rows


def _detail_title_fallback(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    for tag in ("h1", "h2", "title"):
        el = soup.find(tag)
        if el and el.get_text(strip=True):
            return el.get_text(strip=True)
    return None


def discover(source_url: str | None = None) -> dict[str, int]:
    """Run discovery end-to-end. Returns counts dict."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    source_url = source_url or SOURCE_URL
    if not source_url:
        raise SystemExit("Set SOURCE_URL in the environment or pass --source-url.")
    init_db()

    client = _make_client()
    try:
        LOG.info("Fetching listing %s", source_url)
        listing_html = _fetch(client, source_url)
        rows = _discover_listing(listing_html, source_url)
        LOG.info("Found %d policy links on listing page", len(rows))

        # Some PDFs may already be linked directly from the listing page; capture them
        # before we descend into detail pages so we don't miss anything.
        listing_pdfs = _extract_pdf_urls(listing_html, source_url)
        if listing_pdfs:
            LOG.info("Listing page exposes %d direct PDF link(s)", len(listing_pdfs))

        discovered: list[tuple[str, str, str]] = []  # (title, pdf_url, source_page_url)

        # Direct PDFs on listing page (covered for safety).
        for pdf in listing_pdfs:
            discovered.append((pdf.rsplit("/", 1)[-1], pdf, source_url))

        # Follow each detail page and harvest PDF(s).
        for idx, (title, detail_url) in enumerate(rows, start=1):
            try:
                detail_html = _fetch(client, detail_url)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Failed to fetch detail page %s: %s", detail_url, exc)
                time.sleep(POLITE_DELAY_SEC)
                continue
            pdfs = _extract_pdf_urls(detail_html, detail_url)
            if not pdfs:
                LOG.warning("No PDFs found on detail page %s (title=%r)", detail_url, title)
            else:
                # Use list-page title for the first PDF; if the page exposes more,
                # disambiguate using the filename so titles stay unique-ish.
                fallback = _detail_title_fallback(detail_html)
                for i, pdf_url in enumerate(pdfs):
                    base_title = title or fallback or pdf_url.rsplit("/", 1)[-1]
                    if len(pdfs) > 1 and i > 0:
                        suffix = pdf_url.rsplit("/", 1)[-1]
                        full_title = f"{base_title} [{suffix}]"
                    else:
                        full_title = base_title
                    discovered.append((full_title, pdf_url, detail_url))
            if idx < len(rows):
                time.sleep(POLITE_DELAY_SEC)

        LOG.info("Total PDFs discovered: %d", len(discovered))

        inserted = 0
        skipped = 0
        with connect() as conn:
            for title, pdf_url, source_page_url in discovered:
                # We can detect "already existed" by checking before the upsert.
                existed = conn.execute(
                    "SELECT 1 FROM policies WHERE pdf_url = ?", (pdf_url,)
                ).fetchone()
                upsert_policy(conn, title=title, pdf_url=pdf_url, source_page_url=source_page_url)
                if existed:
                    skipped += 1
                else:
                    inserted += 1
            conn.commit()

        LOG.info(
            "Discovery complete: discovered=%d inserted=%d skipped(existing)=%d",
            len(discovered),
            inserted,
            skipped,
        )
        return {"discovered": len(discovered), "inserted": inserted, "skipped": skipped}
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover policy PDFs.")
    parser.add_argument("--source-url", help="Listing page URL (default: SOURCE_URL env).")
    discover(parser.parse_args().source_url)


if __name__ == "__main__":
    main()
