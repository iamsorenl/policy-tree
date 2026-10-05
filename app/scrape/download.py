"""Download every discovered PDF to disk and record the outcome.

Idempotent: skips policies whose latest download row is a clean success.
Failures are recorded so they remain visible in the DB.

Run: `python -m app.scrape.download` from the repo root.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.db import connect, init_db, latest_successful_download, record_download

LOG = logging.getLogger("scrape.download")

USER_AGENT = "policy-tree/0.1"
REQUEST_TIMEOUT = 60.0
POLITE_DELAY_SEC = 0.6
PDF_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "pdfs"


class _RetriableHTTPError(Exception):
    """Marker for HTTP 5xx — used to drive tenacity retries."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


def _make_client() -> httpx.Client:
    return httpx.Client(
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/pdf,*/*;q=0.8",
        },
        follow_redirects=True,
        timeout=REQUEST_TIMEOUT,
    )


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1.0, min=1.0, max=10.0),
    retry=retry_if_exception_type((httpx.TransportError, httpx.TimeoutException, _RetriableHTTPError)),
)
def _download_with_retry(client: httpx.Client, url: str) -> httpx.Response:
    resp = client.get(url)
    if 500 <= resp.status_code < 600:
        raise _RetriableHTTPError(resp.status_code, f"HTTP {resp.status_code} from {url}")
    return resp


def _all_policies(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("SELECT id, title, pdf_url FROM policies ORDER BY id").fetchall()


def _do_download(
    client: httpx.Client,
    policy: sqlite3.Row,
) -> tuple[str | None, int | None, str | None]:
    """Return (stored_location, http_status, error)."""
    url = policy["pdf_url"]
    try:
        resp = _download_with_retry(client, url)
    except _RetriableHTTPError as exc:
        return None, exc.status_code, str(exc)
    except (httpx.TransportError, httpx.TimeoutException) as exc:
        return None, None, f"{type(exc).__name__}: {exc}"
    except RetryError as exc:  # belt-and-suspenders
        return None, None, f"RetryError: {exc}"

    if resp.status_code != 200:
        return None, resp.status_code, f"HTTP {resp.status_code}"

    body = resp.content
    if not body:
        return None, resp.status_code, "Empty response body"

    # Sanity check: looks like a PDF (header `%PDF-`). Don't reject if not — some
    # CDNs serve via wrappers — but log it.
    if not body.lstrip().startswith(b"%PDF"):
        LOG.warning(
            "Response for policy id=%s does not start with %%PDF (content-type=%s, size=%d)",
            policy["id"],
            resp.headers.get("content-type", "?"),
            len(body),
        )

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PDF_DIR / f"{policy['id']}.pdf"
    out_path.write_bytes(body)
    return str(out_path), resp.status_code, None


def download() -> dict[str, int]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    init_db()
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {"attempted": 0, "succeeded": 0, "skipped": 0, "failed": 0}

    client = _make_client()
    try:
        with connect() as conn:
            policies = _all_policies(conn)
            LOG.info("Considering %d policies for download", len(policies))

            for idx, policy in enumerate(policies):
                pid = policy["id"]
                existing = latest_successful_download(conn, pid)
                if existing and existing["stored_location"] and not existing["error"]:
                    # Verify file is still present on disk; if missing, redownload.
                    if Path(existing["stored_location"]).exists():
                        counts["skipped"] += 1
                        continue
                    LOG.info(
                        "Policy id=%s recorded as downloaded but file missing — re-downloading",
                        pid,
                    )

                counts["attempted"] += 1
                LOG.info("Downloading policy id=%s title=%r", pid, policy["title"])
                stored, status, error = _do_download(client, policy)

                record_download(
                    conn,
                    policy_id=pid,
                    stored_location=stored,
                    http_status=status,
                    error=error,
                )
                conn.commit()

                if error or stored is None:
                    counts["failed"] += 1
                    LOG.warning("Download failed for id=%s: %s (status=%s)", pid, error, status)
                else:
                    counts["succeeded"] += 1

                if idx < len(policies) - 1:
                    time.sleep(POLITE_DELAY_SEC)

        LOG.info(
            "Download complete: attempted=%d succeeded=%d skipped=%d failed=%d",
            counts["attempted"],
            counts["succeeded"],
            counts["skipped"],
            counts["failed"],
        )
        return counts
    finally:
        client.close()


def main() -> None:
    download()


if __name__ == "__main__":
    main()
