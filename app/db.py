"""SQLite schema + accessors. Single source of truth for the data contract."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "policies.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS policies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    pdf_url TEXT NOT NULL UNIQUE,
    source_page_url TEXT NOT NULL,
    discovered_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS downloads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_id INTEGER NOT NULL,
    stored_location TEXT,
    downloaded_at TEXT NOT NULL,
    http_status INTEGER,
    error TEXT,
    FOREIGN KEY (policy_id) REFERENCES policies(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_downloads_policy_id ON downloads(policy_id);

CREATE TABLE IF NOT EXISTS structured_policies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_id INTEGER NOT NULL UNIQUE,
    extracted_text TEXT,
    structured_json TEXT,
    structured_at TEXT NOT NULL,
    llm_metadata TEXT,
    validation_error TEXT,
    FOREIGN KEY (policy_id) REFERENCES policies(id) ON DELETE CASCADE
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)
        conn.commit()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
    finally:
        conn.close()


# ---------- policies ----------

def upsert_policy(conn: sqlite3.Connection, *, title: str, pdf_url: str, source_page_url: str) -> int:
    """Insert policy if pdf_url is new; return policy id either way. Idempotent."""
    cur = conn.execute(
        """
        INSERT INTO policies (title, pdf_url, source_page_url, discovered_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(pdf_url) DO NOTHING
        """,
        (title, pdf_url, source_page_url, now_iso()),
    )
    if cur.rowcount > 0:
        return cur.lastrowid  # type: ignore[return-value]
    row = conn.execute("SELECT id FROM policies WHERE pdf_url = ?", (pdf_url,)).fetchone()
    return row["id"]


def list_policies(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT p.id, p.title, p.pdf_url, p.source_page_url, p.discovered_at,
               sp.id IS NOT NULL AS has_structure,
               sp.validation_error
        FROM policies p
        LEFT JOIN structured_policies sp ON sp.policy_id = p.id
        ORDER BY p.title COLLATE NOCASE
        """
    ).fetchall()


def get_policy(conn: sqlite3.Connection, policy_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM policies WHERE id = ?", (policy_id,)).fetchone()


def is_meta_document(title: str) -> bool:
    """True for policy-list docs, change summaries, partnership notices —
    not real medical-necessity policies.

    Centralized so the rule is testable and easy to extend as new
    non-criteria document patterns are observed in the source page.

    Patterns are tuples; each tuple is a logical AND of substrings that
    must all be present in the (lowercased) title. The original SQL used
    LIKE '%Care Guidelines%Update%' which matches non-adjacent words
    (e.g. "MCG Care Guidelines 29th Edition ... Update"), so we model that
    as a two-token AND rather than a single substring.
    """
    if not title:
        return False
    title_lower = title.lower()
    META_PATTERNS: tuple[tuple[str, ...], ...] = (
        ("summary of changes",),
        ("care guidelines", "update"),  # MCG edition-update notices
        ("partnership",),
        # Add more patterns here as new meta-doc shapes are observed.
    )
    return any(all(tok in title_lower for tok in pat) for pat in META_PATTERNS)


def policies_needing_structure(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    """Policies with a successful download and no structured row yet.

    Skips obvious non-criteria meta documents (see `is_meta_document`) so
    the LLM is only asked to structure real medical-necessity policies.
    Filtering is done in Python rather than SQL so the predicate is
    testable in isolation and the SQL stays declarative.
    """
    rows = conn.execute(
        """
        SELECT p.id, p.title, p.pdf_url, d.stored_location
        FROM policies p
        JOIN downloads d ON d.policy_id = p.id
        LEFT JOIN structured_policies sp ON sp.policy_id = p.id
        WHERE d.stored_location IS NOT NULL
          AND d.error IS NULL
          AND sp.id IS NULL
        ORDER BY p.id
        """,
    ).fetchall()
    filtered = [r for r in rows if not is_meta_document(r["title"])]
    return filtered[:limit] if limit > 0 else filtered


# ---------- downloads ----------

def latest_successful_download(conn: sqlite3.Connection, policy_id: int) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT * FROM downloads
        WHERE policy_id = ? AND error IS NULL AND stored_location IS NOT NULL
        ORDER BY id DESC
        LIMIT 1
        """,
        (policy_id,),
    ).fetchone()


def record_download(
    conn: sqlite3.Connection,
    *,
    policy_id: int,
    stored_location: str | None,
    http_status: int | None,
    error: str | None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO downloads (policy_id, stored_location, downloaded_at, http_status, error)
        VALUES (?, ?, ?, ?, ?)
        """,
        (policy_id, stored_location, now_iso(), http_status, error),
    )
    return cur.lastrowid  # type: ignore[return-value]


# ---------- structured_policies ----------

def upsert_structured(
    conn: sqlite3.Connection,
    *,
    policy_id: int,
    extracted_text: str | None,
    structured_json: dict | None,
    llm_metadata: dict | None,
    validation_error: str | None,
) -> int:
    """One structured row per policy. Replace on rerun."""
    payload = (
        policy_id,
        extracted_text,
        json.dumps(structured_json) if structured_json is not None else None,
        now_iso(),
        json.dumps(llm_metadata) if llm_metadata is not None else None,
        validation_error,
    )
    cur = conn.execute(
        """
        INSERT INTO structured_policies
            (policy_id, extracted_text, structured_json, structured_at, llm_metadata, validation_error)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(policy_id) DO UPDATE SET
            extracted_text   = excluded.extracted_text,
            structured_json  = excluded.structured_json,
            structured_at    = excluded.structured_at,
            llm_metadata     = excluded.llm_metadata,
            validation_error = excluded.validation_error
        """,
        payload,
    )
    if cur.rowcount > 0 and cur.lastrowid:
        return cur.lastrowid
    row = conn.execute(
        "SELECT id FROM structured_policies WHERE policy_id = ?", (policy_id,)
    ).fetchone()
    return row["id"]


def get_structured(conn: sqlite3.Connection, policy_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM structured_policies WHERE policy_id = ?", (policy_id,)
    ).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "policy_id": row["policy_id"],
        "extracted_text": row["extracted_text"],
        "structured_json": json.loads(row["structured_json"]) if row["structured_json"] else None,
        "structured_at": row["structured_at"],
        "llm_metadata": json.loads(row["llm_metadata"]) if row["llm_metadata"] else None,
        "validation_error": row["validation_error"],
    }
