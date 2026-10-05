"""Round-trip tests for app.db on a temp SQLite file.

We monkeypatch `app.db.DB_PATH` to a per-test tmp path so the real
data/policies.db is never touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app import db as db_module


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Point app.db at a fresh on-disk sqlite file under tmp_path and init it."""
    fresh = tmp_path / "policies_test.db"
    monkeypatch.setattr(db_module, "DB_PATH", fresh)
    db_module.init_db()
    return fresh


def test_init_db_creates_tables(tmp_db: Path):
    assert tmp_db.exists()
    with db_module.connect() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    names = [r["name"] for r in rows]
    assert "policies" in names
    assert "downloads" in names
    assert "structured_policies" in names


def test_upsert_policy_is_idempotent(tmp_db: Path):
    with db_module.connect() as conn:
        first_id = db_module.upsert_policy(
            conn,
            title="Policy A",
            pdf_url="https://example.com/a.pdf",
            source_page_url="https://example.com/list",
        )
        # Same pdf_url, possibly different title — should NOT create a new row
        second_id = db_module.upsert_policy(
            conn,
            title="Policy A (renamed)",
            pdf_url="https://example.com/a.pdf",
            source_page_url="https://example.com/other",
        )
        assert first_id == second_id

        # And different pdf_url DOES create a new row
        third_id = db_module.upsert_policy(
            conn,
            title="Policy B",
            pdf_url="https://example.com/b.pdf",
            source_page_url="https://example.com/list",
        )
        assert third_id != first_id

        count = conn.execute("SELECT COUNT(*) AS c FROM policies").fetchone()["c"]
        assert count == 2


def test_record_download_and_latest_successful(tmp_db: Path):
    with db_module.connect() as conn:
        policy_id = db_module.upsert_policy(
            conn,
            title="Policy A",
            pdf_url="https://example.com/a.pdf",
            source_page_url="https://example.com/list",
        )
        # First attempt: failure
        db_module.record_download(
            conn,
            policy_id=policy_id,
            stored_location=None,
            http_status=503,
            error="server error",
        )
        # latest_successful_download must skip failures
        assert db_module.latest_successful_download(conn, policy_id) is None

        # Second attempt: success
        db_module.record_download(
            conn,
            policy_id=policy_id,
            stored_location="data/pdfs/a.pdf",
            http_status=200,
            error=None,
        )

        latest = db_module.latest_successful_download(conn, policy_id)
        assert latest is not None
        assert latest["stored_location"] == "data/pdfs/a.pdf"
        assert latest["http_status"] == 200
        assert latest["error"] is None


def test_upsert_structured_round_trips_json_tree(tmp_db: Path):
    tree = {
        "title": "Policy A",
        "payer": "Example Health",
        "rules": {
            "rule_id": "1",
            "rule_text": "Meet ALL of:",
            "operator": "AND",
            "rules": [
                {"rule_id": "1.1", "rule_text": "leaf one"},
                {"rule_id": "1.2", "rule_text": "leaf two"},
            ],
        },
    }
    with db_module.connect() as conn:
        policy_id = db_module.upsert_policy(
            conn,
            title="Policy A",
            pdf_url="https://example.com/a.pdf",
            source_page_url="https://example.com/list",
        )
        db_module.upsert_structured(
            conn,
            policy_id=policy_id,
            extracted_text="initial chunk text",
            structured_json=tree,
            llm_metadata={"model": "gpt-4o-mini", "note": "initial heading found"},
            validation_error=None,
        )
        conn.commit()

        got = db_module.get_structured(conn, policy_id)

    assert got is not None
    assert got["policy_id"] == policy_id
    assert got["extracted_text"] == "initial chunk text"
    assert got["validation_error"] is None
    assert got["llm_metadata"]["model"] == "gpt-4o-mini"
    assert got["structured_json"] == tree


def test_upsert_structured_replaces_on_rerun(tmp_db: Path):
    """One row per policy: rerunning structuring must REPLACE, not duplicate."""
    with db_module.connect() as conn:
        policy_id = db_module.upsert_policy(
            conn,
            title="Policy A",
            pdf_url="https://example.com/a.pdf",
            source_page_url="https://example.com/list",
        )
        # First: a failure
        db_module.upsert_structured(
            conn,
            policy_id=policy_id,
            extracted_text=None,
            structured_json=None,
            llm_metadata={"note": "fallback"},
            validation_error="empty extracted text",
        )
        # Second: success replaces
        good_tree = {
            "title": "Policy A",
            "payer": "Example Health",
            "rules": {"rule_id": "1", "rule_text": "leaf"},
        }
        db_module.upsert_structured(
            conn,
            policy_id=policy_id,
            extracted_text="good text",
            structured_json=good_tree,
            llm_metadata={"model": "gpt-4o-mini"},
            validation_error=None,
        )
        conn.commit()

        count = conn.execute(
            "SELECT COUNT(*) AS c FROM structured_policies WHERE policy_id = ?",
            (policy_id,),
        ).fetchone()["c"]
        got = db_module.get_structured(conn, policy_id)

    assert count == 1
    assert got["validation_error"] is None
    assert got["structured_json"] == good_tree
    assert got["extracted_text"] == "good text"
