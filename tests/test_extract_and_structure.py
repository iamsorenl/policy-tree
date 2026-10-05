"""PDF extraction and the LLM structuring loop, using a fake client (no API key)."""
from __future__ import annotations

import json
from types import SimpleNamespace

from app.structure import llm
from app.structure.extract import extract_text
from app.structure.run import select_initial_chunk
from tests.conftest import FIXTURES, make_pdf

TREE = (FIXTURES / "sample_tree.json").read_text()


def test_extract_text_from_synthetic_pdf(tmp_path):
    lines = (FIXTURES / "policy_text.txt").read_text().splitlines()
    text = extract_text(make_pdf(tmp_path / "p.pdf", lines))
    assert "Initial Criteria" in text and "knee instability" in text


def test_extract_text_bad_file_returns_empty(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    assert extract_text(bad) == ""
    assert extract_text(tmp_path / "missing.pdf") == ""


def test_initial_chunk_from_fixture_text():
    chunk, note = select_initial_chunk((FIXTURES / "policy_text.txt").read_text())
    assert "knee instability" in chunk and "documented improvement" not in chunk
    assert note == "initial heading found"


class FakeClient:
    """Returns canned replies in order, shaped like the OpenAI SDK."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        msg = SimpleNamespace(content=self.replies.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=None)


def _run(monkeypatch, replies):
    fake = FakeClient(replies)
    monkeypatch.setattr(llm, "_client", fake)
    return fake, llm.structure(text="1. criteria", title="Knee Brace Coverage", note="n")


def test_structure_valid_first_try(monkeypatch):
    fake, result = _run(monkeypatch, [TREE])
    assert result.validation_error is None and fake.calls == 1
    assert result.data["rules"]["rules"][2]["operator"] == "OR"


def test_structure_retries_after_schema_error(monkeypatch):
    bad = json.dumps({"title": "x", "rules": {"rule_id": "1", "rule_text": "t", "operator": "AND"}})
    fake, result = _run(monkeypatch, [bad, TREE])
    assert result.validation_error is None and fake.calls == 2


def test_structure_gives_up_after_second_failure(monkeypatch):
    bad = json.dumps({"title": "x", "rules": {"rule_id": "1", "rule_text": "t", "operator": "AND"}})
    fake, result = _run(monkeypatch, [bad, bad])
    assert result.data is None and result.validation_error.startswith("Schema:")


def test_base_url_env_is_passed_to_client(monkeypatch):
    seen = {}
    monkeypatch.setattr(llm, "OpenAI", lambda **kw: seen.update(kw) or object())
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    llm._client_singleton()
    assert seen == {"base_url": "http://localhost:11434/v1"}
