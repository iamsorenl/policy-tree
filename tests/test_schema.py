"""Schema validation tests for app.structure.llm.{Rule, Document}.

Anchors the contract for the Pydantic recursive schema:
  - a sample tree fixture validates clean
  - leaves are minimal (rule_id + rule_text only)
  - non-leaves require BOTH operator AND non-empty rules
  - title is required at top level
  - payer defaults to "unknown" and accepts any name
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.structure.llm import Document, Rule

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_JSON = REPO_ROOT / "tests" / "fixtures" / "sample_tree.json"


def test_sample_tree_validates_cleanly():
    """The sample tree must validate against our schema."""
    with SAMPLE_JSON.open() as f:
        data = json.load(f)
    doc = Document.model_validate(data)
    assert doc.title == "Knee Brace Coverage"
    assert doc.payer == "Example Health"
    assert doc.rules.rule_id == "1"
    assert doc.rules.operator == "AND"
    assert doc.rules.rules is not None and len(doc.rules.rules) == 3


def test_leaf_node_is_valid():
    """A leaf has only rule_id + rule_text. No operator, no rules."""
    leaf = Rule.model_validate({"rule_id": "1.1", "rule_text": "Some criterion"})
    assert leaf.rule_id == "1.1"
    assert leaf.operator is None
    assert leaf.rules is None


def test_non_leaf_with_operator_and_rules_is_valid():
    """Non-leaf must have BOTH an operator and a non-empty rules list."""
    node = Rule.model_validate({
        "rule_id": "1",
        "rule_text": "All of the following:",
        "operator": "AND",
        "rules": [
            {"rule_id": "1.1", "rule_text": "First leaf"},
            {"rule_id": "1.2", "rule_text": "Second leaf"},
        ],
    })
    assert node.operator == "AND"
    assert node.rules is not None and len(node.rules) == 2


def test_non_leaf_with_operator_but_empty_rules_raises():
    """An operator without rules is invalid."""
    with pytest.raises(ValidationError) as excinfo:
        Rule.model_validate({
            "rule_id": "1",
            "rule_text": "Empty branch",
            "operator": "AND",
            "rules": [],
        })
    msg = str(excinfo.value)
    assert "operator" in msg and "rules" in msg


def test_non_leaf_with_operator_and_null_rules_raises():
    """Same constraint: operator present, rules missing/null is invalid."""
    with pytest.raises(ValidationError):
        Rule.model_validate({
            "rule_id": "1",
            "rule_text": "Operator without children",
            "operator": "OR",
        })


def test_leaf_with_rules_but_no_operator_raises():
    """Inverse: rules without operator is also invalid."""
    with pytest.raises(ValidationError):
        Rule.model_validate({
            "rule_id": "1",
            "rule_text": "Children without operator",
            "rules": [{"rule_id": "1.1", "rule_text": "leaf"}],
        })


def test_top_level_title_required():
    """Document.title must be present."""
    with pytest.raises(ValidationError) as excinfo:
        Document.model_validate({
            "rules": {"rule_id": "1", "rule_text": "leaf"},
        })
    assert "title" in str(excinfo.value)


def test_payer_defaults_to_unknown():
    """payer has default 'unknown' when omitted."""
    doc = Document.model_validate({
        "title": "Some Policy",
        "rules": {"rule_id": "1", "rule_text": "leaf"},
    })
    assert doc.payer == "unknown"


def test_payer_accepts_explicit_name():
    """An explicit payer name is kept."""
    doc = Document.model_validate({
        "title": "Some Policy",
        "payer": "Example Health",
        "rules": {"rule_id": "1", "rule_text": "leaf"},
    })
    assert doc.payer == "Example Health"


def test_operator_must_be_and_or_or():
    """Literal['AND', 'OR'] — anything else rejected."""
    with pytest.raises(ValidationError):
        Rule.model_validate({
            "rule_id": "1",
            "rule_text": "Bad operator",
            "operator": "XOR",
            "rules": [{"rule_id": "1.1", "rule_text": "leaf"}],
        })
