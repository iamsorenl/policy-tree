"""Tests for `select_initial_chunk` — the initial-only heuristic in run.py.

Strategy under test (from app/structure/run.py):
  1. Find first 'Initial[/Criteria/Authorization/...]' heading.
  2. End at next Continuation/Reauthorization heading OR next disjoint Initial.
  3. Fallback to first numbered-list block (cap 80K chars) if no Initial heading.
  4. Last resort: first 80K chars of full text.
"""

from __future__ import annotations

from app.structure.run import select_initial_chunk


def test_initial_then_continuation_returns_only_initial():
    text = (
        "Some preamble.\n"
        "Initial Criteria\n"
        "1. Member must meet ALL of the following:\n"
        "   a. condition X\n"
        "   b. condition Y\n"
        "\n"
        "Continuation Criteria\n"
        "1. For ongoing approval, member must show response.\n"
        "   a. lab Z improved\n"
    )
    chunk, note = select_initial_chunk(text)
    assert "Initial Criteria" in chunk
    assert "condition X" in chunk
    assert "condition Y" in chunk
    # The continuation block must NOT bleed in
    assert "Continuation" not in chunk
    assert "lab Z improved" not in chunk
    assert note == "initial heading found"


def test_no_initial_heading_falls_back_to_first_numbered_block():
    """No 'Initial' heading in text => fallback fires on the first '1.' line."""
    text = (
        "Policy summary intro paragraph without any heading.\n"
        "More prose, more prose.\n"
        "1. The member must have a documented diagnosis.\n"
        "2. The member must have failed first-line therapy.\n"
        "3. Provider attestation submitted.\n"
    )
    chunk, note = select_initial_chunk(text)
    # Note the implementation slices from the '1.' line onward
    assert note == "fallback: first numbered block"
    assert chunk.startswith("1. The member must have a documented diagnosis")
    assert "first-line therapy" in chunk


def test_two_initial_pathways_returns_first_one_with_note():
    """Multiple 'Initial' headings => first pathway, note records the count."""
    text = (
        "Indication A:\n"
        "Initial Criteria\n"
        "1. Adult member with condition A.\n"
        "2. ECOG 0-2.\n"
        "\n"
        "Indication B:\n"
        "Initial Criteria\n"
        "1. Pediatric member with condition B.\n"
        "2. Caregiver attestation.\n"
    )
    chunk, note = select_initial_chunk(text)
    # First pathway content present
    assert "condition A" in chunk
    assert "ECOG 0-2" in chunk
    # Second pathway content NOT present
    assert "condition B" not in chunk
    assert "Pediatric" not in chunk
    # Note records that there were multiple pathways
    assert "initial pathways" in note
    assert "2" in note  # "selected first of 2 initial pathways"


def test_initial_with_numbered_prefix_is_recognized():
    """'1. Initial Criteria' (numbered heading) should match."""
    text = (
        "Header garbage.\n"
        "1. Initial Criteria\n"
        "Member must meet ALL of:\n"
        "  a. criterion one\n"
        "  b. criterion two\n"
        "\n"
        "2. Continuation\n"
        "renewal stuff\n"
    )
    chunk, note = select_initial_chunk(text)
    assert "criterion one" in chunk
    assert "renewal stuff" not in chunk
    assert note == "initial heading found"


def test_empty_input_returns_empty_with_note():
    chunk, note = select_initial_chunk("")
    assert chunk == ""
    assert note == "empty input"


def test_initial_heading_no_continuation_includes_to_end():
    """If no Continuation, the chunk extends to end-of-text."""
    text = (
        "preamble\n"
        "Initial Criteria\n"
        "1. Member criterion alpha.\n"
        "2. Member criterion beta.\n"
        "End of policy.\n"
    )
    chunk, note = select_initial_chunk(text)
    assert "alpha" in chunk
    assert "beta" in chunk
    assert "End of policy" in chunk
    assert note == "initial heading found"
