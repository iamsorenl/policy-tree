"""Driver for the structuring pipeline.

Picks N policies needing structure from the DB, extracts text from the stored
PDFs, slices to the "Initial" criteria block, calls the LLM structurer, and
persists the result with `upsert_structured`.

Usage:
    python -m app.structure.run --count 10
    python -m app.structure.run --count 0   # no-op (smoke test)

Never crashes the whole run on a single policy failure — every error is
recorded as `validation_error` on the structured row and surfaced in the
final summary.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from app.db import connect, init_db, policies_needing_structure, upsert_structured
from app.structure.extract import extract_text
from app.structure.llm import structure as llm_structure

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_PAYER = "unknown"

# Heading patterns. We anchor at line start (multiline mode) and tolerate
# leading whitespace + section-number prefixes (e.g. "1. Initial Criteria").
INITIAL_HEADING_RE = re.compile(
    r"(?im)^[ \t]*(?:[\dIVX]+[\.\)]\s*)?"
    r"(initial(?:[ \t]+(?:criteria|authorization|coverage|approval|review|requirements?))?)\b.*$"
)
END_HEADING_RE = re.compile(
    r"(?im)^[ \t]*(?:[\dIVX]+[\.\)]\s*)?"
    r"(continuation|reauthorization|re-?authorization|continued\s+coverage|"
    r"renewal|ongoing|maintenance)\b.*$"
)
NUMBERED_BLOCK_RE = re.compile(r"(?m)^[ \t]*1[\.\)]\s+\S")

# Cap on the size of the chunk we send to the LLM when the heading-based
# slicer falls back to a coarser strategy. gpt-4o-mini supports a 128K-token
# context window (~ a few hundred K chars), so 80K is comfortably under the
# limit while still letting full-length policies through. We previously
# capped at 30K, which silently truncated several real medical-necessity
# policies (e.g. long drug policy docs at exactly 30K
# extracted_text) once they hit the fallback path. The cap exists only to
# bound the request when there is no recognizable structure at all, so
# drug-policy boilerplate doesn't blow up an open-ended prompt.
FALLBACK_CHAR_CAP = 80_000


def select_initial_chunk(text: str) -> tuple[str, str]:
    """Return (chunk, note) for the "initial-only" slice.

    Strategy:
      1. Find the first "Initial[/Criteria/Authorization/...]" heading.
         If multiple are found, slice from the first one and stop at the next
         disjoint Initial heading or at the next Continuation/Reauth heading.
      2. If no Initial heading: fall back to the first numbered list block
         (first occurrence of a line starting "1." or "1)").
      3. Last resort: return the full text.
    """
    if not text:
        return "", "empty input"

    initial_matches = list(INITIAL_HEADING_RE.finditer(text))
    if initial_matches:
        first = initial_matches[0]
        start = first.start()
        # Search for the end after the first heading line.
        search_from = first.end()
        end = len(text)

        # Stop at next Continuation/Reauthorization heading
        end_match = END_HEADING_RE.search(text, search_from)
        if end_match:
            end = min(end, end_match.start())

        # Or stop at the next disjoint Initial pathway (>1 found)
        if len(initial_matches) > 1:
            second_start = initial_matches[1].start()
            if second_start > search_from:
                end = min(end, second_start)
            note = (
                f"selected first of {len(initial_matches)} initial pathways"
            )
        else:
            note = "initial heading found"

        chunk = text[start:end].strip()
        if chunk:
            return chunk, note

    # Fallback: first numbered-list block
    nb = NUMBERED_BLOCK_RE.search(text)
    if nb:
        chunk = text[nb.start():].strip()
        # Cap fallback to a reasonable size to keep the LLM input bounded
        if len(chunk) > FALLBACK_CHAR_CAP:
            chunk = chunk[:FALLBACK_CHAR_CAP]
        return chunk, "fallback: first numbered block"

    # Last resort
    cap = text[:FALLBACK_CHAR_CAP].strip()
    return cap, "fallback: full text (no headings found)"


def process_one(conn, policy, payer: str = DEFAULT_PAYER) -> tuple[bool, str, str]:
    """Returns (ok, title, message)."""
    policy_id = policy["id"]
    title = policy["title"]
    stored = policy["stored_location"]
    if not stored:
        upsert_structured(
            conn,
            policy_id=policy_id,
            extracted_text=None,
            structured_json=None,
            llm_metadata={"note": "no stored_location"},
            validation_error="no stored_location",
        )
        return False, title, "no stored_location"

    pdf_path = Path(stored)
    text = extract_text(pdf_path)
    if not text:
        upsert_structured(
            conn,
            policy_id=policy_id,
            extracted_text=None,
            structured_json=None,
            llm_metadata={"note": "extraction returned empty"},
            validation_error="empty extracted text",
        )
        return False, title, "empty extracted text"

    chunk, note = select_initial_chunk(text)

    result = llm_structure(text=chunk, title=title, note=note)

    if result.data is not None:
        # Enforce the constant on our side, after validation passed.
        result.data["payer"] = payer

    upsert_structured(
        conn,
        policy_id=policy_id,
        extracted_text=chunk,
        structured_json=result.data,
        llm_metadata=result.llm_metadata,
        validation_error=result.validation_error,
    )
    conn.commit()

    if result.validation_error:
        return False, title, result.validation_error
    return True, title, note


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Structure N policies' Initial criteria via the LLM pipeline."
    )
    parser.add_argument(
        "--count", type=int, default=10,
        help="Number of policies to structure (default: 10). 0 = no-op.",
    )
    parser.add_argument(
        "--payer", default=os.getenv("PAYER_NAME", DEFAULT_PAYER),
        help="Payer name stored on each tree (default: PAYER_NAME env).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.count <= 0:
        print("count <= 0; nothing to do.")
        return 0

    init_db()
    succeeded: list[str] = []
    failed: list[tuple[str, str]] = []

    with connect() as conn:
        rows = policies_needing_structure(conn, args.count)
        if not rows:
            print("No policies need structuring (none with successful download + missing structured row).")
            return 0
        print(f"Structuring {len(rows)} policy(ies)...")
        for row in rows:
            try:
                ok, title, msg = process_one(conn, row, args.payer)
            except Exception as exc:  # noqa: BLE001 - never crash the run
                logger.exception("process_one crashed for policy %s", row["id"])
                title = row["title"]
                ok = False
                msg = f"unhandled: {exc}"
                # Best-effort: record failure
                try:
                    upsert_structured(
                        conn,
                        policy_id=row["id"],
                        extracted_text=None,
                        structured_json=None,
                        llm_metadata={"note": "unhandled exception"},
                        validation_error=msg,
                    )
                    conn.commit()
                except Exception:
                    logger.exception("failed to record error for policy %s", row["id"])
            if ok:
                succeeded.append(title)
                print(f"  OK   [{row['id']}] {title}  ({msg})")
            else:
                failed.append((title, msg))
                print(f"  FAIL [{row['id']}] {title}  ({msg})")

    print()
    print(f"Summary: {len(succeeded)} succeeded, {len(failed)} failed.")
    if succeeded:
        print("Structured titles:")
        for t in succeeded:
            print(f"  - {t}")
    if failed:
        print("Failures:")
        for t, m in failed:
            print(f"  - {t}: {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
