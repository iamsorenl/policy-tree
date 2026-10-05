"""PDF -> text extraction.

Primary extractor uses ``pdfplumber`` for higher-quality word/line spacing
(pypdf tends to fuse words on some PDFs, e.g. ``"Clinical GuidelineAcme
Clinical Guideline:..."``). Falls back to ``pypdf`` when pdfplumber returns
empty text. Robust to corrupt / encrypted / empty PDFs (returns ``""`` and
logs a warning rather than raising).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pdfplumber
from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)


def _clean_page(text: str) -> str:
    """Collapse runs of blank lines and trim trailing whitespace per line."""
    if not text:
        return ""
    # Normalize newlines, strip per-line trailing whitespace
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    # Collapse 3+ consecutive blank lines into a single blank line
    cleaned: list[str] = []
    blank_run = 0
    for ln in lines:
        if not ln.strip():
            blank_run += 1
            if blank_run <= 1:
                cleaned.append("")
        else:
            blank_run = 0
            cleaned.append(ln)
    return "\n".join(cleaned).strip()


def _extract_with_pdfplumber(pdf_path: Path) -> str:
    """Try pdfplumber. Returns "" on any failure (logs a warning)."""
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            page_texts: list[str] = []
            for i, page in enumerate(pdf.pages):
                try:
                    raw = page.extract_text() or ""
                except Exception as exc:  # noqa: BLE001 - pdfplumber raises a variety
                    logger.warning(
                        "extract_text[pdfplumber]: page %d extraction failed in %s: %s",
                        i,
                        pdf_path,
                        exc,
                    )
                    raw = ""
                cleaned = _clean_page(raw)
                if cleaned:
                    page_texts.append(cleaned)
    except Exception as exc:  # noqa: BLE001 - pdfplumber bubbles many error types
        logger.warning("extract_text[pdfplumber]: failed to open %s: %s", pdf_path, exc)
        return ""

    if not page_texts:
        return ""

    full = "\n\n".join(page_texts)
    full = re.sub(r"\n{3,}", "\n\n", full).strip()
    return full


def _extract_with_pypdf(pdf_path: Path) -> str:
    """Fallback extractor using pypdf. Returns "" on failure."""
    try:
        reader = PdfReader(str(pdf_path))
    except (PdfReadError, OSError, ValueError) as exc:
        logger.warning("extract_text[pypdf]: failed to open %s: %s", pdf_path, exc)
        return ""

    if reader.is_encrypted:
        try:
            if reader.decrypt("") == 0:
                logger.warning("extract_text[pypdf]: encrypted PDF, cannot decrypt: %s", pdf_path)
                return ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("extract_text[pypdf]: decrypt failed for %s: %s", pdf_path, exc)
            return ""

    page_texts: list[str] = []
    for i, page in enumerate(reader.pages):
        try:
            raw = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "extract_text[pypdf]: page %d extraction failed in %s: %s", i, pdf_path, exc
            )
            raw = ""
        cleaned = _clean_page(raw)
        if cleaned:
            page_texts.append(cleaned)

    if not page_texts:
        return ""

    full = "\n\n".join(page_texts)
    full = re.sub(r"\n{3,}", "\n\n", full).strip()
    return full


def extract_text(pdf_path: Path) -> str:
    """Return the textual content of a PDF, or "" on failure.

    Behavior:
      - Primary path: pdfplumber (better word/line spacing on most PDFs).
      - Fallback: pypdf, used only if pdfplumber returns empty.
      - Encrypted PDFs (without an empty password) -> "" + warning.
      - Corrupt / unreadable PDFs -> "" + warning.
      - Empty PDFs -> "".
      - Excess whitespace and per-page blank-line runs are collapsed.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        logger.warning("extract_text: file does not exist: %s", pdf_path)
        return ""

    text = _extract_with_pdfplumber(pdf_path)
    if text:
        return text

    logger.info(
        "extract_text: pdfplumber returned empty for %s; falling back to pypdf",
        pdf_path,
    )
    text = _extract_with_pypdf(pdf_path)
    if not text:
        logger.warning("extract_text: no text extracted from %s", pdf_path)
    return text


if __name__ == "__main__":
    # Manual smoke check: python -m app.structure.extract path/to/file.pdf
    import sys

    logging.basicConfig(level=logging.INFO)
    target = Path(sys.argv[1])
    text = extract_text(target)
    print(f"Extracted {len(text)} chars from {target}")
    print("--- first 600 chars ---")
    print(text[:600])
    sys.exit(0 if text else 1)
