"""LLM-driven structuring of medical-necessity criteria into a Rule tree.

Pipeline:
  1. Build a prompt instructing the model to emit JSON only, matching a
     recursive Rule schema.
  2. Call OpenAI with response_format=json_object.
  3. Parse JSON. On parse failure, retry once with a corrective follow-up.
  4. Validate with Pydantic. On validation failure, retry once with the error
     message inlined.
  5. Return a StructureResult dataclass. Never raise.

`payer` is intentionally excluded from the LLM contract. The caller
(run.py) sets it from PAYER_NAME after validation passes.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, model_validator

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
PROMPT_VERSION = "v2"  # v2: explicit AND/OR mapping rules + NONE-of pattern + connector-word guidance


# ---------- Pydantic schema ----------

class Rule(BaseModel):
    rule_id: str
    rule_text: str
    operator: Optional[Literal["AND", "OR"]] = None
    rules: Optional[list["Rule"]] = None

    @model_validator(mode="after")
    def _check_operator_rules_consistency(self) -> "Rule":
        has_op = self.operator is not None
        has_kids = bool(self.rules)
        if has_op != has_kids:
            # Either both must be present (non-leaf) or both absent (leaf).
            raise ValueError(
                f"rule {self.rule_id!r}: 'operator' and non-empty 'rules' must "
                "both be present (non-leaf) or both absent (leaf)"
            )
        return self


class Document(BaseModel):
    title: str
    payer: str = "unknown"
    rules: Rule


Rule.model_rebuild()


# ---------- Result type ----------

@dataclass
class StructureResult:
    data: Optional[dict] = None
    validation_error: Optional[str] = None
    llm_metadata: dict = field(default_factory=dict)


# ---------- Prompts ----------

SYSTEM_PROMPT = """You are a clinical-policy structuring assistant. You read medical-necessity criteria from health-insurance clinical guidelines and convert them into a strict, hierarchical decision tree.

You MUST output a single JSON object. No prose. No markdown fences. No commentary.

## Schema (recursive)

{
  "title": <string>,                          // policy title
  "rules": <Rule>                             // root node
}

Each Rule node:
{
  "rule_id": <string>,                        // hierarchical id like "1", "1.1", "1.2.2.3"
  "rule_text": <string>,                      // the criterion text
  "operator": <"AND" | "OR" | null>,          // ONLY on non-leaf nodes
  "rules": [<Rule>, ...] | null               // ONLY on non-leaf nodes
}

## Hard rules — read carefully

1. **Leaf nodes** (atomic criteria) have ONLY `rule_id` and `rule_text`. Do NOT include `operator` or `rules`. Omit them, or set them to `null`. Never put an empty array.

2. **Non-leaf nodes** (groups) MUST include BOTH:
   - an `operator` of either `"AND"` or `"OR"` that joins their children, and
   - a non-empty `rules` array of child nodes.
   Never have one without the other.

3. **AND vs OR — derive from the SOURCE TEXT, do not guess:**
   - Use **AND** when the source says any of:
     - "ALL of the following"
     - "all of the following criteria/conditions are met"
     - "the member meets all of the following"
     - children joined by "; and" or by no explicit conjunction in a list of co-required criteria
   - Use **OR** when the source says any of:
     - "ANY of the following"
     - "ONE of the following" / "at least ONE of the following"
     - "EITHER ... OR ..."
     - children joined by "; or"
     - the parent introduces a list of distinct *indications* / *pathways* / *alternatives* / *contraindications*

   **When the header is ambiguous, look at the connector words BETWEEN the children.** "; or" -> OR. "; and" or no conjunction -> usually AND, but if the items are alternatives (e.g. distinct indications, distinct contraindications listed for OR-of-disqualifiers), use OR. Read the bullets — never default blindly.

4. **"NONE of the following" pattern**: if a criterion is "The member has NONE of the following [contraindications]: ..." it is a non-leaf with operator `"OR"` over the contraindications (because the parent assertion fails if ANY child is true). Keep the parent text as written; do not invert the children.

5. **Non-leaf "header" criteria** — when a criterion ends with a colon or with phrases like "the following", "of", "with", "including", and is followed by a list, that header IS the `rule_text` of a non-leaf, and the listed items are its children.
   Example:
     Source: "Adult aged 18 years or older with documentation of ONE of:
              - BMI >=40
              - BMI >=35 with severe obesity-related comorbidity"
     Output:
       {
         "rule_id": "1.2",
         "rule_text": "Adult aged 18 years or older with documentation of",
         "operator": "OR",
         "rules": [
           { "rule_id": "1.2.1", "rule_text": "BMI >=40" },
           { "rule_id": "1.2.2", "rule_text": "BMI >=35 with severe obesity-related comorbidity" }
         ]
       }

6. **Numbering**: hierarchical, dot-separated. Root = `"1"`. Children of `"1"` = `"1.1"`, `"1.2"`, `"1.3"`, ... Grandchildren of `"1.2"` = `"1.2.1"`, `"1.2.2"`, ... Stay consistent — never skip a level or repeat an id.

7. **Preserve wording**. Keep each criterion's text close to the original. Light whitespace cleanup only. Do not invent criteria that are not in the source. Do not summarize or paraphrase.

8. **Top level**: only `title` and `rules`. Do NOT add `payer` or any other field.

Output a single JSON object and nothing else.
"""


def _build_user_prompt(text: str, title: str) -> str:
    return (
        f"Policy title: {title}\n\n"
        "Below is the extracted text of the policy's INITIAL medical-necessity criteria. "
        "Convert it into the hierarchical decision tree described in the system instructions. "
        "Output only the JSON object.\n\n"
        "---BEGIN POLICY TEXT---\n"
        f"{text}\n"
        "---END POLICY TEXT---\n"
    )


# ---------- Core ----------

_client: Optional[OpenAI] = None


def _client_singleton() -> OpenAI:
    global _client
    if _client is None:
        # OPENAI_API_KEY is read by the SDK. OPENAI_BASE_URL (optional) points
        # at any OpenAI-compatible endpoint such as Groq or Ollama.
        _client = OpenAI(base_url=os.getenv("OPENAI_BASE_URL") or None)
    return _client


def _strip_payer(obj: Any) -> None:
    """The LLM is told not to emit payer; if it does anyway, drop it."""
    if isinstance(obj, dict) and "payer" in obj:
        obj.pop("payer", None)


def _parse_json(content: str) -> tuple[Optional[dict], Optional[str]]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        return None, f"{exc.msg} at line {exc.lineno} col {exc.colno}"
    if not isinstance(data, dict):
        return None, "top-level JSON must be an object"
    return data, None


def _validate(data: dict, title: str) -> tuple[Optional[dict], Optional[str]]:
    """Validate against Document schema, normalizing first."""
    candidate = dict(data)
    # Force the title we know (helps if the LLM renamed it)
    if not candidate.get("title"):
        candidate["title"] = title
    # The LLM should not have emitted payer; strip if present
    _strip_payer(candidate)
    # Pydantic Document requires payer; default fills it in
    try:
        doc = Document.model_validate(candidate)
    except ValidationError as exc:
        # Compact the error message
        return None, "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
            for e in exc.errors()
        )
    return doc.model_dump(exclude_none=True), None


def _lookup_rule_id(data: dict, loc: tuple) -> Optional[str]:
    """Walk the structured dict using a Pydantic error `loc` tuple and return
    the rule_id at that path, if reachable. `loc` looks like
    ('rules', 'rules', 0, 'rules', 1, ...) — alternating keys/indices.
    Returns None on any miss so the caller can fall back gracefully.
    """
    node: Any = data
    try:
        for part in loc:
            if isinstance(part, int):
                if not isinstance(node, list):
                    return None
                node = node[part]
            else:
                if not isinstance(node, dict):
                    return None
                # The error path may end on a field name like 'rules' that the
                # node itself doesn't contain (validator-level error). Bail
                # out once we run out of children to descend into.
                if part not in node:
                    break
                node = node[part]
        if isinstance(node, dict):
            return node.get("rule_id")
    except (KeyError, IndexError, TypeError):
        return None
    return None


def _build_schema_retry_message(data: dict, val_err: str, title: str) -> str:
    """Build a targeted retry message for schema validation failures.

    The validator's most common failure mode is the "operator XOR rules"
    invariant on a specific rule. Re-running validation here lets us pull the
    first failing path out of structured ValidationError data so we can name
    the offending rule_id back to the model — much more actionable than
    handing it the raw concatenated error string.
    """
    candidate = dict(data)
    if not candidate.get("title"):
        candidate["title"] = title
    _strip_payer(candidate)
    first_rule_id: Optional[str] = None
    first_loc: Optional[tuple] = None
    try:
        Document.model_validate(candidate)
    except ValidationError as exc:
        errors = exc.errors()
        if errors:
            first_loc = tuple(errors[0]["loc"])
            first_rule_id = _lookup_rule_id(candidate, first_loc)

    target = (
        f"rule '{first_rule_id}' (path: {'.'.join(str(p) for p in first_loc)})"
        if first_rule_id and first_loc
        else "the rule(s) flagged below"
    )
    return (
        "Your previous JSON did not satisfy the schema. "
        f"The first failure is at {target}.\n"
        "Each rule must be EITHER:\n"
        "  - a leaf: rule_id + rule_text only (no 'operator', no 'rules'); OR\n"
        "  - a non-leaf: rule_id + rule_text + 'operator' (\"AND\" or \"OR\") + a non-empty 'rules' array.\n"
        "Fix the flagged rule (and any siblings with the same shape problem): "
        "if it has children, add the missing 'operator'; if it has an "
        "'operator' but an empty/missing 'rules' array, either add child "
        "rules or convert it to a leaf by removing 'operator'.\n"
        f"Full validation error (for reference): {val_err}\n"
        "Return ONLY the corrected JSON object. Do not include 'payer'."
    )


def _call(client: OpenAI, model: str, messages: list[dict]) -> tuple[str, dict]:
    """Single chat completion call. Returns (content, usage_metadata)."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore[arg-type]
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = resp.choices[0].message.content or ""
    usage = {}
    if resp.usage:
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        }
    return content, usage


def structure(
    *,
    text: str,
    title: str,
    model: Optional[str] = None,
    note: Optional[str] = None,
) -> StructureResult:
    """Structure `text` into a validated Rule tree.

    Returns a StructureResult; never raises. On any failure (network, parse,
    validation) the validation_error field is populated and data is None.
    """
    model = model or DEFAULT_MODEL
    metadata: dict[str, Any] = {
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "note": note or "",
    }

    if not text or not text.strip():
        return StructureResult(
            data=None,
            validation_error="empty input text",
            llm_metadata=metadata,
        )

    try:
        client = _client_singleton()
    except Exception as exc:  # noqa: BLE001 - missing API key etc.
        logger.exception("structure: client init failed")
        return StructureResult(
            data=None,
            validation_error=f"client init: {exc}",
            llm_metadata=metadata,
        )

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(text, title)},
    ]

    # ---- attempt 1 ----
    try:
        content, usage = _call(client, model, messages)
    except Exception as exc:  # noqa: BLE001
        logger.exception("structure: LLM call failed")
        return StructureResult(
            data=None,
            validation_error=f"LLM call: {exc}",
            llm_metadata=metadata,
        )
    metadata["usage"] = usage

    data, parse_err = _parse_json(content)

    # ---- JSON retry ----
    if parse_err is not None:
        logger.warning("structure: JSON parse failed (attempt 1): %s", parse_err)
        retry_messages = messages + [
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    "Your previous response was not valid JSON. "
                    "Return ONLY a valid JSON object matching this schema."
                ),
            },
        ]
        try:
            content2, usage2 = _call(client, model, retry_messages)
        except Exception as exc:  # noqa: BLE001
            logger.exception("structure: retry LLM call failed")
            return StructureResult(
                data=None,
                validation_error=f"LLM call (retry): {exc}",
                llm_metadata=metadata,
            )
        metadata["usage_retry"] = usage2
        data, parse_err = _parse_json(content2)
        if parse_err is not None:
            return StructureResult(
                data=None,
                validation_error=f"JSON parse: {parse_err}",
                llm_metadata=metadata,
            )

    assert data is not None  # narrowed for mypy
    validated, val_err = _validate(data, title)

    # ---- schema retry ----
    if val_err is not None:
        logger.warning("structure: schema validation failed (attempt 1): %s", val_err)
        # Use the most recent assistant content for the retry context
        last_assistant = json.dumps(data)
        retry_messages = messages + [
            {"role": "assistant", "content": last_assistant},
            {
                "role": "user",
                "content": _build_schema_retry_message(data, val_err, title),
            },
        ]
        try:
            content3, usage3 = _call(client, model, retry_messages)
        except Exception as exc:  # noqa: BLE001
            logger.exception("structure: schema-retry LLM call failed")
            return StructureResult(
                data=None,
                validation_error=f"LLM call (schema retry): {exc}",
                llm_metadata=metadata,
            )
        metadata["usage_schema_retry"] = usage3
        data2, parse_err2 = _parse_json(content3)
        if parse_err2 is not None:
            return StructureResult(
                data=None,
                validation_error=f"Schema retry JSON parse: {parse_err2}",
                llm_metadata=metadata,
            )
        assert data2 is not None
        validated, val_err = _validate(data2, title)
        if val_err is not None:
            return StructureResult(
                data=None,
                validation_error=f"Schema: {val_err}",
                llm_metadata=metadata,
            )

    return StructureResult(
        data=validated,
        validation_error=None,
        llm_metadata=metadata,
    )
