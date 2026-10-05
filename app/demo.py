"""Load the made up policies in examples/ into the database so the UI has
something to show without an API key or a real insurer site.

Usage:
    python -m app.demo
    uvicorn app.main:app --reload
"""

import json
from pathlib import Path

from app.db import connect, init_db, upsert_policy, upsert_structured

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def main() -> None:
    init_db()
    with connect() as conn:
        for path in sorted(EXAMPLES.glob("*.json")):
            ex = json.loads(path.read_text())
            policy_id = upsert_policy(
                conn, title=ex["title"], pdf_url=ex["pdf_url"], source_page_url=ex["source_page_url"]
            )
            upsert_structured(
                conn,
                policy_id=policy_id,
                extracted_text=ex["extracted_text"],
                structured_json=ex["tree"],
                llm_metadata={"source": "example"},
                validation_error=None,
            )
            print(f"loaded {ex['title']}")
        conn.commit()
    print("Now run: uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
