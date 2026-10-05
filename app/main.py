from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.db import connect, get_policy, get_structured, init_db, list_policies

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Policy Tree", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def policy_list(request: Request):
    with connect() as conn:
        rows = list_policies(conn)
    total = len(rows)
    structured_ok = sum(1 for r in rows if r["has_structure"] and not r["validation_error"])
    structured_warn = sum(1 for r in rows if r["has_structure"] and r["validation_error"])
    # Approximate "downloaded" via successful download presence; count via SQL.
    with connect() as conn:
        downloaded = conn.execute(
            "SELECT COUNT(DISTINCT policy_id) AS n FROM downloads WHERE error IS NULL AND stored_location IS NOT NULL"
        ).fetchone()["n"]
    return templates.TemplateResponse(
        request,
        "policies.html",
        {
            "policies": rows,
            "stats": {
                "total": total,
                "downloaded": downloaded,
                "structured": structured_ok + structured_warn,
                "structured_ok": structured_ok,
                "structured_warn": structured_warn,
            },
        },
    )


def _parse_expand(expand: str | None) -> int:
    """Parse the ?expand= query param into a tree depth.

    - None -> 1 (default: root + immediate children open)
    - "all" -> 999 (sentinel meaning every <details> renders open)
    - integer string -> clamped to [0, 10]
    - anything else -> 1 (graceful fallback)
    """
    if expand is None:
        return 1
    value = expand.strip().lower()
    if value == "all":
        return 999
    try:
        depth = int(value)
    except (TypeError, ValueError):
        return 1
    if depth < 0:
        return 0
    if depth > 10:
        return 10
    return depth


@app.get("/policies/{policy_id}", response_class=HTMLResponse)
def policy_detail(request: Request, policy_id: int, expand: str | None = None):
    with connect() as conn:
        policy = get_policy(conn, policy_id)
        if policy is None:
            raise HTTPException(status_code=404, detail="Policy not found")
        structured = get_structured(conn, policy_id)
    expand_depth = _parse_expand(expand)
    return templates.TemplateResponse(
        request,
        "policy_detail.html",
        {
            "policy": policy,
            "structured": structured,
            "expand_depth": expand_depth,
        },
    )
