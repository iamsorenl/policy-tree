# policy-tree

Turn an insurer's published medical policies into data you can browse and query.

## What it does

This project reads a listing page of medical policies, finds the PDF behind each one, downloads it, and asks a language model to turn the coverage criteria into a tree. Each node in the tree is a rule. A rule is either a single condition or a group of rules joined by AND or OR. The trees are checked against a pydantic schema, and the results are stored in SQLite. A small web page lets you browse the policies and open each tree.

## Why

Insurers publish their medical policies as PDFs. When a doctor asks for prior authorization, the request is checked against the criteria written in those PDFs. Prose is hard to check by machine, so a tree of rules is a more useful shape. It can be read by a person, walked by code, and compared across policies.

## How it works

1. Discovery fetches the listing page, follows each policy link, and saves every PDF link it finds.
2. Download fetches each PDF with retries and a pause between requests. It skips files it already has.
3. Structure pulls the text out of each PDF with pdfplumber, and falls back to pypdf when that comes back empty. It cuts out the initial criteria section, sends it to the model, and validates the answer. A bad answer gets one retry with the error included. Failures are saved with their error message so they show up in the UI.
4. The UI lists every policy and shows its tree.

Every step can be run again safely. Policies are keyed by PDF URL, so reruns do not create duplicates.

## Setup

You need Python 3.11 or newer.

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
cp .env.example .env
```

Then edit `.env`.

## Configuration

These are read from the environment or from `.env`.

```
SOURCE_URL          listing page to crawl (or pass --source-url)
PAYER_NAME          name stored on each tree (or pass --payer)
OPENAI_API_KEY      API key for the model
OPENAI_BASE_URL     optional, point at another OpenAI compatible server
OPENAI_MODEL        optional, defaults to gpt-4o-mini
PDF_HOST_ALLOWLIST  optional, comma separated hosts allowed to serve PDFs
DETAIL_PATH_PREFIX  optional, only follow listing links under this path
```

By default discovery follows links on the same site that sit in the same folder as the listing page. It keeps any PDF link on the same site, including a CDN host that shares the site's domain. If your PDFs live on an unrelated host, set `PDF_HOST_ALLOWLIST`. If the policy pages live in a different folder than the listing, set `DETAIL_PATH_PREFIX`.

To use Groq, Ollama, or another compatible server, set `OPENAI_BASE_URL` and `OPENAI_MODEL` and use whatever key that server expects. For a local Ollama server the key can be any non empty string.

```
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=qwen3-coder:30b
OPENAI_API_KEY=ollama
```

## Run

```bash
python -m app.scrape.discover --source-url https://example.com/policies/index.html
python -m app.scrape.download
python -m app.structure.run --count 10 --payer "Example Health"
uvicorn app.main:app --reload
```

Then open `localhost:8000` in a browser. The database lives in `data/policies.db` and the PDFs in `data/pdfs/`. Both are ignored by git.

Use `--count` to limit how many policies are sent to the model in one run. Run structure again to pick up policies that still need a tree.

## Tests

```bash
pip install pytest
pytest
```

The tests run offline and need no API key. They use made up pages and a generated PDF from `tests/fixtures/`, and the model is replaced by a fake client that returns canned answers.

## Layout

```
app/scrape/discover.py    find policy PDFs
app/scrape/download.py    download them
app/structure/extract.py  PDF to text
app/structure/llm.py      prompt, schema, validation, retry
app/structure/run.py      pick the initial criteria and store the tree
app/db.py                 SQLite schema and queries
app/main.py               web UI
```

## License

MIT
