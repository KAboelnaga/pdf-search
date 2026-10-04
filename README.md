# PDF Ingestion & Semantic Search API

Upload PDFs, split them into chunks, embed them with an open model, store them in Qdrant, and search them by meaning.

**Stack:** FastAPI · Qdrant · fastembed (`BAAI/bge-small-en-v1.5`, 384-dim, CPU) · pypdf · Docker Compose

## Contents

1. [Quick start](#1-quick-start)
2. [Running the tests](#2-running-the-tests)
3. [Design summary](#3-design-summary)
4. [Decisions and trade-offs](#4-decisions-and-trade-offs)
5. [Deviations from the swagger](#5-deviations-from-the-swagger)
6. [Edge cases tested](#6-edge-cases-tested)
7. [Limitations](#7-limitations)
8. [Next steps](#8-next-steps)

## 1. Quick start

**Requirements:** Docker with Compose V2, `curl`, ports 8000 and 6333 free.

```bash
./orchestrate.sh --action start       # build, start, wait until /health is ready
./orchestrate.sh --action terminate   # remove containers, volumes and networks
```

The first `start` takes a few minutes: the embedding model is downloaded into the image at build time, so the API never downloads anything at runtime.

- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs

### Ingest

The multipart field is always `input`.

```bash
# one PDF
curl -X POST http://localhost:8000/ingest/ -F "input=@data/sample.pdf"

# several PDFs in one request
curl -X POST http://localhost:8000/ingest/ \
  -F "input=@data/sample.pdf" -F "input=@data/pdfs/vector-databases.pdf"

# a directory: a path *inside the container*
curl -X POST http://localhost:8000/ingest/ -F "input=/data/pdfs"
```

```json
{"message": "Successfully ingested 1 PDF document.", "files": ["sample.pdf"]}
```

**Directory input:** the host folder `./data` is mounted read-only at `/data` in the API container. Put PDFs in `./data/<folder>` and send `/data/<folder>` (or just `<folder>`). Only `*.pdf` files directly inside it are read (not subfolders). Paths outside `/data` are rejected.

### Search

```bash
curl -X POST http://localhost:8000/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "How does semantic search work?"}'
```

```json
{"results": [
  {"document": "sample.pdf", "score": 0.86, "content": "Semantic search finds information by meaning ...", "page": 1}
]}
```

Results are the top 5 chunks by cosine similarity. `page` is an extra field (see [Deviations](#5-deviations-from-the-swagger)).

### Errors

Every error returns `{"error": "<message>"}`: 400 bad input, 413 too large, 503 busy or unhealthy, 500 unexpected failure.

### Configuration

All settings are environment variables with defaults, validated at startup (a bad value stops the app with a clear message). Set them under `api.environment` in `docker-compose.yml`.

| Variable | Default | Meaning |
|---|---|---|
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | 800 / 120 | characters per chunk / overlap between chunks |
| `TOP_K` | 5 | number of search results |
| `MAX_QUERY_CHARS` | 1000 | longest accepted query |
| `MAX_FILES` / `MAX_FILE_MB` / `MAX_TOTAL_MB` | 20 / 50 / 100 | limits per request |
| `MAX_CHUNKS` | 1500 | chunk limit per request (about 60 s of embedding) |
| `MAX_CONCURRENT_INGESTS` / `INGEST_WAIT_SECONDS` | 2 / 120 | ingest slots / how long a request waits for one |
| `EMBED_BATCH_SIZE` | 64 | chunks embedded and stored per batch |

### Logs

```bash
docker compose logs -f api
```

Every request is logged as one line (method, path, status, duration). Each ingest also logs its stats: files, pages, chunks, and the time spent parsing, embedding and storing. Unexpected errors are logged with a full traceback; the client gets only a short `{"error"}` message.

## 2. Running the tests

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

./orchestrate.sh --action terminate && ./orchestrate.sh --action start   # fresh stack
pytest -v                  # 49 tests: 30 unit + 19 API
pytest -v tools/suite.py   # the provided suite: 2 tests
```

- **Unit tests** (`tests/test_units.py`) need no running services: chunking, settings validation, upload limits, path safety (including a symlink pointing outside `/data`), PDF parsing (corrupt, encrypted, empty), deterministic IDs.
- **API tests** (`tests/test_api.py`) run against the live stack and are skipped if it isn't running: response shapes, known-answer retrieval (the right document is the top result), idempotent re-ingestion, 12 error cases.
- Run the API tests on a fresh stack so only the sample PDFs are indexed.

Last run: **49 passed**, and `tools/suite.py` **2 passed**.

## 3. Design summary

```
                         ┌──────────────────────── api container ─────────────────────────┐
 client ── POST /ingest/ ─▶ checks: field "input", limits (413), .pdf names, path in /data │
                         │   │                                                            │
                         │   ▼  semaphore: max 2 ingests at once (503 if no slot in 120s) │
                         │  phase 1 (no writes): parse (pypdf) → chunk 800/120 → IDs      │
                         │   │       any bad file → 400, nothing stored                   │
                         │   ▼                                                            │
                         │  phase 2: batches of 64 → embed (bge-small, ONNX) → upsert ────┼──▶ Qdrant
                         │                                                                │   (cosine, 384-dim)
 client ── POST /search/ ─▶ embed query (BGE query prefix) → query_points top 5 ──────────┼──▶ Qdrant
                         └────────────────────────────────────────────────────────────────┘
        ./data (host) ── mounted read-only ──▶ /data (directory input)
```

**Code layout**

| File | Responsibility |
|---|---|
| `app/main.py` | HTTP layer: app, lifespan, request-log middleware, error handlers, endpoints, semaphore |
| `app/ingestion.py` | pipeline (no FastAPI): limits, directory input, parse → chunk → IDs → embed → store |
| `app/pdf.py` | PDF/text extraction and `InvalidDocument` errors |
| `app/chunker.py` | sliding-window chunking |
| `app/embedder.py` | fastembed wrapper (passages vs queries) |
| `app/store.py` | Qdrant wrapper: create collection, upsert, search, ping |
| `app/config.py` | settings from env vars, validated at startup |

**Ingestion is all-or-nothing.** Every file is parsed and checked before anything is written, so 400/413 means nothing was stored and 200 means everything was.

**Re-ingestion is idempotent.** Each chunk's ID is `uuid5(sha256(text) + chunk index)`, so ingesting the same content again (or retrying after a failure) overwrites the same points instead of adding duplicates.

## 4. Decisions and trade-offs

| Decision | Why | Alternatives considered |
|---|---|---|
| **FastAPI** | async, request validation, auto `/docs` | Django REST (heavy for 2 endpoints), Flask (no async or validation built in) |
| **Qdrant** | one container, payloads stored with vectors, simple client, HNSW index | pgvector (needs Postgres tuning), Milvus (several services), Chroma (simpler, aimed at prototyping) |
| **bge-small-en-v1.5** via fastembed | open (MIT), 384 dims, strong retrieval for its size, ONNX on CPU with no PyTorch (smaller image) | all-MiniLM-L6-v2 (weaker retrieval), bge-base (≈2× slower), hosted APIs (not open, need keys) |
| **Model baked into the image** | no download at startup, works offline, predictable start time | download on first request (slow, can fail) |
| **pypdf** | pure Python, BSD license, handles encrypted PDFs | PyMuPDF (faster, but AGPL), pdfplumber (slower) |
| **800/120 character chunks** | ≈150–200 tokens, well inside the model's 512-token limit; overlap keeps sentences cut at a boundary | token-based or sentence-aware chunking (next steps) |
| **Synchronous ingest** | matches the swagger (200 "Successfully ingested"); documents are searchable right away | job queue + worker (next steps) |
| **Semaphore (2) + limits** | embedding uses all CPU cores and uploads sit in RAM; more parallel ingests only slow everyone | unlimited (one big upload can starve the server) |
| **Deterministic IDs** | idempotent retries, no duplicates | random UUIDs (every re-ingest duplicates) |
| **Settings validated at startup** | a bad env var fails at boot with a clear message, not on the first request | read lazily |

## 5. Deviations from the swagger

| What | Why |
|---|---|
| **413** for too many / too large files or too many chunks | the request is valid but too big; 400 would say "your input is wrong" |
| **503** when both ingest slots stay busy for 120 s | the server is temporarily busy; the client should retry later |
| **`page`** in search results | lets the user find the passage in the PDF; extra field, existing clients are unaffected |
| **`GET /health`** | readiness check that also checks Qdrant; `orchestrate.sh` waits for it |
| **Plain text named `.pdf` is accepted** | see below |

**Text uploaded as `.pdf`:** the swagger says "Only PDF files are accepted", but the provided `suite.py` uploads a plain-text file named `sample.pdf` and expects 200. Files are detected by content: a real PDF (starts with `%PDF-`) goes to pypdf, valid UTF-8 text is ingested as one page (with a warning in the log), anything else gets 400. A `.txt` filename is still rejected. In production I would put this fallback behind a config flag, off by default.

## 6. Edge cases tested

| Input | Result |
|---|---|
| wrong HTTP method / unknown route | 405 / 404 with `{"error"}` |
| missing `input` field, JSON instead of multipart | 400 |
| `.txt` file, empty file, random bytes named `.pdf`, corrupt PDF | 400 |
| password-protected PDF | 400 (a PDF encrypted with an empty password is opened normally) |
| scanned PDF with no text | 400, message suggests OCR |
| directory: empty string, `../../etc`, `/etc`, missing folder, no PDFs | 400 |
| symlink inside `/data` pointing outside it | skipped with a warning |
| over the file / size / chunk limits | 413 |
| search: no body, invalid JSON, `""`, `"   "`, a number, 1001 characters | 400 |
| same file ingested twice | 200, point count unchanged |
| 5 concurrent ingests | all 200, processed 2 at a time; a search during them returns in ≈0.05 s |
| 1 slot with a 1 s wait, 5 concurrent ingests | 1 × 200, 4 × 503 |
| Qdrant stopped | `/health` 503; search 500 `{"error": "Search processing failed."}` |

## 7. Limitations

- **No OCR**: scanned PDFs are rejected.
- **English-only model** (bge-small-en).
- **Same filename, edited content**: the new version is added, but the old version's chunks stay (IDs are based on content, not filename). Fix: delete a document's points by filename before re-ingesting.
- **Same content, different filename**: stored once, under the most recent filename.
- **Synchronous ingest**: a large upload keeps the HTTP connection open until it's done (capped at about 60 s of embedding by `MAX_CHUNKS`).
- **Mixed input**: a request with both uploaded files and a directory path checks each part's limits separately, so the combined size can reach twice the limit; `MAX_CHUNKS` still caps the whole request.
- **No authentication or rate limiting.**
- **Single process**: the semaphore limits ingests within one API process, not across replicas.
- `orchestrate.sh` needs `curl` on the host.

## 8. Next steps

- **Job queue + worker** (Redis + RQ/Celery): `/ingest/` returns a `job_id` and a status endpoint reports progress, so large ingests don't block requests and survive restarts. Once ingestion runs in workers, the API only serves light requests and can run several processes (gunicorn with uvicorn workers) or replicas behind a load balancer.
- **Object storage** (MinIO/S3) for the raw PDFs: needed by a worker, and lets every document be re-indexed after a model or chunking change.
- **Partial ingestion for directories**: 200 with `files` (stored) plus `skipped: [{file, error}]`; 400 only if nothing succeeded.
- **OCR** for scanned PDFs (e.g. Tesseract).
- **Hybrid search**: BM25 sparse vectors for exact keywords, plus a reranker on the top results.
- **Sentence-aware / token-based chunking**, and a measured chunk-size experiment.
- **Delete / re-index by document.**
- **Multilingual model** if needed (e.g. bge-m3, multilingual-e5).
- **Auth + rate limits.**
- **Observability**: metrics (Prometheus), request IDs, tracing.
