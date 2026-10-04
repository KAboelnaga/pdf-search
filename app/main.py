import logging 
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import settings
from app.embedder import Embedder
from app.store import VectorStore

from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.chunker import chunk_text

from pathlib import Path
from app.pdf import InvalidDocument, extract_document

import hashlib
import uuid

import asyncio

import time

ingest_slots = asyncio.Semaphore(settings.max_concurrent_ingests)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")
logging.getLogger("httpx").setLevel(logging.WARNING)

def check_limits(files: list[tuple[str, int]]) -> None:
    mb = 1024 * 1024
    if len(files) > settings.max_files:
        raise PayloadTooLarge(f"Too many files: {len(files)} exceeds limit of {settings.max_files}.")
    for filename, size in files:
        if size > settings.max_file_mb * mb:
            raise PayloadTooLarge(f"{filename} is too large: {size / mb:.2f} MB exceeds limit of {settings.max_file_mb} MB.")
    total_size = sum(size for _, size in files)
    if total_size > settings.max_total_mb * mb:
        raise PayloadTooLarge(f"Total size of files exceeds limit of {settings.max_total_mb} MB.")

class PayloadTooLarge(Exception):
    """Request is valid but over a size limit -> 413"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("loading embedding model %s", settings.model_name)
    app.state.embedder = Embedder(settings.model_name, settings.model_cache_dir)
    app.state.store = VectorStore(settings.qdrant_url, settings.collection, settings.vector_size)
    logger.info("Ready: collection %s, model %s", settings.collection, settings.model_name)
    yield


app = FastAPI(
    title="PDF Ingestor & Semantic Search API", 
    version="1.0.0", 
    description="API for ingesting PDF files and performing semantic search on their content.",
    lifespan=lifespan
    )

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.info("%s %s 500 %.2fs", request.method, request.url.path, time.perf_counter() - start)
        raise
    logger.info("%s %s %d %.2fs", request.method, request.url.path,
                response.status_code, time.perf_counter() - start)
    return response

@app.exception_handler(PayloadTooLarge)
async def payload_too_large(request: Request, exc: PayloadTooLarge) -> JSONResponse:
    logger.warning("%s %s rejected: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=413,
        content={"error": str(exc)},
    )

@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    logger.warning("%s %s rejected: %s", request.method, request.url.path, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail},
    )

@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"error": "Invalid request body."}
    )

@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error."}
    )

@app.exception_handler(InvalidDocument)
async def invalid_document(request: Request, exc: InvalidDocument) -> JSONResponse:
    logger.warning("%s %s rejected: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=400,
        content={"error": str(exc)},
    )

class SearchRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    try:
        app.state.store.ping()
    except Exception:
        logger.warning("health check failed: Qdrant unreachable.")
        return JSONResponse(status_code=503, content={"status": "unhealthy", "qdrant": "unreachable"})
    return {"status": "healthy", "qdrant": "ok"}

@app.post("/search/")
def search(body: SearchRequest, request: Request) -> dict:
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if len(query) > settings.max_query_chars:
        raise HTTPException(status_code=400, detail=f"Query exceeds maximum length of {settings.max_query_chars} characters.")
    vector = request.app.state.embedder.embed_query(query)
    return {"results": request.app.state.store.search(vector, settings.top_k)}

def build_chunks(filename: str, data: bytes) -> tuple[list[str], list[str], list[dict]]:
    pages = extract_document(filename, data)
    content = "\n".join(f"{n}:{''.join(t.split())}" for n, t in pages) 
    text_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    ids, chunks, payloads = [], [], []
    for page_number, text in pages:
        for chunk in chunk_text(text, settings.chunk_size, settings.chunk_overlap):
            index = len(chunks)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{text_hash}:{index}"))
            ids.append(point_id)
            chunks.append(chunk)
            payloads.append({"document": filename, "page": page_number, "content": chunk})
    if not chunks:
        raise InvalidDocument(f"No extractable text in {filename} (scanned PDF? OCR is not supported).")
    return ids, chunks, payloads

def ingest_documents(docs: list[tuple[str, bytes]], embedder: Embedder, store: VectorStore) -> int:
    t0 = time.perf_counter()
    all_chunks, all_payloads, all_ids = [], [], []
    for filename, data in docs:
        ids, chunks, payloads = build_chunks(filename, data)
        all_chunks.extend(chunks)
        all_payloads.extend(payloads)
        all_ids.extend(ids)
    if len(all_chunks) > settings.max_chunks:
        raise PayloadTooLarge(f"Total number of chunks {len(all_chunks)} exceeds limit of {settings.max_chunks}.")
    parse_s = time.perf_counter() - t0
    embed_s = store_s = 0.0
    size = settings.embed_batch_size
    for start in range(0, len(all_chunks), size):
        end = start + size
        batch_chunks = all_chunks[start:end]
        batch_payloads = all_payloads[start:end]
        batch_ids = all_ids[start:end]
        t = time.perf_counter()
        vectors = embedder.embed_passages(batch_chunks)
        embed_s += time.perf_counter() - t
        t = time.perf_counter()
        store.upsert(batch_ids, vectors, batch_payloads)
        store_s += time.perf_counter() - t
    pages = len({(p["document"], p["page"]) for p in all_payloads})
    logger.info("ingest: %d files, %d pages, %d chunks | parse %.2fs, embed %.2fs, store %.2fs",
                len(docs), pages, len(all_chunks), parse_s, embed_s, store_s)
    return len(all_chunks)

def collect_directory(raw: str) -> list[tuple[str, bytes]]:
    root = Path(settings.data_dir).resolve()
    path = (root / raw.strip()).resolve()
    if not path.is_relative_to(root):
        raise InvalidDocument(f"Directory must be inside {root}.")
    if not path.is_dir():
        raise InvalidDocument(f"Directory not found: {raw}.")

    docs = []
    for p in sorted(path.iterdir()):
        if p.suffix.lower() != ".pdf":
            continue
        real = p.resolve()
        if not real.is_relative_to(root):
            logger.warning("Skipping %s: it points outside %s.", p.name, root)
            continue
        if not real.is_file():
            continue
        docs.append((p.name, real))
    if not docs:
        raise InvalidDocument(f"No PDF files found in {raw}.")
    check_limits([(name, real.stat().st_size) for name, real in docs])
    return [(name, real.read_bytes()) for name, real in docs]

@app.post("/ingest/", openapi_extra={"requestBody": {"required": True, "content": {"multipart/form-data": {"schema": {
    "type": "object",
    "required": ["input"],
    "properties": {"input": {
        "description": "File(s) or directory path",
        "oneOf": [
            {"type": "string", "description": "Directory path (e.g., /data/pdfs)"},
            {"type": "string", "format": "binary", "description": "Single PDF file"},
            {"type": "array", "items": {"type": "string", "format": "binary"}, "description": "Multiple PDF files"},
        ],
    }},
}}}}})
async def ingest(request: Request) -> dict:
    form = await request.form()
    items = form.getlist("input")
    if not items:
        raise HTTPException(status_code=400, detail="Field 'input' is required.")
    uploads = [item for item in items if isinstance(item, UploadFile)]
    check_limits([(u.filename or "", u.size or 0) for u in uploads])
    for u in uploads:
        if not (u.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    try:
        await asyncio.wait_for(ingest_slots.acquire(), timeout=settings.ingest_wait_seconds)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=503, detail="Server is busy ingesting other files. Try again shortly.")
    try:
        docs: list[tuple[str, bytes]] = []
        for item in items:
            if isinstance(item, UploadFile):
                filename = item.filename or ""
                docs.append((filename, await item.read()))
            else:
                docs.extend(await run_in_threadpool(collect_directory, item))
        await run_in_threadpool(ingest_documents, docs, request.app.state.embedder, request.app.state.store)
    finally:
        ingest_slots.release()                # ALWAYS give the slot back
    
    names = [name for name, _ in docs]
    return {"message": f"Successfully ingested {len(names)} PDF documents.", "files": names}
