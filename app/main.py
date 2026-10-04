import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.config import settings
from app.embedder import Embedder
from app.ingestion import PayloadTooLarge, check_limits, collect_directory, ingest_documents
from app.pdf import InvalidDocument
from app.store import VectorStore

ingest_slots = asyncio.Semaphore(settings.max_concurrent_ingests)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")
logging.getLogger("httpx").setLevel(logging.WARNING)

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
    messages = {"/ingest/": "Failed to process uploaded file.", "/search/": "Search processing failed."}  # their swagger's wording
    return JSONResponse(
        status_code=500,
        content={"error": messages.get(request.url.path, "Internal server error.")}
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
    noun = "document" if len(names) == 1 else "documents"
    return {"message": f"Successfully ingested {len(names)} PDF {noun}.", "files": names}
