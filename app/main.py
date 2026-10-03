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

from app.chunker import chunk_text

from pathlib import Path
from app.pdf import InvalidDocument, extract_document

import hashlib
import uuid

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

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

@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
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
    return JSONResponse(
        status_code=400,
        content={"error": str(exc)},
    )

class SearchRequest(BaseModel):
    query: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}

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
    file_hash = hashlib.sha256(data).hexdigest()
    ids, chunks, payloads = [], [], []
    for page_number, text in extract_document(filename, data):
        for chunk in chunk_text(text, settings.chunk_size, settings.chunk_overlap):
            index = len(chunks)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{file_hash}:{index}"))
            ids.append(point_id)
            chunks.append(chunk)
            payloads.append({"document": filename, "page": page_number, "content": chunk})
    if not chunks:
        raise InvalidDocument(f"No extractable text in {filename} (scanned PDF? OCR is not supported).")
    return ids, chunks, payloads

def ingest_documents(docs: list[tuple[str, bytes]], embedder: Embedder, store: VectorStore) -> int:
    all_chunks, all_payloads, all_ids = [], [], []
    for filename, data in docs:
        ids, chunks, payloads = build_chunks(filename, data)
        all_chunks.extend(chunks)
        all_payloads.extend(payloads)
        all_ids.extend(ids)
    size = settings.embed_batch_size
    for start in range(0, len(all_chunks), size):
        end = start + size
        batch_chunks = all_chunks[start:end]
        batch_payloads = all_payloads[start:end]
        batch_ids = all_ids[start:end]
        vectors = embedder.embed_passages(batch_chunks)
        store.upsert(batch_ids, vectors, batch_payloads)
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
        docs.append((p.name, real.read_bytes()))
    if not docs:
        raise InvalidDocument(f"No PDF files found in {raw}.")
    return docs

@app.post("/ingest/")
async def ingest(request: Request) -> dict:
    form = await request.form()
    items = form.getlist("input")
    if not items:
        raise HTTPException(status_code=400, detail="Field 'input' is required.")
    docs: list[tuple[str, bytes]] = []
    for item in items:
        if isinstance(item, UploadFile):
            filename = item.filename or ""
            if not filename.lower().endswith(".pdf"):
                raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
            docs.append((filename, await item.read()))
        else:
            docs.extend(await run_in_threadpool(collect_directory, item))
    await run_in_threadpool(ingest_documents, docs, request.app.state.embedder, request.app.state.store)
    names = [name for name, _ in docs]
    return {"message": f"Successfully ingested {len(names)} PDF documents.", "files": names}
