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
from app.pdf import extract_pages

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
    logger.error("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error."}
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

def ingest_pdf(filename: str, data: bytes, embedder: Embedder, store: VectorStore) -> int:
    chunks, payloads = [], []
    for page_number, page_text in extract_pages(data):
        for chunk in chunk_text(page_text, settings.chunk_size, settings.chunk_overlap):
            chunks.append(chunk)
            payloads.append({"document": filename, "content": chunk, "page_number": page_number})
    if not chunks:
        raise HTTPException(status_code=400, detail=f"No extractable text in {filename}.")
    vectors = embedder.embed_passages(chunks)
    store.upsert(vectors, payloads)
    return len(chunks)

@app.post("/ingest/")
async def ingest(request: Request) -> dict:
    form = await request.form()
    items = form.getlist("input")
    if not items:
        raise HTTPException(status_code=400, detail="Field 'input' is required.")
    item = items[0]
    if not isinstance(item, UploadFile):
        raise HTTPException(status_code=400, detail="Directory input is not supported yet.")
    filename = item.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    data = await item.read()
    await run_in_threadpool(ingest_pdf, filename, data, request.app.state.embedder, request.app.state.store)
    return {"message": "Successfully ingested 1 PDF documents.", "files": [filename]}
