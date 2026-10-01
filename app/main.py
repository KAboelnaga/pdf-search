import logging 
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import settings
from app.embedder import Embedder
from app.store import VectorStore

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
