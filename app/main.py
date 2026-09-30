import logging 

from fastapi import FastAPI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :%(message)s")
logger = logging.getLogger("app")

app = FastAPI(title="PDF Ingestor & Semantic Search API", version="1.0.0", description="API for ingesting PDF files and performing semantic search on their content.")

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}