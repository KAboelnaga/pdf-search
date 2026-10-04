"""API tests against the running stack (./orchestrate.sh --action start).

Run them on a fresh stack (terminate, then start) so only the sample PDFs are indexed.

Skipped automatically when the API isn't running, so the unit tests can run alone.
"""
import os
from pathlib import Path

import pytest
import requests

API = os.getenv("API_URL", "http://localhost:8000")
QDRANT = os.getenv("QDRANT_HOST_URL", "http://localhost:6333")
COLLECTION = os.getenv("COLLECTION_NAME", "pdf_chunks")
DATA = Path(__file__).resolve().parent.parent / "data"


def _api_is_up() -> bool:
    try:
        return requests.get(f"{API}/health", timeout=2).status_code == 200
    except requests.RequestException:
        return False


pytestmark = pytest.mark.skipif(not _api_is_up(), reason="API not running: start it with ./orchestrate.sh --action start")


def ingest(*paths: Path) -> requests.Response:
    files = [("input", (p.name, p.read_bytes(), "application/pdf")) for p in paths]
    return requests.post(f"{API}/ingest/", files=files, timeout=120)


def search(query) -> requests.Response:
    return requests.post(f"{API}/search/", json={"query": query}, timeout=30)


def point_count() -> int:
    r = requests.post(f"{QDRANT}/collections/{COLLECTION}/points/count", json={"exact": True}, timeout=10)
    return r.json()["result"]["count"]


@pytest.fixture(scope="module", autouse=True)
def samples_ingested():
    r = ingest(DATA / "sample.pdf", DATA / "pdfs" / "vector-databases.pdf")
    assert r.status_code == 200, r.text


# ---------- happy paths ----------

def test_health_reports_qdrant():
    assert requests.get(f"{API}/health").json() == {"status": "healthy", "qdrant": "ok"}


def test_ingest_response_shape():
    r = ingest(DATA / "sample.pdf")
    assert r.status_code == 200
    assert r.json() == {"message": "Successfully ingested 1 PDF document.", "files": ["sample.pdf"]}


def test_search_response_shape():
    body = search("vector embeddings").json()
    assert 1 <= len(body["results"]) <= 5
    for hit in body["results"]:
        assert set(hit) == {"document", "score", "content", "page"}
        assert isinstance(hit["score"], float) and hit["content"]


# ---------- known-answer retrieval ----------

@pytest.mark.parametrize("query, expected_document, expected_text", [
    ("How does semantic search work?", "sample.pdf", "by meaning rather than by exact words"),
    ("What is an HNSW index?", "vector-databases.pdf", "HNSW"),
    ("How do machine learning models learn from examples?", "sample.pdf", "learn"),
])
def test_the_right_passage_is_the_top_result(query, expected_document, expected_text):
    top = search(query).json()["results"][0]
    assert top["document"] == expected_document
    assert expected_text in top["content"]


# ---------- idempotency ----------

def test_ingesting_the_same_file_twice_adds_nothing():
    ingest(DATA / "sample.pdf")
    before = point_count()
    assert ingest(DATA / "sample.pdf").status_code == 200
    assert point_count() == before


# ---------- errors: right status, always {"error": ...} ----------

@pytest.mark.parametrize("method, path, kwargs, status", [
    ("get", "/ingest/", {}, 405),
    ("post", "/nope/", {}, 404),
    ("post", "/ingest/", {"files": {"file": ("a.pdf", b"%PDF-1.7")}}, 400),        # wrong field name
    ("post", "/ingest/", {"files": {"input": ("notes.txt", b"hello")}}, 400),       # not .pdf
    ("post", "/ingest/", {"files": {"input": ("empty.pdf", b"")}}, 400),
    ("post", "/ingest/", {"files": {"input": ("junk.pdf", b"%PDF-1.7 junk")}}, 400),
    ("post", "/ingest/", {"data": {"input": "../../etc"}}, 400),                    # path traversal
    ("post", "/search/", {"json": {"query": ""}}, 400),
    ("post", "/search/", {"json": {"query": "   "}}, 400),
    ("post", "/search/", {"json": {"query": 123}}, 400),
    ("post", "/search/", {"json": {"query": "a" * 1001}}, 400),
    ("post", "/search/", {"data": "{bad json", "headers": {"Content-Type": "application/json"}}, 400),
])
def test_errors_use_the_right_status_and_format(method, path, kwargs, status):
    r = getattr(requests, method)(f"{API}{path}", timeout=30, **kwargs)
    assert r.status_code == status
    assert set(r.json()) == {"error"} and r.json()["error"]
