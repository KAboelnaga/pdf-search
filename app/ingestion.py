"""The ingestion pipeline: request limits, directory input, chunking with deterministic IDs, embedding and storage.

No FastAPI here: main.py turns these exceptions into HTTP responses (PayloadTooLarge -> 413, InvalidDocument -> 400).
"""
import hashlib
import logging
import time
import uuid
from pathlib import Path

from app.chunker import chunk_text
from app.config import settings
from app.embedder import Embedder
from app.pdf import InvalidDocument, extract_document
from app.store import VectorStore

logger = logging.getLogger("app.ingestion")


class PayloadTooLarge(Exception):
    """Request is valid but over a size limit -> 413"""


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


def collect_directory(raw: str) -> list[tuple[str, bytes]]:
    if not raw.strip():
        raise InvalidDocument("Directory path is empty.")
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
