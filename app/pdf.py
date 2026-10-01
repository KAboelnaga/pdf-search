import io
from pypdf import PdfReader

import logging
logger = logging.getLogger("app.pdf")

def extract_pages(data: bytes) -> list[tuple[int, str]]:
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        if text:
            pages.append((page_number, text))
    return pages

class InvalidDocument(ValueError):
    """A user error about an input file -> mapped to HTTP 400 in main.py."""

def extract_document(filename: str, data: bytes) -> list[tuple[int, str]]:
    if data.startswith(b"%PDF-"):
        return extract_pages(data)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise InvalidDocument(f"{filename} is not a valid PDF.")
    logger.warning("%s has no PDF header, ingesting it as plain text.", filename)
    text = " ".join(text.split())
    return [(1, text)] if text else []
