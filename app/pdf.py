import io
from pypdf import PdfReader

import logging
logger = logging.getLogger("app.pdf")

class InvalidDocument(ValueError):
    """A user error about an input file -> mapped to HTTP 400 in main.py."""

def extract_pages(filename: str, data: bytes) -> list[tuple[int, str]]:
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise InvalidDocument(f"{filename} is corrupted or not a readable PDF.") from exc

    if reader.is_encrypted:
        try:
            unlocked = reader.decrypt("")
        except Exception:
            unlocked = 0
        if not unlocked:
            raise InvalidDocument(f"{filename} is password-protected.")
    pages, empty = [], 0
    try:
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = " ".join(text.split())
            if text:
                pages.append((page_number, text))
            else:
                empty += 1
    except Exception as exc:
        raise InvalidDocument(f"{filename} is corrupted or not a readable PDF.") from exc
    if empty:
        logger.info("%s: %d of %d pages had no text (scanned?).", filename, empty, len(reader.pages))
    return pages


def extract_document(filename: str, data: bytes) -> list[tuple[int, str]]:
    if not data.strip():
        raise InvalidDocument(f"{filename} is empty.")
    if data.startswith(b"%PDF-"):
        return extract_pages(filename, data)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise InvalidDocument(f"{filename} is corrupted or not a readable PDF.")
    logger.warning("%s has no PDF header, ingesting it as plain text.", filename)
    text = " ".join(text.split())
    return [(1, text)] if text else []
