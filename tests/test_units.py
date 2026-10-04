"""Unit tests: call the pipeline functions directly. No server, no Docker, no Qdrant."""
import io
from pathlib import Path

import pytest
from pypdf import PdfWriter

from app import ingestion
from app.chunker import chunk_text
from app.config import Settings
from app.ingestion import PayloadTooLarge, build_chunks, check_limits, collect_directory
from app.pdf import InvalidDocument, extract_document

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample.pdf"
MB = 1024 * 1024


# ---------- chunker ----------

def test_chunks_overlap_by_the_configured_amount():
    text = "".join(chr(65 + i % 26) for i in range(2000))
    chunks = chunk_text(text, size=800, overlap=120)
    assert [len(c) for c in chunks] == [800, 800, 640]          # windows start at 0, 680, 1360
    assert chunks[0][-120:] == chunks[1][:120]                  # neighbours share 120 characters


def test_no_duplicate_tail_sliver():
    # One window already covers 700 chars, so the break must stop before a 20-char sliver at 680.
    assert chunk_text("x" * 700, size=800, overlap=120) == ["x" * 700]


def test_empty_text_gives_no_chunks():
    assert chunk_text("", size=800, overlap=120) == []


# ---------- settings (fail fast) ----------

@pytest.mark.parametrize("kwargs", [
    {"chunk_size": 0},
    {"chunk_overlap": -1},
    {"chunk_overlap": 800, "chunk_size": 800},
    {"max_chunks": 0},
    {"max_concurrent_ingests": 0},
])
def test_invalid_settings_refuse_to_start(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs)


# ---------- limits (413) ----------

def test_limits_accept_a_normal_request():
    check_limits([("a.pdf", 1 * MB), ("b.pdf", 2 * MB)])        # no exception


def test_too_many_files():
    with pytest.raises(PayloadTooLarge, match="Too many files"):
        check_limits([(f"{i}.pdf", 1) for i in range(21)])


def test_file_too_large():
    with pytest.raises(PayloadTooLarge, match="too large"):
        check_limits([("big.pdf", 51 * MB)])


def test_total_too_large():
    with pytest.raises(PayloadTooLarge, match="Total size"):
        check_limits([(f"{i}.pdf", 40 * MB) for i in range(3)])  # each under 50, total 120 > 100


# ---------- directory input (path safety) ----------

@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """A fake /data folder with one real PDF, one non-PDF, and a symlink pointing outside."""
    root = tmp_path / "data"
    docs = root / "docs"
    docs.mkdir(parents=True)
    (docs / "real.pdf").write_bytes(SAMPLE.read_bytes())
    (docs / "notes.txt").write_text("not a pdf")
    secret = tmp_path / "secret.pdf"
    secret.write_text("password=hunter2")
    (docs / "leak.pdf").symlink_to(secret)
    monkeypatch.setattr(ingestion, "settings", Settings(data_dir=str(root)))
    return root


def test_directory_reads_only_pdfs_inside_data(data_root):
    names = [name for name, _ in collect_directory("docs")]
    assert names == ["real.pdf"]          # notes.txt skipped (not .pdf), leak.pdf skipped (points outside)


def test_absolute_path_inside_data_works(data_root):
    assert [n for n, _ in collect_directory(str(data_root / "docs"))] == ["real.pdf"]


@pytest.mark.parametrize("raw", ["../../etc", "/etc", "../secret.pdf"])
def test_paths_outside_data_are_rejected(data_root, raw):
    with pytest.raises(InvalidDocument, match="must be inside"):
        collect_directory(raw)


def test_empty_directory_path_is_rejected(data_root):
    with pytest.raises(InvalidDocument, match="empty"):
        collect_directory("   ")


def test_missing_directory(data_root):
    with pytest.raises(InvalidDocument, match="not found"):
        collect_directory("nope")


# ---------- parsing ----------

def test_real_pdf_gives_numbered_pages():
    pages = extract_document("sample.pdf", SAMPLE.read_bytes())
    assert pages[0][0] == 1
    assert "Semantic search finds information by meaning" in pages[0][1]


def test_text_named_pdf_is_accepted():
    # Their suite.py uploads plain text named sample.pdf and expects 200.
    assert extract_document("sample.pdf", b"AI learns from data.") == [(1, "AI learns from data.")]


@pytest.mark.parametrize("data, message", [
    (b"", "empty"),
    (b"   \n", "empty"),
    (b"%PDF-1.7\nthis is not really a pdf", "corrupted"),
    (bytes(range(256)) * 4, "corrupted"),                       # binary junk, not UTF-8
])
def test_bad_files_are_rejected(data, message):
    with pytest.raises(InvalidDocument, match=message):
        extract_document("x.pdf", data)


def _encrypted(user_password: str) -> bytes:
    writer = PdfWriter(clone_from=io.BytesIO(SAMPLE.read_bytes()))
    writer.encrypt(user_password=user_password, owner_password="owner", algorithm="AES-256")
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_password_protected_pdf_is_rejected():
    with pytest.raises(InvalidDocument, match="password-protected"):
        extract_document("locked.pdf", _encrypted("secret"))


def test_empty_password_pdf_is_readable():
    assert extract_document("open.pdf", _encrypted(""))       # opens like in any viewer


# ---------- chunks + deterministic IDs ----------

def test_same_content_gets_the_same_ids_whatever_the_name():
    data = SAMPLE.read_bytes()
    ids_a, chunks_a, payloads_a = build_chunks("a.pdf", data)
    ids_b, _, payloads_b = build_chunks("b.pdf", data)
    assert ids_a == ids_b                                       # re-ingest overwrites, never duplicates
    assert len(set(ids_a)) == len(ids_a)                        # every chunk has its own ID
    assert payloads_a[0]["document"] == "a.pdf" and payloads_b[0]["document"] == "b.pdf"
    assert len(ids_a) == len(chunks_a) == len(payloads_a)


def test_spacing_differences_do_not_change_the_ids():
    a, _, _ = build_chunks("a.pdf", b"Queue + worker uploads return fast.")
    b, _, _ = build_chunks("b.pdf", b"Queue+ worker   uploads return  fast.")
    assert a == b


def test_different_content_gets_different_ids():
    a, _, _ = build_chunks("a.pdf", b"first document")
    b, _, _ = build_chunks("a.pdf", b"second document")
    assert a != b
