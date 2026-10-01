def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    step = size - overlap
    chunks = []
    for start in range(0, len(text), step):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        if start + size >= len(text):
            break
    return chunks
