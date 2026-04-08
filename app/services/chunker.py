from collections.abc import Iterable


def chunk_text(
    text: str, min_words: int = 200, max_words: int = 450, overlap_words: int = 60
) -> Iterable[str]:
    """
    Meaningful chunking for transcripts:
    - 200-450 words per chunk by default (configurable)
    - word overlap preserves continuity for retrieval
    """
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    words = cleaned.split(" ")
    if not words:
        return []

    chunks: list[str] = []
    cursor = 0
    n = len(words)
    step = max(1, max_words - overlap_words)

    while cursor < n:
        end = min(cursor + max_words, n)
        chunk_words = words[cursor:end]
        if len(chunk_words) >= min_words or end == n:
            chunks.append(" ".join(chunk_words).strip())
        if end == n:
            break
        cursor += step

    return chunks
