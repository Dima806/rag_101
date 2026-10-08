"""Chunkers: the cheap, ignored step that decides what can ever be retrieved.

Four strategies, each a few lines, built plainly so the central lesson of the project
lives in comparing them:

- :func:`fixed_chunker` -- the naive default: cut every ``size`` characters.
- :func:`sentence_chunker` -- never cut mid-sentence; pack whole sentences up to ``size``.
- :func:`recursive_chunker` -- split on structure (paragraphs, then sentences), then pack.
- :class:`SemanticChunker` -- group adjacent sentences while they stay on-topic by
  embedding similarity.

Every chunker returns non-empty, order-preserving chunks that together contain every
word of the input (no characters lost), which the tests assert.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Protocol

import numpy as np

# A sentence ends at ., ! or ? followed by whitespace. Good enough for a 101; the point
# is to show that respecting the boundary matters, not to ship a perfect segmenter.
_SENTENCE_RE = re.compile(r"[^.!?]*[.!?]+(?:\s+|$)|[^.!?]+$")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping terminators, dropping empty fragments."""
    return [m.group().strip() for m in _SENTENCE_RE.finditer(text) if m.group().strip()]


def _pack(units: list[str], size: int, joiner: str = " ") -> list[str]:
    """Greedily pack whole units into chunks no longer than ``size`` characters."""
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for unit in units:
        extra = len(unit) + (len(joiner) if current else 0)
        if current and length + extra > size:
            chunks.append(joiner.join(current))
            current, length = [], 0
            extra = len(unit)
        current.append(unit)
        length += extra
    if current:
        chunks.append(joiner.join(current))
    return chunks


def fixed_chunker(text: str, size: int = 400, overlap: int = 50) -> list[str]:
    """Cut every ``size`` characters with ``overlap`` carried forward.

    The naive default. It happily slices mid-word and mid-sentence, orphaning answers
    across chunk boundaries -- which is exactly the failure Notebook 02 exposes.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    text = text.strip()
    if not text:
        return []
    if not 0 <= overlap < size:
        raise ValueError("overlap must satisfy 0 <= overlap < size")
    step = size - overlap
    chunks = [text[start : start + size] for start in range(0, len(text), step)]
    # Drop a trailing chunk that is wholly contained in the previous one via overlap.
    if len(chunks) >= 2 and chunks[-1] in chunks[-2]:
        chunks.pop()
    return chunks


def sentence_chunker(text: str, size: int = 400, overlap: int = 0) -> list[str]:
    """Pack whole sentences up to ``size``; never cut mid-sentence.

    ``overlap`` is accepted for a uniform signature but unused: sentence packing keeps
    sentences intact, so there is nothing to overlap at the character level.
    """
    del overlap
    sentences = split_sentences(text)
    return _pack(sentences, size)


def recursive_chunker(text: str, size: int = 400, overlap: int = 0) -> list[str]:
    """Split on structure first (paragraphs), then sentences, then pack to ``size``.

    The sensible default for structured text: it respects the document's own
    boundaries before falling back to finer splits.
    """
    del overlap
    chunks: list[str] = []
    for paragraph in _PARAGRAPH_RE.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= size:
            chunks.append(paragraph)
        else:
            chunks.extend(_pack(split_sentences(paragraph), size))
    return chunks


class _Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class SemanticChunker:
    """Group adjacent sentences while they stay on-topic by embedding similarity.

    Starts a new chunk when the cosine similarity between consecutive sentence
    embeddings drops below ``threshold`` (a topic shift), and still caps chunks at
    ``size`` characters so a long on-topic run does not produce a blurry mega-chunk.
    """

    def __init__(self, encoder: _Encoder, threshold: float = 0.55, size: int = 400) -> None:
        self.encoder = encoder
        self.threshold = threshold
        self.size = size

    def __call__(self, text: str) -> list[str]:
        return self.chunk(text)

    def chunk(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        if len(sentences) <= 1:
            return sentences
        vectors = self.encoder.encode(sentences)
        chunks: list[str] = []
        current = [sentences[0]]
        for i in range(1, len(sentences)):
            similarity = float(np.dot(vectors[i - 1], vectors[i]))
            too_long = len(" ".join([*current, sentences[i]])) > self.size
            if similarity < self.threshold or too_long:
                chunks.append(" ".join(current))
                current = [sentences[i]]
            else:
                current.append(sentences[i])
        chunks.append(" ".join(current))
        return chunks


# Name -> callable for the simple (non-semantic) strategies, used by the ablation sweep.
Chunker = Callable[[str, int, int], list[str]]

CHUNKERS: dict[str, Chunker] = {
    "fixed": fixed_chunker,
    "sentence": sentence_chunker,
    "recursive": recursive_chunker,
}


def get_chunker(strategy: str) -> Chunker:
    """Return a non-semantic chunker by name (``fixed``/``sentence``/``recursive``)."""
    try:
        return CHUNKERS[strategy]
    except KeyError:
        valid = ", ".join([*sorted(CHUNKERS), "semantic"])
        raise ValueError(f"unknown chunker {strategy!r}; choose one of: {valid}") from None


def make_chunk_fn(
    strategy: str,
    *,
    size: int = 400,
    overlap: int = 0,
    semantic_threshold: float = 0.55,
    encoder: _Encoder | None = None,
) -> Callable[[str], list[str]]:
    """Return a single-argument ``text -> chunks`` function for the named strategy.

    The uniform one-argument signature is what the corpus chunker and the ablation sweeps
    consume. ``semantic`` requires an ``encoder``; the others ignore it.
    """
    if strategy == "semantic":
        if encoder is None:
            raise ValueError("semantic chunking requires an encoder")
        return SemanticChunker(encoder, threshold=semantic_threshold, size=size).chunk
    base = get_chunker(strategy)

    def chunk_fn(text: str) -> list[str]:
        return base(text, size, overlap)

    return chunk_fn


ALL_STRATEGIES: tuple[str, ...] = ("fixed", "sentence", "recursive", "semantic")
