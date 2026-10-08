"""Retrievers: the pluggable, independently-measurable component that caps the answer.

Three retrievers behind one :class:`Retriever` protocol, each indexed over the same
chunk list so they can be swapped and compared:

- :class:`CosineRetriever` -- dense cosine top-k over a numpy matrix. At this scale numpy
  is a perfectly good vector store (the honest reminder from ``retrieval_arena``).
- :class:`BM25Retriever` -- classic lexical BM25 (``rank-bm25``), no model required.
- :class:`HybridRetriever` -- min-max-normalised blend of a dense and a lexical retriever.

``retrieve`` returns ranked :class:`Retrieved` records so that retrieval can be measured
on its own (recall@k, MRR) before any generation happens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from src.pipeline.embed import Embedder

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


@dataclass(frozen=True)
class Retrieved:
    """One retrieved chunk: its position in the index, its score, and its text."""

    index: int
    score: float
    text: str


class Retriever(Protocol):
    """Index a list of chunks, then return the top-k for a query."""

    def index(self, chunks: list[str]) -> None: ...

    def retrieve(self, query: str, top_k: int) -> list[Retrieved]: ...


def _top_k(scores: np.ndarray, chunks: list[str], top_k: int) -> list[Retrieved]:
    """Turn a score vector into the top-k ranked :class:`Retrieved` records."""
    k = min(top_k, len(chunks))
    if k <= 0:
        return []
    # argpartition for the top-k, then sort just those by score descending.
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    return [Retrieved(index=int(i), score=float(scores[i]), text=chunks[i]) for i in top]


class CosineRetriever:
    """Dense retrieval: cosine similarity (dot product of normalised rows) top-k."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._chunks: list[str] = []
        self._matrix: np.ndarray | None = None

    def index(self, chunks: list[str]) -> None:
        self._chunks = list(chunks)
        self._matrix = self.embedder.encode(self._chunks) if chunks else None

    def retrieve(self, query: str, top_k: int) -> list[Retrieved]:
        if self._matrix is None:
            return []
        query_vector = self.embedder.encode([query])[0]
        scores = self._matrix @ query_vector
        return _top_k(scores, self._chunks, top_k)


class BM25Retriever:
    """Lexical retrieval with Okapi BM25. ``rank-bm25`` is imported lazily."""

    def __init__(self) -> None:
        self._chunks: list[str] = []
        # rank_bm25 ships no type stubs, so the model object is untyped (Any).
        self._bm25: Any = None

    def index(self, chunks: list[str]) -> None:
        from rank_bm25 import BM25Okapi

        self._chunks = list(chunks)
        self._bm25 = BM25Okapi([_tokenize(c) for c in chunks]) if chunks else None

    def retrieve(self, query: str, top_k: int) -> list[Retrieved]:
        if self._bm25 is None:
            return []
        scores = np.asarray(self._bm25.get_scores(_tokenize(query)), dtype=np.float64)
        return _top_k(scores, self._chunks, top_k)


def _minmax(scores: np.ndarray) -> np.ndarray:
    """Scale scores to [0, 1]; a flat vector maps to all-zeros."""
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-12:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


class HybridRetriever:
    """Blend a dense and a lexical retriever: ``alpha`` on dense, ``1 - alpha`` on BM25."""

    def __init__(self, embedder: Embedder, alpha: float = 0.5) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha
        self._dense = CosineRetriever(embedder)
        self._lexical = BM25Retriever()
        self._chunks: list[str] = []

    def index(self, chunks: list[str]) -> None:
        self._chunks = list(chunks)
        self._dense.index(chunks)
        self._lexical.index(chunks)

    def retrieve(self, query: str, top_k: int) -> list[Retrieved]:
        if not self._chunks:
            return []
        n = len(self._chunks)
        dense = self._scatter(self._dense.retrieve(query, n), n)
        lexical = self._scatter(self._lexical.retrieve(query, n), n)
        scores = self.alpha * _minmax(dense) + (1.0 - self.alpha) * _minmax(lexical)
        return _top_k(scores, self._chunks, top_k)

    @staticmethod
    def _scatter(results: list[Retrieved], n: int) -> np.ndarray:
        scores = np.zeros(n, dtype=np.float64)
        for r in results:
            scores[r.index] = r.score
        return scores


def build_retriever(method: str, embedder: Embedder, hybrid_alpha: float = 0.5) -> Retriever:
    """Construct a retriever by name (``cosine`` / ``bm25`` / ``hybrid``)."""
    if method == "cosine":
        return CosineRetriever(embedder)
    if method == "bm25":
        return BM25Retriever()
    if method == "hybrid":
        return HybridRetriever(embedder, alpha=hybrid_alpha)
    raise ValueError(f"unknown retriever method {method!r}")
