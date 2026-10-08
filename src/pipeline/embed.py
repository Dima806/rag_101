"""Embedders: turn chunks and queries into vectors the retriever can compare.

Two backends behind one :class:`Embedder` protocol:

- :class:`SentenceTransformerEmbedder` -- a small sentence-transformer (MiniLM / bge-small),
  tens of megabytes, CPU-fast. The real embedder; imported lazily so merely importing this
  module (e.g. during test collection) never pulls in torch.
- :class:`HashingEmbedder` -- a deterministic, dependency-free bag-of-words hashing
  embedder. No model download, so tests and offline notebook runs work without network.

Both return L2-normalised rows, so a dot product is a cosine similarity. A
:class:`CachedEmbedder` memoises by text to keep notebooks quick.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

import numpy as np

from src.config import EmbedSettings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise each row; leave all-zero rows as zeros (no divide-by-zero)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns a list of strings into a 2-D array of row vectors."""

    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic hashing embedder: no model, no network, stable across runs.

    Each token is hashed (via md5, so the result does not depend on Python's per-process
    hash seed) into one of ``dim`` buckets and counted; the row is then L2-normalised.
    Crude, but enough to show how embedding interacts with chunk size and to drive the
    retrieval tests offline.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _bucket(self, token: str) -> int:
        # md5 is used only as a stable, seed-independent hash (not for security).
        digest = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self.dim

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            for token in _tokenize(text):
                matrix[row, self._bucket(token)] += 1.0
        return _normalize(matrix)


class SentenceTransformerEmbedder:
    """Wrap a small sentence-transformer model. Imported lazily to keep startup cheap."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, cache_folder=cache_dir)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float64)


class CachedEmbedder:
    """Memoise an embedder by exact text, so repeated chunks/queries are encoded once."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            fresh = self.embedder.encode(missing)
            for text, vector in zip(missing, fresh, strict=True):
                self._cache[text] = vector
        return np.vstack([self._cache[t] for t in texts])


def build_embedder(settings: EmbedSettings) -> CachedEmbedder:
    """Construct the embedder named in settings, wrapped in a cache."""
    if settings.backend == "hashing":
        base: Embedder = HashingEmbedder(dim=settings.dim)
    elif settings.backend == "sentence_transformer":
        base = SentenceTransformerEmbedder(
            model_name=settings.model_name,
            cache_dir=settings.cache_dir,
        )
    else:
        raise ValueError(f"unknown embed backend {settings.backend!r}")
    return CachedEmbedder(base)
