"""The retriever returns the planted gold passage on easy queries."""

from __future__ import annotations

import pytest

from src.config import Settings
from src.datasets.corpus import Corpus, chunk_corpus
from src.pipeline.chunk import make_chunk_fn
from src.pipeline.embed import CachedEmbedder
from src.pipeline.retrieve import build_retriever


def _chunked(corpus: Corpus, embedder: CachedEmbedder, settings: Settings):
    chunk_fn = make_chunk_fn(
        settings.chunk.strategy,
        size=settings.chunk.size,
        overlap=settings.chunk.overlap,
        encoder=embedder,
    )
    return chunk_corpus(corpus, chunk_fn)


@pytest.mark.parametrize("method", ["cosine", "bm25", "hybrid"])
def test_retriever_finds_gold_passage_for_answerable_questions(
    method: str,
    corpus: Corpus,
    embedder: CachedEmbedder,
    settings: Settings,
) -> None:
    chunks, chunk_doc_ids = _chunked(corpus, embedder, settings)
    retriever = build_retriever(method, embedder, settings.retrieve.hybrid_alpha)
    retriever.index(chunks)

    for question in corpus.questions_by_kind("answerable"):
        results = retriever.retrieve(question.text, settings.retrieve.top_k)
        retrieved_docs = {chunk_doc_ids[r.index] for r in results}
        gold = set(question.gold_doc_ids)
        assert gold & retrieved_docs, (
            f"{method}: expected one of {gold} in top-{settings.retrieve.top_k} "
            f"for {question.question_id!r}, got {retrieved_docs}"
        )


def test_scores_are_sorted_descending(
    corpus: Corpus,
    embedder: CachedEmbedder,
    settings: Settings,
) -> None:
    chunks, _ = _chunked(corpus, embedder, settings)
    retriever = build_retriever("cosine", embedder)
    retriever.index(chunks)
    results = retriever.retrieve("What currency does the archipelago use?", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_empty_index_returns_nothing(embedder: CachedEmbedder) -> None:
    retriever = build_retriever("cosine", embedder)
    retriever.index([])
    assert retriever.retrieve("anything", top_k=5) == []
