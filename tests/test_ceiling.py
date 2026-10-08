"""Retrieval is the ceiling: oracle-passage accuracy >= retrieved-passage accuracy.

The central asserted fact of the project. Feeding the generator the gold passage by hand
answers correctly every time; feeding it what the retriever actually returned can only do
as well as retrieval did, and no better.
"""

from __future__ import annotations

import pytest

from src.config import Settings
from src.datasets.corpus import Corpus, chunk_corpus
from src.evaluation.ceiling import evaluate_ceiling
from src.pipeline.chunk import make_chunk_fn
from src.pipeline.embed import CachedEmbedder
from src.pipeline.generate import CallableGenerator
from src.pipeline.rag import RagPipeline
from src.pipeline.retrieve import build_retriever


def _pipeline(
    corpus: Corpus,
    embedder: CachedEmbedder,
    generator: CallableGenerator,
    settings: Settings,
) -> tuple[RagPipeline, list[str]]:
    chunk_fn = make_chunk_fn(
        settings.chunk.strategy,
        size=settings.chunk.size,
        overlap=settings.chunk.overlap,
        encoder=embedder,
    )
    chunks, chunk_doc_ids = chunk_corpus(corpus, chunk_fn)
    retriever = build_retriever(settings.retrieve.method, embedder)
    pipeline = RagPipeline(retriever=retriever, generator=generator, settings=settings)
    pipeline.index(chunks)
    return pipeline, chunk_doc_ids


def test_oracle_accuracy_is_at_least_retrieved_accuracy(
    corpus: Corpus,
    embedder: CachedEmbedder,
    scripted_generator: CallableGenerator,
    settings: Settings,
) -> None:
    pipeline, chunk_doc_ids = _pipeline(corpus, embedder, scripted_generator, settings)
    result = evaluate_ceiling(pipeline, corpus, chunk_doc_ids)

    assert result.n_questions == len(corpus.questions_by_kind("answerable"))
    # The ceiling: the whole system can never beat retrieval.
    assert result.oracle_accuracy >= result.retrieved_accuracy


def test_oracle_passage_answers_correctly(
    corpus: Corpus,
    embedder: CachedEmbedder,
    scripted_generator: CallableGenerator,
    settings: Settings,
) -> None:
    pipeline, chunk_doc_ids = _pipeline(corpus, embedder, scripted_generator, settings)
    result = evaluate_ceiling(pipeline, corpus, chunk_doc_ids)
    # Given the gold passage by hand, the generator is right every time.
    assert result.oracle_accuracy == 1.0


def test_retrieved_accuracy_tracks_retrieval_recall(
    corpus: Corpus,
    embedder: CachedEmbedder,
    scripted_generator: CallableGenerator,
    settings: Settings,
) -> None:
    pipeline, chunk_doc_ids = _pipeline(corpus, embedder, scripted_generator, settings)
    result = evaluate_ceiling(pipeline, corpus, chunk_doc_ids)
    # With a generator that answers only from the gold passage, answer accuracy from the
    # retrieved passage equals the retriever's recall -- the quantitative form of the cap.
    assert result.retrieved_accuracy == pytest.approx(result.retrieval_recall)
