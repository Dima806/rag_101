"""Ablation sweeps -- hold everything fixed, vary one part, measure the swing.

These power the two showpiece experiments:

- :func:`chunking_sweep` -- hold embedder/retriever/generator fixed, vary only the chunker,
  and watch answer accuracy swing (Notebook 02). The cheap step nobody tunes.
- :func:`model_swap` -- hold chunking/retrieval fixed, swap the generator, and watch the
  answer barely move under a retrieval bottleneck (Notebook 04).
- :func:`top_k_sweep` -- vary how many passages are retrieved.

Each returns a tidy :class:`pandas.DataFrame` ready to plot.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pandas as pd

from src.config import Settings
from src.datasets.corpus import Corpus, chunk_corpus
from src.evaluation.answer_metrics import answer_correct
from src.pipeline.chunk import make_chunk_fn
from src.pipeline.embed import Embedder
from src.pipeline.generate import Generator
from src.pipeline.rag import RagPipeline
from src.pipeline.retrieve import build_retriever


def _build_pipeline(
    corpus: Corpus,
    chunk_fn: Callable[[str], list[str]],
    embedder: Embedder,
    generator: Generator,
    settings: Settings,
) -> tuple[RagPipeline, list[str]]:
    chunks, chunk_doc_ids = chunk_corpus(corpus, chunk_fn)
    retriever = build_retriever(settings.retrieve.method, embedder, settings.retrieve.hybrid_alpha)
    pipeline = RagPipeline(retriever=retriever, generator=generator, settings=settings)
    pipeline.index(chunks)
    return pipeline, chunk_doc_ids


def _answer_accuracy(
    pipeline: RagPipeline,
    corpus: Corpus,
    chunk_doc_ids: list[str],
    top_k: int,
) -> tuple[float, float]:
    """Return (answer accuracy, retrieval recall) over the answerable questions."""
    questions = corpus.questions_by_kind("answerable")
    if not questions:
        return 0.0, 0.0
    hits = 0
    recall_sum = 0.0
    for question in questions:
        result = pipeline.answer(question.text, top_k=top_k)
        hits += answer_correct(result.answer, question.answer)
        gold = set(question.gold_doc_ids)
        got = {chunk_doc_ids[r.index] for r in result.retrieved if r.index >= 0}
        recall_sum += len(gold & got) / len(gold) if gold else 0.0
    n = len(questions)
    return hits / n, recall_sum / n


def chunking_sweep(
    corpus: Corpus,
    embedder: Embedder,
    generator: Generator,
    settings: Settings,
    strategies: tuple[str, ...] = ("fixed", "sentence", "recursive", "semantic"),
) -> pd.DataFrame:
    """Vary only the chunker; report chunk count, retrieval recall, and answer accuracy."""
    rows: list[dict[str, object]] = []
    for strategy in strategies:
        chunk_fn = make_chunk_fn(
            strategy,
            size=settings.chunk.size,
            overlap=settings.chunk.overlap,
            semantic_threshold=settings.chunk.semantic_threshold,
            encoder=embedder,
        )
        chunks, chunk_doc_ids = chunk_corpus(corpus, chunk_fn)
        retriever = build_retriever(
            settings.retrieve.method, embedder, settings.retrieve.hybrid_alpha
        )
        pipeline = RagPipeline(retriever=retriever, generator=generator, settings=settings)
        pipeline.index(chunks)
        accuracy, recall = _answer_accuracy(
            pipeline, corpus, chunk_doc_ids, settings.retrieve.top_k
        )
        rows.append(
            {
                "chunker": strategy,
                "n_chunks": len(chunks),
                "retrieval_recall": recall,
                "answer_accuracy": accuracy,
            }
        )
    return pd.DataFrame(rows)


def model_swap(
    corpus: Corpus,
    embedder: Embedder,
    generators: Mapping[str, Generator],
    settings: Settings,
) -> pd.DataFrame:
    """Hold chunking/retrieval fixed; swap the generator and compare answer accuracy."""
    chunk_fn = make_chunk_fn(
        settings.chunk.strategy,
        size=settings.chunk.size,
        overlap=settings.chunk.overlap,
        semantic_threshold=settings.chunk.semantic_threshold,
        encoder=embedder,
    )
    rows: list[dict[str, object]] = []
    for name, generator in generators.items():
        pipeline, chunk_doc_ids = _build_pipeline(corpus, chunk_fn, embedder, generator, settings)
        accuracy, recall = _answer_accuracy(
            pipeline, corpus, chunk_doc_ids, settings.retrieve.top_k
        )
        rows.append({"model": name, "retrieval_recall": recall, "answer_accuracy": accuracy})
    return pd.DataFrame(rows)


def top_k_sweep(
    corpus: Corpus,
    embedder: Embedder,
    generator: Generator,
    settings: Settings,
    k_values: tuple[int, ...] = (1, 2, 3, 5, 10),
) -> pd.DataFrame:
    """Vary the number of retrieved passages; report recall and answer accuracy."""
    chunk_fn = make_chunk_fn(
        settings.chunk.strategy,
        size=settings.chunk.size,
        overlap=settings.chunk.overlap,
        semantic_threshold=settings.chunk.semantic_threshold,
        encoder=embedder,
    )
    pipeline, chunk_doc_ids = _build_pipeline(corpus, chunk_fn, embedder, generator, settings)
    rows: list[dict[str, object]] = []
    for k in k_values:
        accuracy, recall = _answer_accuracy(pipeline, corpus, chunk_doc_ids, k)
        rows.append({"top_k": k, "retrieval_recall": recall, "answer_accuracy": accuracy})
    return pd.DataFrame(rows)
