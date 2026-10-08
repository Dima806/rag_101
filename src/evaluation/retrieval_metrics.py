"""Retrieval metrics -- measured on retrieval ALONE, before any generation.

Measuring retrieval separately is the single most clarifying thing a RAG project can do:
it exposes the ceiling. These functions score a ranked list of retrieved doc ids against
the set of gold doc ids for a question.

- :func:`recall_at_k`  -- did any gold passage make it into the top-k?
- :func:`mrr`          -- reciprocal rank of the first gold passage (how high it ranked).
- :func:`hit_rate`     -- fraction of questions with at least one gold passage in top-k.
"""

from __future__ import annotations

from collections.abc import Sequence


def recall_at_k(retrieved_doc_ids: Sequence[str], gold_doc_ids: Sequence[str], k: int) -> float:
    """Fraction of a question's gold passages found in the top-k retrieved doc ids.

    For single-gold questions this is 1.0 if the gold passage is in the top-k else 0.0;
    for multi-gold questions it is the fraction of gold passages retrieved.
    """
    gold = set(gold_doc_ids)
    if not gold:
        return 0.0
    top = set(retrieved_doc_ids[:k])
    return len(gold & top) / len(gold)


def hit_at_k(retrieved_doc_ids: Sequence[str], gold_doc_ids: Sequence[str], k: int) -> float:
    """1.0 if at least one gold passage is in the top-k, else 0.0."""
    gold = set(gold_doc_ids)
    if not gold:
        return 0.0
    return 1.0 if gold & set(retrieved_doc_ids[:k]) else 0.0


def reciprocal_rank(retrieved_doc_ids: Sequence[str], gold_doc_ids: Sequence[str]) -> float:
    """Reciprocal of the rank (1-based) of the first gold passage; 0.0 if none appear."""
    gold = set(gold_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in gold:
            return 1.0 / rank
    return 0.0


def mrr(
    retrieved_per_question: Sequence[Sequence[str]],
    gold_per_question: Sequence[Sequence[str]],
) -> float:
    """Mean reciprocal rank across questions."""
    if not retrieved_per_question:
        return 0.0
    total = sum(
        reciprocal_rank(retrieved, gold)
        for retrieved, gold in zip(retrieved_per_question, gold_per_question, strict=True)
    )
    return total / len(retrieved_per_question)


def hit_rate(
    retrieved_per_question: Sequence[Sequence[str]],
    gold_per_question: Sequence[Sequence[str]],
    k: int,
) -> float:
    """Mean hit@k across questions (fraction with a gold passage in the top-k)."""
    if not retrieved_per_question:
        return 0.0
    total = sum(
        hit_at_k(retrieved, gold, k)
        for retrieved, gold in zip(retrieved_per_question, gold_per_question, strict=True)
    )
    return total / len(retrieved_per_question)


def mean_recall_at_k(
    retrieved_per_question: Sequence[Sequence[str]],
    gold_per_question: Sequence[Sequence[str]],
    k: int,
) -> float:
    """Mean recall@k across questions."""
    if not retrieved_per_question:
        return 0.0
    total = sum(
        recall_at_k(retrieved, gold, k)
        for retrieved, gold in zip(retrieved_per_question, gold_per_question, strict=True)
    )
    return total / len(retrieved_per_question)
