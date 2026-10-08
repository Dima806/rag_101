"""Abstention: does the system say "not in the documents" when it should?

The honest crux of the project. A RAG system that cannot abstain will answer absence
questions with a confident, fabricated answer -- hallucination with a footnote. This
module decides when to abstain and detects when an answer already is an abstention.

Two complementary signals raise the abstention floor:

- a **retrieval-score threshold**: if nothing was retrieved above a minimum score, the
  context is probably irrelevant, so refuse before even generating; and
- a **groundedness check**: if the generated answer is not supported by the context,
  reject it and abstain instead.
"""

from __future__ import annotations

from src.grounding.groundedness import groundedness_score
from src.pipeline.generate import ABSTENTION_TEXT

# Phrases a model uses when it (correctly) declines to answer.
_ABSTENTION_MARKERS = (
    "do not contain",
    "does not contain",
    "not contain",
    "no information",
    "not mentioned",
    "not in the",
    "cannot answer",
    "can't answer",
    "i don't know",
    "i do not know",
    "unable to answer",
    "not available in the",
)


def is_abstention(answer: str) -> bool:
    """True when the answer is a refusal/abstention rather than a substantive claim."""
    low = answer.lower()
    return any(marker in low for marker in _ABSTENTION_MARKERS)


def should_abstain(
    answer: str,
    contexts: list[str],
    retrieval_scores: list[float],
    *,
    retrieval_score_threshold: float = 0.15,
    groundedness_threshold: float = 0.3,
) -> bool:
    """Decide whether the system should abstain instead of returning ``answer``.

    Abstains when retrieval is too weak to trust, or when the answer is not grounded in
    the retrieved context. An answer that is already an abstention is left to stand.
    """
    if is_abstention(answer):
        return True
    top_score = max(retrieval_scores) if retrieval_scores else 0.0
    if top_score < retrieval_score_threshold:
        return True
    return groundedness_score(answer, contexts) < groundedness_threshold


__all__ = ["ABSTENTION_TEXT", "is_abstention", "should_abstain"]
