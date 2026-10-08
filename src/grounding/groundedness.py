"""Groundedness: is the answer actually supported by the retrieved context?

Groundedness is distinct from correctness. A correct answer can be ungrounded (the model
knew it from training, the passage did not say it) and a grounded answer can be wrong (it
faithfully repeats a passage that does not address the question). This module measures
support only, with a deliberately simple lexical proxy so a 101 reader can see the whole
mechanism: the fraction of the answer's content words that also appear in the context.

The real RAG-triad version (reused from ``llm_judge_benchmark``) would use an LLM judge;
the lexical proxy is enough to drive the groundedness check and the tests, and it makes
the point that an unsupported answer is detectable.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]+")

# A small stopword list so "the answer is" does not count as grounding.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "with",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "as",
        "it",
        "its",
        "it's",
        "into",
        "over",
        "under",
        "not",
        "no",
        "yes",
        "do",
        "does",
        "did",
        "i",
        "you",
        "he",
        "she",
        "they",
        "we",
        "me",
        "him",
        "her",
        "them",
        "us",
        "my",
        "your",
        "his",
        "their",
        "our",
        "answer",
        "based",
        "according",
        "context",
        "provided",
        "information",
        "document",
        "documents",
        "passage",
        "passages",
        "question",
    ]
)


def _content_tokens(text: str) -> list[str]:
    return [t for t in _WORD_RE.findall(text.lower()) if t not in _STOPWORDS]


def groundedness_score(answer: str, contexts: list[str]) -> float:
    """Fraction of the answer's content words that appear in the context (0.0-1.0).

    Returns 0.0 when the answer has no content words (e.g. pure boilerplate), which is
    the right signal: nothing substantive to ground.
    """
    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return 0.0
    context_tokens = set(_content_tokens(" ".join(contexts)))
    supported = sum(1 for t in answer_tokens if t in context_tokens)
    return supported / len(answer_tokens)


def is_grounded(answer: str, contexts: list[str], threshold: float = 0.3) -> bool:
    """True when the answer's groundedness score meets ``threshold``."""
    return groundedness_score(answer, contexts) >= threshold
