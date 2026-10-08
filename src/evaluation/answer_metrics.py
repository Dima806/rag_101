"""Answer metrics -- the RAG-triad lens, scored against known gold answers.

Three things, kept distinct on purpose (the lesson of Notebook 04):

- :func:`answer_correct` / :func:`correctness` -- is the answer right? Scored against the
  known gold answer with normalised containment plus a token-F1 fallback, so small wording
  differences from a tiny local model still count.
- :func:`groundedness` -- is the answer supported by the context? (re-exported from
  :mod:`src.grounding.groundedness`; distinct from correctness).
- :func:`abstention_rate` -- how often the system (correctly or not) declines to answer.

The qualitative gaps the project demonstrates are large enough to survive the fuzziness of
answer scoring.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from src.grounding.abstention import is_abstention
from src.grounding.groundedness import groundedness_score

_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def token_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between a prediction and a reference answer."""
    pred, ref = _normalize(prediction), _normalize(reference)
    if not pred or not ref:
        return 0.0
    pred_set, ref_set = set(pred), set(ref)
    overlap = len(pred_set & ref_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_set)
    recall = overlap / len(ref_set)
    return 2 * precision * recall / (precision + recall)


def answer_correct(prediction: str, gold_answer: str, f1_threshold: float = 0.6) -> bool:
    """True if the prediction matches the gold answer by containment or token-F1.

    Containment catches the common case where the tiny model wraps the right span in a
    sentence ("The capital is Thornmouth." vs gold "Thornmouth"); F1 catches reorderings.
    """
    pred_tokens = _normalize(prediction)
    gold_tokens = _normalize(gold_answer)
    if not gold_tokens:
        return False
    # Containment: all gold tokens appear (in order) within the prediction.
    if _is_subsequence(gold_tokens, pred_tokens):
        return True
    return token_f1(prediction, gold_answer) >= f1_threshold


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    it = iter(haystack)
    return all(token in it for token in needle)


def correctness(
    predictions: Sequence[str],
    gold_answers: Sequence[str],
    f1_threshold: float = 0.6,
) -> float:
    """Fraction of predictions judged correct against their gold answers."""
    if not predictions:
        return 0.0
    hits = sum(
        answer_correct(pred, gold, f1_threshold)
        for pred, gold in zip(predictions, gold_answers, strict=True)
    )
    return hits / len(predictions)


def groundedness(answer: str, contexts: list[str]) -> float:
    """Re-export: fraction of the answer's content words supported by the context."""
    return groundedness_score(answer, contexts)


def abstention_rate(answers: Sequence[str]) -> float:
    """Fraction of answers that are abstentions ("not in the documents")."""
    if not answers:
        return 0.0
    return sum(is_abstention(a) for a in answers) / len(answers)


def evidence_present(gold_answer: str, contexts: Sequence[str]) -> bool:
    """True if the gold answer text actually appears in the retrieved context.

    This is the real cap on a faithful generator and the honest sense of "retrieval
    recall" for the ceiling: a passage you never retrieved (or chopped so the answer span
    was orphaned across chunks) is an answer the model cannot ground. Distinct from
    document-level recall@k, which can stay high while the answer span is lost inside a
    retrieved-but-fragmented document.
    """
    return gold_answer.lower() in " ".join(contexts).lower()
