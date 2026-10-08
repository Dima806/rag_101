"""The ceiling: oracle-passage accuracy vs retrieved-passage accuracy.

The clean experiment that proves retrieval caps the answer. For each answerable question:

- feed the generator the *oracle* gold passage by hand -> it answers correctly almost
  always (this is what the generator can do when retrieval is perfect); and
- feed it the passage the *retriever* actually returned -> accuracy falls to the
  retriever's recall.

The second can never exceed the first, which :mod:`tests.test_ceiling` asserts.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.datasets.corpus import Corpus, Question
from src.evaluation.answer_metrics import answer_correct
from src.pipeline.rag import RagPipeline


@dataclass(frozen=True)
class CeilingResult:
    """Oracle vs retrieved accuracy for a set of questions, plus retrieval recall."""

    oracle_accuracy: float
    retrieved_accuracy: float
    retrieval_recall: float
    n_questions: int


def evaluate_ceiling(
    pipeline: RagPipeline,
    corpus: Corpus,
    chunk_doc_ids: list[str],
    questions: list[Question] | None = None,
    top_k: int | None = None,
) -> CeilingResult:
    """Compare oracle-passage and retrieved-passage answer accuracy on answerable questions.

    ``chunk_doc_ids`` maps each indexed chunk to its source document, so a retrieved chunk
    can be scored against a question's gold documents for the recall figure.
    """
    pool = questions if questions is not None else corpus.questions_by_kind("answerable")
    if not pool:
        return CeilingResult(0.0, 0.0, 0.0, 0)

    oracle_hits = 0
    retrieved_hits = 0
    recall_sum = 0.0
    for question in pool:
        gold_passages = corpus.gold_passages(question)
        oracle = pipeline.answer_from_contexts(question.text, gold_passages)
        oracle_hits += answer_correct(oracle.answer, question.answer)

        result = pipeline.answer(question.text, top_k=top_k)
        retrieved_hits += answer_correct(result.answer, question.answer)

        gold = set(question.gold_doc_ids)
        retrieved_docs = {chunk_doc_ids[r.index] for r in result.retrieved if r.index >= 0}
        recall_sum += len(gold & retrieved_docs) / len(gold) if gold else 0.0

    n = len(pool)
    return CeilingResult(
        oracle_accuracy=oracle_hits / n,
        retrieved_accuracy=retrieved_hits / n,
        retrieval_recall=recall_sum / n,
        n_questions=n,
    )
