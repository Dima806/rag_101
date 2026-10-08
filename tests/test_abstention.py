"""On absence questions, the grounded system abstains more than the naive one.

The honest crux: by default a RAG system answers absence questions with a confident,
fabricated answer. Adding the abstention path (retrieval-score threshold + groundedness
check) raises how often it correctly says "the documents do not contain this".
"""

from __future__ import annotations

from src.config import Settings
from src.datasets.corpus import Corpus, chunk_corpus
from src.evaluation.answer_metrics import abstention_rate
from src.grounding.abstention import is_abstention
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
    *,
    abstain: bool,
) -> RagPipeline:
    chunk_fn = make_chunk_fn(
        settings.chunk.strategy,
        size=settings.chunk.size,
        overlap=settings.chunk.overlap,
        encoder=embedder,
    )
    chunks, _ = chunk_corpus(corpus, chunk_fn)
    retriever = build_retriever(settings.retrieve.method, embedder)
    pipeline = RagPipeline(
        retriever=retriever, generator=generator, settings=settings, abstain=abstain
    )
    pipeline.index(chunks)
    return pipeline


def _abstention_rate_on_absence(
    pipeline: RagPipeline, corpus: Corpus, settings: Settings
) -> float:
    answers = [
        pipeline.answer(q.text, top_k=settings.retrieve.top_k).answer
        for q in corpus.questions_by_kind("absence")
    ]
    return abstention_rate(answers)


def test_grounded_abstains_more_than_naive_on_absence(
    corpus: Corpus,
    embedder: CachedEmbedder,
    scripted_generator: CallableGenerator,
    settings: Settings,
) -> None:
    naive = _pipeline(corpus, embedder, scripted_generator, settings, abstain=False)
    grounded = _pipeline(corpus, embedder, scripted_generator, settings, abstain=True)

    naive_rate = _abstention_rate_on_absence(naive, corpus, settings)
    grounded_rate = _abstention_rate_on_absence(grounded, corpus, settings)

    assert grounded_rate > naive_rate


def test_naive_system_fabricates_on_absence(
    corpus: Corpus,
    embedder: CachedEmbedder,
    scripted_generator: CallableGenerator,
    settings: Settings,
) -> None:
    naive = _pipeline(corpus, embedder, scripted_generator, settings, abstain=False)
    for question in corpus.questions_by_kind("absence"):
        result = naive.answer(question.text, top_k=settings.retrieve.top_k)
        # It produces a confident answer and does not abstain -- hallucination.
        assert not result.abstained
        assert not is_abstention(result.answer)


def test_grounded_system_still_answers_answerable(
    corpus: Corpus,
    embedder: CachedEmbedder,
    scripted_generator: CallableGenerator,
    settings: Settings,
) -> None:
    grounded = _pipeline(corpus, embedder, scripted_generator, settings, abstain=True)
    # The abstention path must not make the system refuse questions it can answer.
    answered = 0
    for question in corpus.questions_by_kind("answerable"):
        result = grounded.answer(question.text, top_k=settings.retrieve.top_k)
        if not result.abstained:
            answered += 1
    assert answered > 0
