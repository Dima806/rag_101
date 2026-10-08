"""The chain: the four parts wired into one working RAG system.

``chunk -> embed -> retrieve -> generate``, plus an optional abstention path. This is the
complete, from-scratch system the reader can take apart. Every part is injected, so any
one can be swapped and measured in isolation -- which is the whole point.

Set ``abstain=True`` for the *grounded* system (refuses when retrieval is weak or the
answer is unsupported) and ``abstain=False`` for the *naive* system (always answers from
whatever was retrieved). Comparing the two is how Notebook 05 measures the abstention rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import Settings
from src.grounding.abstention import ABSTENTION_TEXT, should_abstain
from src.pipeline.generate import Generator, build_prompt
from src.pipeline.retrieve import Retrieved, Retriever


@dataclass
class RagResult:
    """Everything the chain produced for one question, so the whole chain is inspectable."""

    question: str
    retrieved: list[Retrieved]
    prompt: str
    raw_answer: str
    answer: str
    abstained: bool
    groundedness: float = 0.0

    @property
    def contexts(self) -> list[str]:
        return [r.text for r in self.retrieved]

    @property
    def scores(self) -> list[float]:
        return [r.score for r in self.retrieved]


@dataclass
class RagPipeline:
    """Hold the retriever and generator; answer questions over an indexed corpus."""

    retriever: Retriever
    generator: Generator
    settings: Settings
    abstain: bool = True
    _indexed: bool = field(default=False, init=False, repr=False)

    def index(self, chunks: list[str]) -> None:
        """Index the pre-chunked corpus into the retriever."""
        self.retriever.index(chunks)
        self._indexed = True

    def answer(self, question: str, top_k: int | None = None) -> RagResult:
        """Run the full chain for one question and return every intermediate stage."""
        if not self._indexed:
            raise RuntimeError("call index(chunks) before answer()")
        k = top_k if top_k is not None else self.settings.retrieve.top_k
        retrieved = self.retriever.retrieve(question, k)
        contexts = [r.text for r in retrieved]
        prompt = build_prompt(question, contexts)
        raw_answer = self.generator.generate(prompt)
        return self._finalize(question, retrieved, prompt, raw_answer)

    def answer_from_contexts(self, question: str, contexts: list[str]) -> RagResult:
        """Generate from passages supplied by hand -- used for the oracle ceiling experiment.

        Bypasses the retriever so the generator can be fed the gold passage directly,
        isolating what the generator can do when retrieval is perfect.
        """
        retrieved = [Retrieved(index=-1, score=1.0, text=c) for c in contexts]
        prompt = build_prompt(question, contexts)
        raw_answer = self.generator.generate(prompt)
        return self._finalize(question, retrieved, prompt, raw_answer)

    def _finalize(
        self,
        question: str,
        retrieved: list[Retrieved],
        prompt: str,
        raw_answer: str,
    ) -> RagResult:
        from src.grounding.groundedness import groundedness_score

        contexts = [r.text for r in retrieved]
        scores = [r.score for r in retrieved]
        grounded = groundedness_score(raw_answer, contexts)
        answer, abstained = raw_answer, False
        if self.abstain and should_abstain(
            raw_answer,
            contexts,
            scores,
            retrieval_score_threshold=self.settings.grounding.retrieval_score_threshold,
            groundedness_threshold=self.settings.grounding.groundedness_threshold,
        ):
            answer, abstained = ABSTENTION_TEXT, True
        return RagResult(
            question=question,
            retrieved=retrieved,
            prompt=prompt,
            raw_answer=raw_answer,
            answer=answer,
            abstained=abstained,
            groundedness=grounded,
        )
