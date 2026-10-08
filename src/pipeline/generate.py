"""Generators: the visible part everyone tunes, shown to be the least impactful.

A generator is anything that maps a prompt string to an answer string. Two live here:

- :class:`OllamaGenerator` -- a tiny local model served by Ollama, called over HTTP with
  ``httpx`` (the CPU setup from ``llm_judge_benchmark``).
- :class:`CallableGenerator` -- wraps a plain function. This is the seam that lets CI run
  without a live model: tests inject a stubbed or recorded generator and the deterministic
  parts of the pipeline (chunking, retrieval, the ceiling) run green with no Ollama.

:func:`build_prompt` is the one place prompt assembly lives -- the retrieved context, the
question, and the instruction to answer only from the context and to abstain when the
answer is absent. Keeping it here (not retyped per notebook) means the abstention
behaviour is a property of the system, not of how someone phrased a cell.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from src.config import GenerateSettings

ABSTENTION_TEXT = "The documents do not contain this information."

PROMPT_TEMPLATE = """\
You are a careful assistant. Answer the question using ONLY the context below.
If the context does not contain the answer, reply exactly: "{abstention}"
Do not use any outside knowledge. Keep the answer short.

Context:
{context}

Question: {question}
Answer:"""


def build_prompt(question: str, contexts: list[str]) -> str:
    """Assemble the generation prompt from the retrieved passages and the question."""
    numbered = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    if not numbered:
        numbered = "(no passages retrieved)"
    return PROMPT_TEMPLATE.format(
        abstention=ABSTENTION_TEXT,
        context=numbered,
        question=question.strip(),
    )


class Generator(Protocol):
    """Anything that turns a prompt into an answer."""

    def generate(self, prompt: str) -> str: ...


class CallableGenerator:
    """Adapt a ``str -> str`` function into a :class:`Generator`.

    The injection seam for tests and notebooks: pass a stub that returns canned or
    recorded answers and the rest of the pipeline is unchanged.
    """

    def __init__(self, fn: Callable[[str], str]) -> None:
        self._fn = fn

    def generate(self, prompt: str) -> str:
        return self._fn(prompt)


class OllamaGenerator:
    """Call a tiny local model through Ollama's HTTP API with ``httpx``."""

    def __init__(self, settings: GenerateSettings) -> None:
        self.settings = settings

    def generate(self, prompt: str) -> str:
        import httpx

        response = httpx.post(
            f"{self.settings.base_url.rstrip('/')}/api/generate",
            json={
                "model": self.settings.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.settings.temperature,
                    "num_predict": self.settings.max_tokens,
                },
            },
            timeout=self.settings.timeout,
        )
        response.raise_for_status()
        return str(response.json().get("response", "")).strip()


def build_generator(settings: GenerateSettings) -> Generator:
    """Construct the generator named in settings (only ``ollama`` is built here).

    The ``callable`` backend has no default implementation on purpose: a stub or recorded
    generator is injected by the caller (tests, notebooks) rather than configured.
    """
    if settings.backend == "ollama":
        return OllamaGenerator(settings)
    raise ValueError(
        f"backend {settings.backend!r} must be injected as a CallableGenerator by the caller"
    )
