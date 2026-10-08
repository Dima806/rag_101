"""Support glue for the notebooks and the app -- artifacts and generator selection.

The notebooks and the Streamlit app hold no core logic; the reusable glue lives here:

- :func:`save_fig` / :func:`save_json` write **every figure as a PNG** under
  ``outputs/figures/`` and **every textual/numerical result as JSON** under
  ``outputs/results/``, so the Medium article (and any reader) can rebuild the story from
  the committed artifacts without re-running generation.
- :func:`make_scripted_generator` is the offline, deterministic stand-in for the tiny local
  model: it answers from the retrieved context when the gold answer is actually present,
  and otherwise fabricates a confident, ungrounded answer. That is the behaviour the
  project demonstrates, recorded so notebooks run with no Ollama.
- :func:`select_generator` prefers a live Ollama model and falls back to the stand-in,
  returning the mode string so a notebook can state which it used.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from src.config import Settings, get_settings
from src.datasets.corpus import Corpus
from src.pipeline.generate import CallableGenerator, Generator, OllamaGenerator

FIGURES_DIR = Path(get_settings().paths.figures_dir)
RESULTS_DIR = Path("outputs/results")

# A confident, fluent answer built from tokens that appear nowhere in the corpus, so its
# groundedness against any retrieved context is ~0 -- it reads as a hallucination.
FABRICATION = "Based on the retrieved passages, the figure is roughly seventeen zettabytes."

_QUESTION_RE = re.compile(r"Question:\s*(.*?)\s*\nAnswer:", re.DOTALL)


def _json_default(obj: object) -> object:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def save_fig(fig: Figure, stem: str, dpi: int = 150) -> Path:
    """Save a matplotlib figure as ``outputs/figures/<stem>.png`` and return the path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{stem}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def save_json(data: Any, stem: str) -> Path:
    """Save any JSON-able result as ``outputs/results/<stem>.json`` and return the path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{stem}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_default))
    return path


def make_scripted_generator(corpus: Corpus) -> CallableGenerator:
    """An offline generator that answers from context when the gold answer is present.

    It is deterministic and label-aware (a recorded stand-in): for an answerable question
    it returns the known answer *only if that answer actually appears in the retrieved
    context*, which is how chunking and retrieval failures turn into wrong answers. For an
    absence question it always fabricates -- never abstaining on its own.
    """
    question_by_text = {q.text: q for q in corpus.questions}

    def generate(prompt: str) -> str:
        match = _QUESTION_RE.search(prompt)
        if match is None:
            return FABRICATION
        question = question_by_text.get(match.group(1).strip())
        if question is None or not question.gold_doc_ids:
            return FABRICATION
        context = prompt.split("Question:")[0].lower()
        return question.answer if question.answer.lower() in context else FABRICATION

    return CallableGenerator(generate)


def ollama_available(settings: Settings | None = None) -> bool:
    """True if an Ollama server answers at the configured base URL."""
    settings = settings or get_settings()
    try:
        import httpx

        httpx.get(f"{settings.generate.base_url.rstrip('/')}/api/tags", timeout=1.0)
    except Exception:
        return False
    return True


def select_generator(
    corpus: Corpus,
    settings: Settings | None = None,
) -> tuple[Generator, str]:
    """Return (generator, mode): a live Ollama model if reachable, else the stand-in."""
    settings = settings or get_settings()
    if ollama_available(settings):
        return OllamaGenerator(settings.generate), f"ollama:{settings.generate.model_name}"
    return make_scripted_generator(corpus), "scripted-offline"
