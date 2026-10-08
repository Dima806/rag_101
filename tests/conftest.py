"""Shared fixtures: an offline embedder and a stubbed generator, so CI needs no model.

The whole test suite runs deterministically with no network and no Ollama:

- the embedder is the :class:`~src.pipeline.embed.HashingEmbedder` (no download);
- the generator is the scripted stand-in from :mod:`src.support` -- built from the corpus's
  known answers, it returns the correct answer only when that answer is actually present in
  the retrieved context, and otherwise fabricates a confident, ungrounded answer (never
  abstaining on its own). That is what makes the ceiling and abstention claims checkable
  offline, and it is the same generator the notebooks use when no model is running.
"""

from __future__ import annotations

import pytest

from src.config import EmbedSettings, Settings
from src.datasets.corpus import Corpus, build_corpus
from src.pipeline.embed import CachedEmbedder, build_embedder
from src.pipeline.generate import CallableGenerator
from src.support import make_scripted_generator


@pytest.fixture
def corpus() -> Corpus:
    return build_corpus()


@pytest.fixture
def settings() -> Settings:
    """Settings with the offline hashing embedder; everything else at its default."""
    return Settings(embed=EmbedSettings(backend="hashing", dim=256))


@pytest.fixture
def embedder(settings: Settings) -> CachedEmbedder:
    return build_embedder(settings.embed)


@pytest.fixture
def scripted_generator(corpus: Corpus) -> CallableGenerator:
    return make_scripted_generator(corpus)
