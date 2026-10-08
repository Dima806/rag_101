"""Typed settings for rag_101, loaded from ``config/settings.yaml``.

Every knob in the pipeline -- chunk sizes, top-k, model names, seeds, thresholds --
lives in the YAML file and is validated here. Code and notebooks read from a
:class:`Settings` instance rather than hardcoding literals, so a single file controls
the whole experiment and runs stay reproducible.

Environment variables (``RAG_SEED``, ``RAG_RETRIEVE__TOP_K``, ... -- the ``__`` delimiter
descends into nested sections) fill in any key not set in the YAML file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_CONFIG_PATH = Path("config/settings.yaml")


class ChunkSettings(BaseModel):
    """How documents are split into passages -- the highest-leverage choice."""

    strategy: str = "recursive"
    size: int = 400
    overlap: int = 50
    semantic_threshold: float = 0.55


class EmbedSettings(BaseModel):
    """Which embedder turns text into vectors, and where its cache lives."""

    backend: str = "sentence_transformer"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dim: int = 256
    cache_dir: str = "data/cache/embeddings"


class RetrieveSettings(BaseModel):
    """The retriever: method, how many passages, and the hybrid mix."""

    method: str = "cosine"
    top_k: int = 5
    hybrid_alpha: float = 0.5


class GenerateSettings(BaseModel):
    """The generator: which tiny local model and how it is prompted."""

    backend: str = "ollama"
    base_url: str = "http://localhost:11434"
    model_name: str = "qwen2.5:0.5b"
    temperature: float = 0.0
    max_tokens: int = 256
    timeout: float = 120.0


class GroundingSettings(BaseModel):
    """Thresholds for the groundedness check and the abstention path."""

    groundedness_threshold: float = 0.3
    retrieval_score_threshold: float = 0.15


class PathSettings(BaseModel):
    """Where data and figures live on disk."""

    data_dir: str = "data"
    corpus_file: str = "data/corpus/corpus.json"
    figures_dir: str = "outputs/figures"


class Settings(BaseSettings):
    """Root settings object; the single source of truth for every knob."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    seed: int = 42
    chunk: ChunkSettings = Field(default_factory=ChunkSettings)
    embed: EmbedSettings = Field(default_factory=EmbedSettings)
    retrieve: RetrieveSettings = Field(default_factory=RetrieveSettings)
    generate: GenerateSettings = Field(default_factory=GenerateSettings)
    grounding: GroundingSettings = Field(default_factory=GroundingSettings)
    paths: PathSettings = Field(default_factory=PathSettings)


def load_settings(path: Path | str = DEFAULT_CONFIG_PATH) -> Settings:
    """Build settings from a YAML file; env vars supply anything the file omits."""
    config_path = Path(path)
    if not config_path.exists():
        return Settings()
    loaded: Any = yaml.safe_load(config_path.read_text())
    if not isinstance(loaded, dict):
        return Settings()
    data: dict[str, Any] = {str(key): value for key, value in loaded.items()}
    return Settings(**data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` built from the default config path."""
    return load_settings()
