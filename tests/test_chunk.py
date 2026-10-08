"""Chunkers respect their size/boundary rules and lose no content."""

from __future__ import annotations

import re

from src.pipeline.chunk import (
    SemanticChunker,
    fixed_chunker,
    recursive_chunker,
    sentence_chunker,
    split_sentences,
)
from src.pipeline.embed import HashingEmbedder

SAMPLE = (
    "The harbor opens at dawn. Fishermen leave before sunrise. "
    "They return by noon with the day's catch.\n\n"
    "The market sells fish and bread. Prices rise in winter. "
    "Tourists crowd the stalls each summer.\n\n"
    "A lighthouse guards the eastern cliffs. It was built a century ago. "
    "Its lamp can be seen for miles."
)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def test_fixed_chunker_respects_size_and_loses_no_characters() -> None:
    chunks = fixed_chunker(SAMPLE, size=60, overlap=0)
    assert len(chunks) > 1
    assert all(0 < len(c) <= 60 for c in chunks)
    # No characters lost or added: with zero overlap the chunks tile the text exactly.
    assert "".join(chunks) == SAMPLE.strip()


def test_fixed_chunker_overlap_carries_characters_forward() -> None:
    chunks = fixed_chunker(SAMPLE, size=60, overlap=15)
    assert len(chunks) > 1
    assert all(len(c) <= 60 for c in chunks)
    # The last 15 chars of one chunk reappear as the first 15 of the next.
    assert chunks[0][-15:] == chunks[1][:15]


def test_sentence_chunker_never_cuts_mid_sentence() -> None:
    chunks = sentence_chunker(SAMPLE, size=120)
    assert chunks
    # Re-splitting every chunk into sentences recovers exactly the original sentences.
    recovered = [s for chunk in chunks for s in split_sentences(chunk)]
    assert recovered == split_sentences(SAMPLE)
    assert _words(" ".join(chunks)) == _words(SAMPLE)


def test_recursive_chunker_respects_size_and_preserves_words() -> None:
    chunks = recursive_chunker(SAMPLE, size=120)
    assert chunks
    assert all(len(c) <= 120 for c in chunks)
    assert _words(" ".join(chunks)) == _words(SAMPLE)


def test_semantic_chunker_preserves_words_and_is_nonempty() -> None:
    encoder = HashingEmbedder(dim=128)
    chunks = SemanticChunker(encoder, threshold=0.55, size=120).chunk(SAMPLE)
    assert chunks
    assert all(c.strip() for c in chunks)
    assert _words(" ".join(chunks)) == _words(SAMPLE)


def test_empty_input_yields_no_chunks() -> None:
    assert fixed_chunker("   ", size=50) == []
    assert sentence_chunker("", size=50) == []
    assert recursive_chunker("\n\n", size=50) == []
