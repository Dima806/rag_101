"""The frozen corpus and question set -- an arena with known truth.

A small, self-contained, entirely fictional document collection (the Zephyr Archipelago,
so there is no outside knowledge a model could lean on and every fact's truth is under our
control) paired with three kinds of questions, because the project's claims are about
*where answers come from* and that requires knowing the truth:

- ``answerable``  -- a single known gold passage contains the answer (retrieval recall and
  the oracle ceiling are measurable);
- ``multi``       -- the answer is split across two passages (chunking failures are visible);
- ``absence``     -- the answer is deliberately nowhere in the collection (hallucination and
  abstention are measurable).

Running this module as a script (``python -m src.datasets.corpus`` / ``make data``) writes a
frozen copy to disk so the notebooks run with no download.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from src.config import get_settings

ANSWERABLE = "answerable"
MULTI = "multi"
ABSENCE = "absence"


@dataclass(frozen=True)
class Document:
    """One short source document."""

    doc_id: str
    text: str


@dataclass(frozen=True)
class Question:
    """A question with its known answer, gold passages, and kind.

    ``gold_doc_ids`` is empty for absence questions; ``answer`` is the expected
    (abstention) response in that case.
    """

    question_id: str
    text: str
    answer: str
    gold_doc_ids: tuple[str, ...]
    kind: str


@dataclass(frozen=True)
class Corpus:
    """The documents and questions together."""

    documents: tuple[Document, ...]
    questions: tuple[Question, ...]

    def doc_by_id(self, doc_id: str) -> Document:
        for doc in self.documents:
            if doc.doc_id == doc_id:
                return doc
        raise KeyError(doc_id)

    def gold_passages(self, question: Question) -> list[str]:
        """The text of a question's gold documents (empty for absence questions)."""
        return [self.doc_by_id(doc_id).text for doc_id in question.gold_doc_ids]

    def questions_by_kind(self, kind: str) -> list[Question]:
        return [q for q in self.questions if q.kind == kind]

    def to_dict(self) -> dict[str, object]:
        return {
            "documents": [asdict(d) for d in self.documents],
            "questions": [asdict(q) for q in self.questions],
        }


_DOCUMENTS: tuple[tuple[str, str], ...] = (
    (
        "doc01",
        "The island of Volara is the administrative centre of the Zephyr "
        "Archipelago. Its capital city is Thornmouth, which hosts the "
        "Archipelago Assembly.",
    ),
    (
        "doc02",
        "Volara has a population of 48,000 residents and covers an area of 310 square kilometres.",
    ),
    (
        "doc03",
        "The island of Myrelink is famous for its windmills. The tallest "
        "windmill, called the Graywing, stands 62 metres high.",
    ),
    (
        "doc04",
        "Myrelink's economy depends on wind power. The Graywing windmill alone "
        "supplies electricity to roughly 4,000 homes.",
    ),
    (
        "doc05",
        "Caldera Reef surrounds the volcanic island of Pelagos. Diving season "
        "on Caldera Reef runs from March to September.",
    ),
    (
        "doc06",
        "Pelagos is the only island in the archipelago with an active volcano, "
        "Mount Sear, which last erupted in 1911.",
    ),
    (
        "doc07",
        "The Zephyr Archipelago uses a single currency called the zeph. One "
        "zeph is divided into 100 breezes.",
    ),
    (
        "doc08",
        "Ferries connect the islands daily. The fastest route, Thornmouth to "
        "Myrelink, takes 90 minutes by hydrofoil.",
    ),
    (
        "doc09",
        "The archipelago's national bird is the silver tern, which nests "
        "exclusively on the cliffs of Pelagos.",
    ),
    (
        "doc10",
        "Education in the archipelago is free until the age of 18. The oldest "
        "school, Thornmouth Academy, was founded in 1842.",
    ),
    (
        "doc11",
        "The island of Suncrest grows most of the archipelago's citrus. Its "
        "orchards produce about 9,000 tonnes of oranges each year.",
    ),
    (
        "doc12",
        "Suncrest holds the annual Citrus Festival every October, drawing "
        "visitors from across the archipelago.",
    ),
    (
        "doc13",
        "The archipelago's highest point is the summit of Mount Sear, at 1,340 "
        "metres above sea level.",
    ),
    (
        "doc14",
        "Rainfall is heaviest on Myrelink, which receives about 1,800 "
        "millimetres of rain per year.",
    ),
    (
        "doc15",
        "The Archipelago Assembly has 60 members, elected every four years by "
        "the residents of all islands.",
    ),
    (
        "doc16",
        "Freshwater on Suncrest comes from the Blue Spring, the largest natural "
        "spring in the archipelago.",
    ),
)

_QUESTIONS: tuple[tuple[str, str, str, tuple[str, ...], str], ...] = (
    ("q01", "What is the capital city of Volara?", "Thornmouth", ("doc01",), ANSWERABLE),
    ("q02", "How tall is the Graywing windmill?", "62 metres", ("doc03",), ANSWERABLE),
    ("q03", "What currency does the Zephyr Archipelago use?", "the zeph", ("doc07",), ANSWERABLE),
    ("q04", "When did Mount Sear last erupt?", "1911", ("doc06",), ANSWERABLE),
    (
        "q05",
        "How many members does the Archipelago Assembly have?",
        "60 members",
        ("doc15",),
        ANSWERABLE,
    ),
    ("q06", "Where does the silver tern nest?", "the cliffs of Pelagos", ("doc09",), ANSWERABLE),
    (
        "q07",
        "How many homes does the tallest windmill on Myrelink supply with electricity?",
        "4,000 homes",
        ("doc03", "doc04"),
        MULTI,
    ),
    (
        "q08",
        "On which island is the archipelago's highest point located?",
        "Pelagos",
        ("doc13", "doc06"),
        MULTI,
    ),
    (
        "q09",
        "What is the population of Myrelink?",
        "The documents do not contain this information.",
        (),
        ABSENCE,
    ),
    (
        "q10",
        "Which island in the archipelago has an international airport?",
        "The documents do not contain this information.",
        (),
        ABSENCE,
    ),
    (
        "q11",
        "How much does a ferry ticket between the islands cost?",
        "The documents do not contain this information.",
        (),
        ABSENCE,
    ),
    (
        "q12",
        "What is the average summer temperature on Suncrest?",
        "The documents do not contain this information.",
        (),
        ABSENCE,
    ),
)


def build_corpus() -> Corpus:
    """Construct the frozen corpus and question set from the literals above."""
    documents = tuple(Document(doc_id=d, text=t) for d, t in _DOCUMENTS)
    questions = tuple(
        Question(question_id=q, text=text, answer=ans, gold_doc_ids=gold, kind=kind)
        for q, text, ans, gold, kind in _QUESTIONS
    )
    return Corpus(documents=documents, questions=questions)


def corpus_from_dict(data: dict[str, object]) -> Corpus:
    """Rebuild a :class:`Corpus` from its serialised form."""
    raw_docs = data["documents"]
    raw_qs = data["questions"]
    assert isinstance(raw_docs, list)
    assert isinstance(raw_qs, list)
    documents = tuple(Document(**d) for d in raw_docs)
    questions = tuple(
        Question(
            question_id=q["question_id"],
            text=q["text"],
            answer=q["answer"],
            gold_doc_ids=tuple(q["gold_doc_ids"]),
            kind=q["kind"],
        )
        for q in raw_qs
    )
    return Corpus(documents=documents, questions=questions)


def load_corpus(path: Path | str | None = None) -> Corpus:
    """Load the corpus from its frozen JSON copy, falling back to building it in memory."""
    corpus_path = Path(path) if path is not None else Path(get_settings().paths.corpus_file)
    if corpus_path.exists():
        return corpus_from_dict(json.loads(corpus_path.read_text()))
    return build_corpus()


def save_corpus(corpus: Corpus, path: Path | str | None = None) -> Path:
    """Write the corpus to its frozen JSON copy and return the path."""
    corpus_path = Path(path) if path is not None else Path(get_settings().paths.corpus_file)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    corpus_path.write_text(json.dumps(corpus.to_dict(), indent=2, ensure_ascii=False))
    return corpus_path


def chunk_corpus(
    corpus: Corpus,
    chunker: Callable[[str], list[str]],
) -> tuple[list[str], list[str]]:
    """Chunk every document, returning the chunks and the parallel list of source doc ids.

    The doc-id mapping is what lets retrieval metrics score a chunk hit against the
    question's gold documents.
    """
    chunks: list[str] = []
    chunk_doc_ids: list[str] = []
    for doc in corpus.documents:
        for piece in chunker(doc.text):
            chunks.append(piece)
            chunk_doc_ids.append(doc.doc_id)
    return chunks, chunk_doc_ids


def main() -> None:
    """Build and cache the corpus (invoked by ``make data``)."""
    corpus = build_corpus()
    path = save_corpus(corpus)
    print(
        f"Cached {len(corpus.documents)} documents and {len(corpus.questions)} questions to {path}"
    )


if __name__ == "__main__":
    main()
