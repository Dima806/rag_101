"""Streamlit app: see all four parts of the RAG pipeline at once.

Four views, each reading from ``src`` (the app holds no pipeline logic of its own):

1. Pipeline inspector  -- one question, every stage visible.
2. Chunking playground -- change the chunker and watch retrieval and the answer change.
3. Ceiling meter       -- oracle-passage answer vs retrieved-passage answer.
4. Hallucination probe -- an absence question, with the groundedness check toggled.

Run with ``make run``. Generation uses the Ollama model from ``config/settings.yaml``; if
Ollama is not reachable, the app says so rather than crashing.
"""

from __future__ import annotations

import streamlit as st

from src.config import load_settings
from src.datasets.corpus import chunk_corpus, load_corpus
from src.grounding.groundedness import groundedness_score
from src.pipeline.chunk import ALL_STRATEGIES, make_chunk_fn
from src.pipeline.embed import build_embedder
from src.pipeline.generate import ABSTENTION_TEXT, OllamaGenerator
from src.pipeline.rag import RagPipeline
from src.pipeline.retrieve import build_retriever

st.set_page_config(page_title="rag_101", layout="wide")


@st.cache_resource
def _load():  # noqa: ANN202 - streamlit cache wrapper
    settings = load_settings()
    corpus = load_corpus()
    embedder = build_embedder(settings.embed)
    return settings, corpus, embedder


def _build_pipeline(strategy: str, abstain: bool):  # noqa: ANN202
    settings, corpus, embedder = _load()
    chunk_fn = make_chunk_fn(
        strategy,
        size=settings.chunk.size,
        overlap=settings.chunk.overlap,
        semantic_threshold=settings.chunk.semantic_threshold,
        encoder=embedder,
    )
    chunks, chunk_doc_ids = chunk_corpus(corpus, chunk_fn)
    retriever = build_retriever(settings.retrieve.method, embedder, settings.retrieve.hybrid_alpha)
    generator = OllamaGenerator(settings.generate)
    pipeline = RagPipeline(
        retriever=retriever, generator=generator, settings=settings, abstain=abstain
    )
    pipeline.index(chunks)
    return pipeline, chunk_doc_ids, corpus, settings


def _safe_answer(pipeline: RagPipeline, question: str):  # noqa: ANN202
    try:
        return pipeline.answer(question), None
    except Exception as exc:  # noqa: BLE001 - surface any Ollama error to the UI
        return None, str(exc)


settings, corpus, _ = _load()
st.title("RAG 101 — the four parts, visible")
view = st.sidebar.radio(
    "View",
    ["Pipeline inspector", "Chunking playground", "Ceiling meter", "Hallucination probe"],
)
st.sidebar.caption("Retrieval does not fix hallucination — it just moves the blame.")

questions = [q.text for q in corpus.questions]


if view == "Pipeline inspector":
    st.header("Pipeline inspector")
    question = st.selectbox("Question", questions)
    pipeline, _, _, _ = _build_pipeline(settings.chunk.strategy, abstain=True)
    result, error = _safe_answer(pipeline, question)
    if error:
        st.error(f"Generation failed (is Ollama running? `make models`): {error}")
    elif result is not None:
        st.subheader("Retrieved passages")
        for rank, passage in enumerate(result.retrieved, start=1):
            st.markdown(f"**{rank}. score={passage.score:.3f}** — {passage.text}")
        st.subheader("Answer")
        st.write(result.answer)
        st.caption(f"groundedness = {result.groundedness:.2f} · abstained = {result.abstained}")


elif view == "Chunking playground":
    st.header("Chunking playground")
    st.write("Same question, same model — change only the chunker.")
    question = st.selectbox("Question", questions)
    strategy = st.select_slider(
        "Chunker", options=list(ALL_STRATEGIES), value=settings.chunk.strategy
    )
    pipeline, _, _, _ = _build_pipeline(strategy, abstain=True)
    result, error = _safe_answer(pipeline, question)
    if error:
        st.error(f"Generation failed (is Ollama running? `make models`): {error}")
    elif result is not None:
        st.metric("Chunker", strategy)
        for rank, passage in enumerate(result.retrieved, start=1):
            st.markdown(f"**{rank}. score={passage.score:.3f}** — {passage.text}")
        st.subheader("Answer")
        st.write(result.answer)


elif view == "Ceiling meter":
    st.header("Ceiling meter")
    st.write("The answer from the retrieved passage can never beat the one from the gold passage.")
    answerable = corpus.questions_by_kind("answerable")
    labels = [q.text for q in answerable]
    question = st.selectbox("Answerable question", labels)
    chosen = answerable[labels.index(question)]
    pipeline, _, _, _ = _build_pipeline(settings.chunk.strategy, abstain=True)
    oracle, oracle_err = None, None
    try:
        oracle = pipeline.answer_from_contexts(chosen.text, corpus.gold_passages(chosen))
    except Exception as exc:  # noqa: BLE001
        oracle_err = str(exc)
    retrieved, retrieved_err = _safe_answer(pipeline, chosen.text)
    if oracle_err or retrieved_err:
        st.error(
            f"Generation failed (is Ollama running? `make models`): {oracle_err or retrieved_err}"
        )
    else:
        col_oracle, col_retrieved = st.columns(2)
        if oracle is not None:
            col_oracle.subheader("Oracle passage")
            col_oracle.write(oracle.answer)
        if retrieved is not None:
            col_retrieved.subheader("Retrieved passage")
            col_retrieved.write(retrieved.answer)
        st.caption(f"Gold answer: {chosen.answer}")


else:  # Hallucination probe
    st.header("Hallucination probe")
    st.write("Ask something the documents do not answer, and watch the system fabricate.")
    absence = corpus.questions_by_kind("absence")
    question = st.selectbox("Absence question", [q.text for q in absence])
    use_check = st.toggle("Groundedness + abstention check", value=True)
    pipeline, _, _, _ = _build_pipeline(settings.chunk.strategy, abstain=use_check)
    result, error = _safe_answer(pipeline, question)
    if error:
        st.error(f"Generation failed (is Ollama running? `make models`): {error}")
    elif result is not None:
        st.subheader("Answer")
        st.write(result.answer)
        grounded = groundedness_score(result.raw_answer, result.contexts)
        if result.answer == ABSTENTION_TEXT:
            st.success("The system abstained — it did not fabricate.")
        else:
            st.warning("The system answered from an absence of evidence — a hallucination.")
        st.caption(f"raw groundedness = {grounded:.2f}")
