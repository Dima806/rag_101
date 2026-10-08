# rag_101

**Retrieval does not fix hallucination — it just moves the blame.**

A hands-on, from-scratch introduction to retrieval-augmented generation, built end to end
to run on a **2-CPU / 8 GB GitHub Codespace with no GPU**. It proves, by measurement, that a
RAG system is four plain parts you can assemble in an afternoon — **chunk → embed → retrieve
→ generate** — and that:

1. **Retrieval recall is the ceiling.** The generator can only answer from what it is given;
   final answer accuracy can never beat the retrieval. So measure retrieval *first*.
2. **Chunking dominates.** Hold the model fixed and change only the chunker, and the answer
   swings from mostly-wrong to mostly-right. The cheap step nobody tunes decides the quality.
3. **The generator is the least impactful knob.** Under a retrieval bottleneck, a bigger
   model barely moves the answer.
4. **RAG reduces but does not remove hallucination.** On questions whose answer is absent,
   the model fabricates confident answers — sometimes quoting a retrieved passage that does
   not support the claim. *Hallucination with a footnote.*

Everything is open-source and CPU-only: a small `sentence-transformers` embedder and a tiny
local model served by **Ollama** (`qwen2.5:0.5b`).

---

## Quickstart

```bash
make setup      # install deps with uv, register the Jupyter kernel
make models     # (optional) install Ollama + pull qwen2.5:0.5b for live generation
make data       # build & cache the frozen corpus
make test       # run the test suite (no model needed)
make notebooks  # execute all 6 notebooks headless, writing figures + results to outputs/
make run        # launch the Streamlit app on :8501
```

> **No Ollama? Still works.** The notebooks and app prefer a live Ollama model but fall back
> to a deterministic offline generator, so everything runs (and the tests always do) with no
> model and no network. Run `make models` only when you want live generation.

Requires Python 3.11+ (the repo pins 3.12) and [`uv`](https://docs.astral.sh/uv/). Torch is
installed CPU-only — no CUDA, no GPU.

---

## The notebooks

Run them in order; each saves its figures to `outputs/figures/*.png` and its numbers to
`outputs/results/*.json`.

| # | Notebook | The point |
|---|----------|-----------|
| 01 | [Four Parts](notebooks/01_four_parts.ipynb) | Build the four parts as four functions, wire the chain, answer a question end to end. |
| 02 | [Chunking Decides Everything](notebooks/02_chunking_decides_everything.ipynb) | Model fixed, vary only the chunker/size — answer quality swings wildly. The showpiece. |
| 03 | [Retrieval Is the Ceiling](notebooks/03_retrieval_is_the_ceiling.ipynb) | recall@k alone, then show answer accuracy can never exceed evidence-recall. |
| 04 | [Generation and Grounding](notebooks/04_generation_and_grounding.ipynb) | Swap 0.5b → 1.5b: barely moves under a retrieval bottleneck. Groundedness ≠ correctness. |
| 05 | [When RAG Hallucinates](notebooks/05_when_rag_hallucinates.ipynb) | Absence questions; a fabricated answer that quotes the context scores *high* groundedness. The abstention lever: safety vs coverage. |
| 06 | [Decision Framework](notebooks/06_decision_framework.ipynb) | Build-and-debug checklist; RAG vs fine-tune vs prompt. |

Every figure and result table also lands in [`outputs/`](outputs/) so the write-up can be
rebuilt from committed artifacts without re-running generation.

---

## How it's built

Four separable, independently measurable parts plus the harness to measure each one:

```
src/
  config.py            # typed settings loaded from config/settings.yaml (the single source of knobs)
  pipeline/
    chunk.py           # fixed / sentence-aware / recursive / semantic chunkers
    embed.py           # small sentence-transformer (cached) + an offline hashing backend
    retrieve.py        # cosine top-k over numpy + BM25 + hybrid
    generate.py        # tiny Ollama model via httpx; prompt assembly; a stubbable seam
    rag.py             # the end-to-end chain + an abstention path
  grounding/           # groundedness (is the answer supported?) + abstention (should it refuse?)
  datasets/corpus.py   # a frozen, fictional corpus + answerable / multi-passage / absence questions
  evaluation/          # recall@k & MRR, groundedness/correctness/abstention, the ceiling, ablations
  support.py           # artifact saving (PNG/JSON) + generator selection (live Ollama or offline)
  visualisation.py     # shared matplotlib helpers
app/streamlit_app.py   # pipeline inspector / chunking playground / ceiling meter / hallucination probe
notebooks/             # 01..06
tests/                 # chunkers, retrieval, the ceiling invariant, abstention
```

The corpus is **fictional** on purpose (the Zephyr Archipelago): the model has no training
priors to lean on, so an answer it gets right *must* have come from a retrieved passage —
which is what makes "accuracy ≤ retrieval recall" a clean, measurable fact.

---

## Related projects

Part of a from-scratch series on how retrieval, embeddings and small models actually work:

- **[retrieval_arena](https://github.com/Dima806/retrieval_arena)** — benchmarks retrievers
  (dense / BM25 / hybrid) head to head. The retrieval half of RAG, measured; `rag_101` builds
  the whole pipeline around it.
- **[embeddings_101](https://github.com/Dima806/embeddings_101)** — how text embeddings work
  from scratch. The vectors retrieval runs on.
- **[graphrag_vs_vectorrag](https://github.com/Dima806/graphrag_vs_vectorrag)** — graph RAG vs
  vector RAG architectures compared; `rag_101` is the from-scratch vector-RAG foundation under
  both.
- **[llm_judge_benchmark](https://github.com/Dima806/llm_judge_benchmark)** — LLM-as-judge
  evaluation and the tiny-local-model-on-CPU (Ollama) setup reused here; source of the
  RAG-triad metrics.
- **[lora_101](https://github.com/Dima806/lora_101)** *(companion — private for now, public
  soon)* — fine-tuning with LoRA: the "when *not* to use RAG" fork — style and format vs.
  changing document facts.

More at [github.com/Dima806](https://github.com/Dima806).

---

## License

[Apache-2.0](LICENSE).
