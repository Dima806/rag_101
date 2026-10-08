# CLAUDE.md — rag_101

Guidance for Claude Code when working in this repository. Read this before writing
code, notebooks, or tests. The full spec lives in [.llm/PRD_rag_101.md](.llm/PRD_rag_101.md);
this file is the working contract.

---

## 1. What this project is

`rag_101` is an **educational, from-scratch** retrieval-augmented-generation pipeline,
built end to end to run on a **2-CPU / 8 GB GitHub Codespace with no GPU**. It is a
portfolio/teaching project, not a production system. Everything is open-source and CPU-only.

It ships as a GitHub repo with **6 Jupyter notebooks**, a **`src/` library**, and a
**Streamlit app**, all driven from a `Makefile`.

### The thesis (this drives every design decision)

RAG is four separable parts wired in series — **chunk → embed → retrieve → generate** —
and the project exists to make four claims *measurable*, not just assertable:

1. **Retrieval recall is the ceiling.** The generator can only answer from what it is
   given. Final answer accuracy can never exceed retrieval recall. So **measure retrieval
   in isolation first** — it is where the leverage is.
2. **Chunking dominates.** Holding embedder/retriever/generator fixed and changing *only*
   the chunker swings answer quality more than swapping the model does. The cheap step
   nobody tunes decides most of the quality.
3. **The generator is the least impactful knob.** Under a retrieval bottleneck, a bigger
   model barely moves the answer. Generation matters only for *faithfully using* the
   provided context.
4. **RAG reduces but does not remove hallucination.** On questions whose answer is absent
   from the corpus, the model fabricates confident, fluent answers — sometimes citing a
   retrieved passage that does not support the claim. "Hallucination with a footnote."

**When you write code, notebooks, or tests, serve these claims.** If a change would make
one of them harder to demonstrate or measure, it is probably wrong. The claims are
*relative* (chunking moves it, retrieval caps it, absence triggers fabrication) and hold
regardless of how weak the tiny local model is — do not "fix" weak answers by reaching for
a bigger model; that defeats the lesson.

---

## 2. Tech stack & hard constraints

- **Python 3.11+**. Package manager is **`uv`** — never `pip`, never `conda`, never a bare
  `python`/`pytest`; always `uv run ...` (or a `make` target). The repo pins **Python 3.12**
  (`.python-version`) and installs **CPU-only torch** via a `[tool.uv.sources]` pin in
  `pyproject.toml` — no CUDA wheels, no GPU.
- **No GPU, ever.** 2 CPU cores, 8 GB RAM is the target. Keep everything CPU-fast.
- **Embeddings:** `sentence-transformers`, a *small* model — `all-MiniLM-L6-v2` or
  `bge-small` (tens of MB). Cache embeddings so notebooks stay quick.
- **Generation:** a **tiny local model served by Ollama** (e.g. `qwen2.5:0.5b` or
  `llama3.2:1b`), called over HTTP with **`httpx`**. This mirrors the `llm_judge_benchmark`
  CPU setup. Do not add a hosted-API LLM dependency.
- **Vector store is numpy.** Cosine top-k over a numpy matrix is the honest default at this
  scale (the `retrieval_arena` lesson). Also build **BM25** (`rank-bm25`) and a **hybrid**.
  Do not introduce FAISS, Chroma, or a managed vector DB.
- **No LangChain / LlamaIndex.** The entire point is to build the four parts transparently;
  a framework that hides them defeats the project. Do not add one.

Runtime deps: `sentence-transformers`, `rank-bm25`, `httpx`, `numpy`, `scipy`, `pandas`,
`matplotlib`, `plotly`, `streamlit`, `pydantic`, `pydantic-settings`, `pyyaml`.
Dev deps: `pytest`, `ruff`, `ty`, `jupyter`, `ipykernel`, `nbconvert`, `nbclient`.

---

## 3. Repository layout

```
config/settings.yaml          # Chunk sizes, top-k, model names, seeds — single source of knobs
src/
  config.py                   # pydantic-settings loader for settings.yaml
  pipeline/
    chunk.py                  # fixed / sentence-aware / recursive / semantic chunkers
    embed.py                  # small sentence-transformer embedder, cached
    retrieve.py               # cosine top-k (numpy) + BM25 + hybrid
    generate.py               # Ollama tiny model via httpx; prompt assembly with context
    rag.py                    # the end-to-end chain tying the four parts together
  grounding/
    groundedness.py           # is the answer supported by the retrieved context?
    abstention.py             # does the system say "not in the documents" when it should?
  datasets/
    corpus.py                 # doc collection + question sets + gold/absence labels
  evaluation/
    retrieval_metrics.py      # recall@k, MRR, hit rate (retrieval in isolation)
    answer_metrics.py         # groundedness, correctness, abstention rate; evidence_present
    ceiling.py                # oracle-passage vs retrieved-passage answer accuracy
    ablations.py              # chunking sweep, model swap, top-k sweep
  support.py                  # notebook/app glue: save_fig (PNG) / save_json + generator selection
  visualisation.py            # shared plotting helpers (matplotlib)
notebooks/                    # 01..06, see §7 (executed, with outputs committed)
app/streamlit_app.py          # pipeline inspector / chunking playground / ceiling / hallucination probe
tests/                        # test_chunk / test_retrieve / test_ceiling / test_abstention
data/corpus/                  # frozen corpus committed here (notebooks run with no download)
data/cache/                   # model/embedding caches + Ollama models — GITIGNORED, never commit
outputs/figures/              # every notebook figure as a PNG (committed)
outputs/results/              # every notebook result table/metric as JSON (committed)
```

**Dependency direction:** `datasets` and `config` are leaves. `pipeline` parts depend only
on `config`/`datasets`. `grounding` and `evaluation` depend on `pipeline`. `app` and
`notebooks` sit on top and import from `src`; they contain **no core logic** — anything
reusable belongs in `src/`.

---

## 4. Commands (always via `make`)

```
make setup        # first-time: install uv, uv sync --all-extras, register ipykernel
make models       # install Ollama if missing + pull the tiny model (qwen2.5:0.5b)
make data         # build/cache the corpus and question set into data/
make lint         # ruff format + ruff check --fix + ty check   (run before committing)
make test         # pytest
make test-cov     # pytest with coverage
make notebooks    # execute all notebooks headless (nbconvert, 600s/cell timeout)
make run          # streamlit app on :8501
make lab          # JupyterLab on :8888
make ci           # sync + lint + test  (what CI runs)
make dev          # lint + test  (fast local loop)
make clean/reset  # clean caches / full reset (drops .venv and data/corpus)
```

Before saying work is done: `make lint` and `make test` must pass with **zero errors**.

---

## 5. Coding conventions

- **Line length 99**, `ruff` with `E,F,W,I,UP,N,B,A,SIM,PTH` (ignore `E501`). Note **PTH**:
  use `pathlib.Path`, not `os.path`. Note **A**: don't shadow builtins (`id`, `type`,
  `input`, ...).
- **Type-checked with `ty`** (`make typecheck` runs `ty check src/`). Add type hints to all
  public functions; keep signatures honest.
- **All configuration flows through `config/settings.yaml` → `src/config.py`** (pydantic
  `BaseSettings`). Chunk sizes, top-k, model names, and **seeds** are config, not literals
  scattered in code or notebooks. A notebook that needs a knob reads it from settings.
- **Reproducibility is a requirement, not a nicety.** Seed retrieval and any sampling. The
  corpus and question set are **frozen and committed** so notebooks run offline. Any claim
  about the ceiling or abstention is asserted against the **known gold/absence labels**.
- Keep each pipeline part **small and independently callable** — the pedagogy depends on the
  reader (and the tests) being able to run and measure one part alone. Resist clever
  abstractions that fuse the four parts; they must stay separable.
- Prefer pure functions with explicit inputs/outputs over hidden state. Cache deliberately
  (embeddings, generation outputs) and make the cache visible/clearable.

---

## 6. Datasets — the three question kinds

`src/datasets/corpus.py` produces a small, self-contained, permissively-sourced document
collection (a few dozen to a few hundred short docs) and **three** question kinds. The
project's claims require knowing the ground truth:

1. **Answerable** — a single known **gold passage** contains the answer. Enables retrieval
   recall and the oracle ceiling.
2. **Multi-passage** — the answer is split across chunks. Exposes chunking failures
   (mid-sentence cuts, undersized chunks orphaning the answer).
3. **Absence** — the answer is deliberately **nowhere** in the collection. Makes
   hallucination vs. abstention measurable.

Gold-passage and absence labels are the backbone of the test assertions and the ceiling
experiment. Do not add questions without labels.

---

## 7. Notebooks

Order and purpose (narrative build-up — keep it intact):

1. **01_four_parts** — build chunker/embedder/retriever/generator as four functions, wire
   the chain, answer one question end to end. State the thesis up front.
2. **02_chunking_decides_everything** (showpiece) — model fixed, vary only the chunker;
   watch answer quality swing. Show the mechanisms (blur / lost context / orphaned answers).
3. **03_retrieval_is_the_ceiling** — recall@k & MRR alone, then whole-system accuracy;
   oracle-passage vs retrieved-passage; the second tracks recall.
4. **04_generation_and_grounding** — swap tiny→larger model, show it barely helps under a
   retrieval bottleneck; introduce groundedness as distinct from correctness.
5. **05_when_rag_hallucinates** (honest crux) — absence questions; measure the low default
   abstention rate; show prompting + score threshold + groundedness check raise it.
6. **06_decision_framework** — the build-and-debug guide; RAG vs fine-tune vs prompt.

Notebook rules:
- **Each notebook must run in < 8 minutes** on the 2-CPU Codespace (generation-bound;
  cached between runs). The heaviest cell is the chunking sweep over the question set.
- Notebooks pick their generator with `support.select_generator(corpus, settings)`: a **live
  Ollama model** if one is reachable, otherwise the **deterministic offline stand-in** (same
  one the tests use). So `make notebooks` runs with or without `make models`.
- **Every figure is saved as a PNG** (`support.save_fig` → `outputs/figures/NN_*.png`) and
  **every textual/numerical result as JSON** (`support.save_json` → `outputs/results/NN_*.json`),
  so the write-up rebuilds from committed artifacts without re-running generation.
- Notebooks import from `src/`; they **don't** redefine pipeline logic. Executed in order by
  `make notebooks`, with outputs committed.
- The corpus is tiny with a weak CPU model, so some numbers are non-monotonic and the
  oracle can even sit below retrieved accuracy (the single oracle passage gives the weak
  model less context than top-k). The robust, asserted invariant is **answer accuracy ≤
  evidence-recall** (`evidence_present`: was the answer *text* retrieved, not just the
  document); the deterministic `oracle ≥ retrieved` form lives in `tests/test_ceiling.py`.

---

## 8. Testing & asserted invariants

Tests live in `tests/` and encode the project's measurable claims:

- `test_chunk.py` — chunkers respect their size/boundary rules; **no characters lost**.
- `test_retrieve.py` — the retriever returns the planted gold passage on easy queries.
- `test_ceiling.py` — **oracle-passage accuracy ≥ retrieved-passage accuracy** (asserted);
  retrieved-passage accuracy tracks retrieval recall.
- `test_abstention.py` — on absence questions, the grounded system **abstains more** than
  the naive one.

**CI must pass without a live model.** Make the generator **stubbable/recordable** (inject a
fake or use recorded outputs) so the deterministic parts — chunking, retrieval, ceiling —
run green on CI without Ollama. `make models` is only for full local runs.

---

## 9. Ollama / generation notes

- Talk to Ollama over HTTP with `httpx` (default `http://localhost:11434`). Make the base
  URL and model name config, not hardcoded.
- The generation **prompt must** include: the retrieved context, the question, and an
  explicit instruction to **answer only from the context and abstain if the answer is
  absent**. The abstention behavior is a measured outcome, so keep the prompt in one place
  and versioned in config/code (not retyped per notebook).
- Short answers, low temperature for reproducibility. Don't stream in library code used by
  tests.

---

## 10. Streamlit app

Four views, all reading from `src/` (no logic of its own):
1. **Pipeline inspector** — one question, all four stages visible (chunks, similarities,
   retrieved passages + scores, final answer).
2. **Chunking playground** — change chunker/size, watch retrieved passages and answer change.
3. **Ceiling meter** — oracle-passage answer vs retrieved-passage answer, recall shown as cap.
4. **Hallucination probe** — absence question; abstain vs fabricate, groundedness check toggle.

---

## 11. Do / don't quick reference

**Do**
- Run everything through `uv` / `make`.
- Keep the four parts separable and independently measurable.
- Put knobs and seeds in `config/settings.yaml`.
- Assert the ceiling and abstention claims against known labels.
- Make generation stubbable so CI is green without Ollama.

**Don't**
- Add LangChain/LlamaIndex, FAISS/Chroma, a GPU, or a hosted LLM API.
- Use `pip`, `os.path`, or hardcoded model names / ports / seeds.
- Put reusable logic in notebooks or the Streamlit app.
- "Fix" weak tiny-model answers by swapping in a bigger model — that erases the lesson.
- Add corpus questions without gold/absence labels.

---

## 12. Related projects (cross-links, same author tooling)

From-scratch series by the same author; shared tooling across all: uv / Makefile / ruff / ty / pytest.

- [retrieval_arena](https://github.com/Dima806/retrieval_arena) — benchmarks retrievers
  (dense / BM25 / hybrid) head to head; the retrieval half this project builds the pipeline around.
- [embeddings_101](https://github.com/Dima806/embeddings_101) — how embeddings work from
  scratch; the vectors retrieval runs on.
- [graphrag_vs_vectorrag](https://github.com/Dima806/graphrag_vs_vectorrag) — graph RAG vs
  vector RAG architectures; `rag_101` is the from-scratch vector-RAG foundation under both.
- [llm_judge_benchmark](https://github.com/Dima806/llm_judge_benchmark) — LLM-as-judge
  evaluation + the tiny-local-model-on-CPU (Ollama) setup reused here; source of the RAG-triad metrics.
- [lora_101](https://github.com/Dima806/lora_101) — fine-tuning with LoRA, the "when NOT to
  use RAG" fork (style/format vs. changing document facts). *Private for now; public soon.*
