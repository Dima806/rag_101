"""rag_101: retrieval-augmented generation from scratch, end to end, on CPU.

The package is organised as the four separable parts of a RAG system plus the
machinery to measure each one:

- ``src.pipeline`` -- chunk, embed, retrieve, generate, and the chain that ties them.
- ``src.grounding`` -- groundedness and abstention checks.
- ``src.datasets`` -- the frozen corpus and the three kinds of questions.
- ``src.evaluation`` -- retrieval metrics, answer metrics, the ceiling, and ablations.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
