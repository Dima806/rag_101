"""Shared plotting helpers for the notebooks and the app.

Thin wrappers around matplotlib that return a ``Figure`` the caller can save or display.
A non-interactive backend is selected on import so the notebooks render under
``make notebooks`` (headless) without a display.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - must follow backend selection
from matplotlib.figure import Figure  # noqa: E402


def bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    ylabel: str = "",
    ylim: tuple[float, float] | None = None,
) -> Figure:
    """A simple labelled bar chart (e.g. answer accuracy per chunker)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(list(labels), list(values), color="#4C72B0")
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    for i, value in enumerate(values):
        ax.text(i, value, f"{value:.2f}", ha="center", va="bottom")
    fig.tight_layout()
    return fig


def ceiling_chart(
    oracle_accuracy: float,
    retrieved_accuracy: float,
    retrieval_recall: float,
) -> Figure:
    """Visualise the ceiling: oracle vs retrieved accuracy, with recall drawn as the cap."""
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ["oracle passage", "retrieved passage"]
    ax.bar(bars, [oracle_accuracy, retrieved_accuracy], color=["#55A868", "#C44E52"])
    ax.axhline(
        retrieval_recall,
        linestyle="--",
        color="#333333",
        label=f"retrieval recall = {retrieval_recall:.2f} (the cap)",
    )
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("answer accuracy")
    ax.set_title("Retrieval is the ceiling")
    ax.legend()
    fig.tight_layout()
    return fig


def grouped_lines(
    x: Sequence[float],
    series: dict[str, Sequence[float]],
    title: str,
    xlabel: str = "",
    ylabel: str = "",
) -> Figure:
    """Line chart with one line per series (e.g. recall and accuracy vs top-k)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, ys in series.items():
        ax.plot(list(x), list(ys), marker="o", label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    return fig
