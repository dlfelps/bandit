"""Comparative visualization of bandit algorithm performance.

Generates cumulative CTR line plots and final CTR bar charts from
simulation results.  Uses the Agg backend for headless compatibility.
"""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_cumulative_ctr(
    results: list[dict[str, Any]],
    output_dir: str | Path,
) -> Path:
    """Plot cumulative CTR over impression rounds for all algorithms.

    For each algorithm, computes cumulative_clicks / round_number at
    every step and draws a line chart.

    Args:
        results: List of result dicts from SimulationEngine.run(),
            each containing 'algorithm' and 'history' keys.
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved PNG file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for result in results:
        history = result["history"]
        cumulative_clicks = 0
        rounds = []
        ctrs = []
        for entry in history:
            cumulative_clicks += int(entry["reward"])
            round_num = entry["round"]
            rounds.append(round_num)
            ctrs.append(cumulative_clicks / round_num)
        ax.plot(rounds, ctrs, label=result["algorithm"])

    ax.set_xlabel("Impression Round")
    ax.set_ylabel("Cumulative CTR")
    ax.set_title("Cumulative Click-Through Rate by Algorithm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = output_dir / "cumulative_ctr.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_final_ctr_bar(
    results: list[dict[str, Any]],
    output_dir: str | Path,
) -> Path:
    """Plot a bar chart of final CTR for each algorithm.

    Args:
        results: List of result dicts from SimulationEngine.run(),
            each containing 'algorithm' and 'click_through_rate' keys.
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved PNG file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    names = [r["algorithm"] for r in results]
    ctrs = [r["click_through_rate"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, ctrs)

    for bar, ctr in zip(bars, ctrs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{ctr:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Click-Through Rate")
    ax.set_title("Final CTR Comparison")
    ax.grid(True, axis="y", alpha=0.3)

    path = output_dir / "final_ctr.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
