"""CSV logging for bandit simulation results.

Provides save_results() to persist per-round history and aggregate
summary metrics to CSV files.  Uses Pandas for structured I/O,
making the output easy to load for downstream analysis and
visualization.
"""

from pathlib import Path
from typing import Any

import pandas as pd


def save_results(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save simulation results to CSV files.

    Creates two types of CSV files in *output_dir*:

    1. Per-algorithm history: ``history_{algorithm_name}.csv``
       with columns: round, user_id, selected_arm, reward.
    2. Aggregate summary: ``summary.csv`` with columns:
       algorithm, total_impressions, total_clicks,
       click_through_rate.

    Args:
        results: List of result dicts from
            ``SimulationEngine.run()`` or ``compare_algorithms()``.
            Each dict must have keys: algorithm,
            total_impressions, total_clicks, click_through_rate,
            history.
        output_dir: Directory where CSV files will be written.
            Created if it does not exist.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write per-algorithm history CSVs
    for result in results:
        algorithm_name = result["algorithm"]
        history_df = pd.DataFrame(result["history"])
        history_path = output_dir / f"history_{algorithm_name}.csv"
        history_df.to_csv(history_path, index=False)

    # Write aggregate summary CSV
    summary_rows = [
        {
            "algorithm": r["algorithm"],
            "total_impressions": r["total_impressions"],
            "total_clicks": r["total_clicks"],
            "click_through_rate": r["click_through_rate"],
        }
        for r in results
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
