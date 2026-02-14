"""Metric logging utilities for bandit simulation experiments.

Provides helpers to convert simulation results into structured
DataFrames and persist them as CSV files for downstream analysis.
"""

import csv
from pathlib import Path
from typing import Any

import pandas as pd


def results_to_dataframe(results: dict[str, Any]) -> pd.DataFrame:
    """Convert a simulation results dict to a tidy DataFrame.

    Each row represents one impression round.  A running cumulative
    CTR column is appended so that performance over time can be
    plotted directly.

    Args:
        results: The dict returned by ``SimulationEngine.run()``.

    Returns:
        DataFrame with columns: algorithm, round, user_id,
        selected_arm, reward, cumulative_ctr.
    """
    history = results["history"]
    algorithm_name = results["algorithm"]

    rows: list[dict[str, Any]] = []
    cumulative_clicks = 0
    for record in history:
        cumulative_clicks += int(record["reward"])
        rows.append(
            {
                "algorithm": algorithm_name,
                "round": record["round"],
                "user_id": record["user_id"],
                "selected_arm": record["selected_arm"],
                "reward": record["reward"],
                "cumulative_ctr": cumulative_clicks / record["round"],
            }
        )

    return pd.DataFrame(rows)


def save_results_csv(
    dataframes: list[pd.DataFrame],
    output_path: str | Path,
) -> Path:
    """Concatenate result DataFrames and write to a CSV file.

    Args:
        dataframes: List of DataFrames (one per algorithm) as
            returned by ``results_to_dataframe``.
        output_path: Destination file path for the CSV.

    Returns:
        The resolved Path of the written CSV file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined = pd.concat(dataframes, ignore_index=True)
    combined.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
    return output_path
