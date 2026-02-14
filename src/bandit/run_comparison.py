"""Run a head-to-head comparison of bandit algorithms on MIND data.

Loads the MIND small-dev dataset, runs RandomChoice and EpsilonGreedy
through the SimulationEngine, logs per-round metrics, and writes a
combined CSV of results.

Usage:
    python -m bandit.run_comparison [--data-dir PATH] [--output PATH]
"""

import argparse
from pathlib import Path

from bandit.algorithms.epsilon_greedy import EpsilonGreedy
from bandit.algorithms.random_choice import RandomChoice
from bandit.data.loader import MINDDataLoader
from bandit.simulation.engine import SimulationEngine
from bandit.utils.metrics import results_to_dataframe, save_results_csv

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data" / "MINDsmall_dev"
_DEFAULT_OUTPUT = Path("output") / "comparison_results.csv"


def main() -> None:
    """Parse arguments, run simulations, and save results."""
    parser = argparse.ArgumentParser(
        description="Compare bandit algorithms on the MIND dataset.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help="Path to the MINDsmall_dev directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="Output CSV path for combined results.",
    )
    args = parser.parse_args()

    print(f"Loading MIND data from {args.data_dir} ...")
    loader = MINDDataLoader(args.data_dir)
    print(f"  {len(loader)} impression rounds loaded.\n")

    algorithms = [
        RandomChoice(seed=42),
        EpsilonGreedy(epsilon=0.1, seed=42),
    ]

    dataframes = []
    for algo in algorithms:
        print(f"Running {algo.name} ...")
        engine = SimulationEngine(algorithm=algo, data_loader=loader)
        results = engine.run()

        print(
            f"  Impressions: {results['total_impressions']}  "
            f"Clicks: {results['total_clicks']}  "
            f"CTR: {results['click_through_rate']:.4f}\n"
        )

        dataframes.append(results_to_dataframe(results))

    csv_path = save_results_csv(dataframes, args.output)
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
