"""Run a head-to-head comparison of bandit algorithms on MIND data.

Runs both RandomChoice and EpsilonGreedy against the MINDsmall_dev
dataset, prints summary metrics to the console, and saves results
to CSV files in the output directory.
"""

from pathlib import Path

from bandit.algorithms.epsilon_greedy import EpsilonGreedy
from bandit.algorithms.random_choice import RandomChoice
from bandit.data.loader import MINDDataLoader
from bandit.metrics.csv_logger import save_results
from bandit.simulation.comparison import compare_algorithms


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data" / "MINDsmall_dev"
_OUTPUT_DIR = _PROJECT_ROOT / "results"


def main() -> None:
    """Load data, run both algorithms, print and save results."""
    print(f"Loading MIND dataset from: {_DATA_DIR}")
    loader = MINDDataLoader(_DATA_DIR)
    print(f"Loaded {len(loader)} impression rounds.\n")

    algorithms = [
        RandomChoice(seed=42),
        EpsilonGreedy(epsilon=0.1, seed=42),
    ]

    print("Running comparison...")
    all_results = compare_algorithms(algorithms, loader)

    print("\n=== Results ===")
    for result in all_results:
        print(
            f"  {result['algorithm']:20s}  "
            f"CTR={result['click_through_rate']:.4f}  "
            f"({result['total_clicks']}"
            f"/{result['total_impressions']})"
        )

    save_results(all_results, _OUTPUT_DIR)
    print(f"\nResults saved to: {_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
