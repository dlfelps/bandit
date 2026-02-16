"""Run a head-to-head comparison of bandit algorithms on MIND data.

Runs RandomChoice, EpsilonGreedy, ThompsonSampling, and LinUCB against
the MINDlarge_train dataset (first 100K impressions), prints summary
metrics to the console, saves results to CSV files, and generates
comparative visualizations.
"""

from pathlib import Path

from bandit.algorithms.epsilon_greedy import EpsilonGreedy
from bandit.algorithms.linucb import LinUCB
from bandit.algorithms.random_choice import RandomChoice
from bandit.algorithms.thompson_sampling import ThompsonSampling
from bandit.data.loader import MINDDataLoader
from bandit.metrics.csv_logger import save_results
from bandit.metrics.plotting import plot_cumulative_ctr, plot_final_ctr_bar
from bandit.simulation.comparison import compare_algorithms


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data" / "MINDlarge_train"
_MAX_IMPRESSIONS = 100_000
_OUTPUT_DIR = _PROJECT_ROOT / "results"


def main() -> None:
    """Load data, run both algorithms, print and save results."""
    print(f"Loading MIND dataset from: {_DATA_DIR}")
    loader = MINDDataLoader(_DATA_DIR, max_impressions=_MAX_IMPRESSIONS)
    print(f"Loaded {len(loader)} impression rounds.\n")

    algorithms = [
        RandomChoice(seed=42),
        EpsilonGreedy(epsilon=0.1, seed=42),
        ThompsonSampling(prior_alpha=1.0, prior_beta=8.0, seed=42),
        LinUCB(alpha=1.0, seed=42),
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

    ctr_path = plot_cumulative_ctr(all_results, _OUTPUT_DIR)
    bar_path = plot_final_ctr_bar(all_results, _OUTPUT_DIR)
    print(f"\nResults saved to: {_OUTPUT_DIR}")
    print(f"  Cumulative CTR plot: {ctr_path}")
    print(f"  Final CTR bar chart: {bar_path}")


if __name__ == "__main__":
    main()
