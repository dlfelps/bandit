"""Run multiple bandit algorithms and compare their performance.

Provides a compare_algorithms() function that runs each algorithm
through the SimulationEngine against the same dataset and returns
a list of result dictionaries for side-by-side comparison.
"""

from typing import Any

from bandit.algorithms.base import BanditAlgorithm
from bandit.simulation.engine import SimulationEngine


def compare_algorithms(
    algorithms: list[BanditAlgorithm],
    data_loader: Any,
) -> list[dict[str, Any]]:
    """Run each algorithm against the same dataset and collect results.

    Each algorithm gets its own SimulationEngine instance but is
    evaluated against the same data_loader.  This ensures a fair
    comparison on identical impression sequences.

    Note:
        The data_loader must support repeated iteration (its
        ``__iter__`` should return a fresh iterator each time).
        ``MINDDataLoader`` satisfies this requirement.

    Args:
        algorithms: List of BanditAlgorithm instances to evaluate.
        data_loader: A data source supporting iteration and
            ``impressions``.

    Returns:
        A list of result dicts (one per algorithm), each containing:
        algorithm, total_impressions, total_clicks,
        click_through_rate, history.
    """
    results: list[dict[str, Any]] = []
    for algorithm in algorithms:
        engine = SimulationEngine(
            algorithm=algorithm,
            data_loader=data_loader,
        )
        result = engine.run()
        results.append(result)
    return results
