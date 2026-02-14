"""Simulation engine for running bandit recommendation experiments.

The SimulationEngine orchestrates the interaction between a bandit
algorithm and a stream of impression data. For each impression round
it:

1. Presents candidate article IDs to the algorithm.
2. Records which article the algorithm selects.
3. Looks up the actual reward (click or no-click) from the dataset.
4. Feeds the reward back to the algorithm via update().
5. Accumulates performance metrics (clicks, impressions, CTR).
"""

from typing import Any

from bandit.algorithms.base import BanditAlgorithm


class SimulationEngine:
    """Run a bandit algorithm over a sequence of impression rounds.

    The engine is deliberately decoupled from the data loader
    implementation — it only requires an object with an ``impressions``
    attribute (list of dicts) and iteration support.

    Each impression dict must contain:
        - ``user_id`` (str): Identifier for the user.
        - ``candidates`` (list[str]): Article IDs available this round.
        - ``rewards`` (dict[str, float]): Ground-truth click labels
          mapping each candidate to 0.0 or 1.0.

    Attributes:
        algorithm: The bandit algorithm being evaluated.
    """

    def __init__(
        self,
        algorithm: BanditAlgorithm,
        data_loader: Any,
    ) -> None:
        """Initialize the engine with an algorithm and data source.

        Args:
            algorithm: A concrete BanditAlgorithm instance to evaluate.
            data_loader: An object exposing ``impressions`` (list of
                dicts) and supporting ``__iter__`` / ``__len__``.
        """
        self._algorithm = algorithm
        self._data_loader = data_loader

    @property
    def algorithm(self) -> BanditAlgorithm:
        """Return the algorithm being evaluated."""
        return self._algorithm

    def run(self) -> dict[str, Any]:
        """Execute the full simulation loop over all impression rounds.

        For each round the algorithm selects an arm from the available
        candidates, receives the ground-truth reward, and updates its
        internal state.

        Returns:
            A dict containing:
                - algorithm (str): Name of the algorithm.
                - total_impressions (int): Number of rounds processed.
                - total_clicks (int): Number of rounds where the
                  selected arm was clicked.
                - click_through_rate (float): total_clicks /
                  total_impressions (0.0 if no impressions).
                - history (list[dict]): Per-round records with keys
                  round, user_id, selected_arm, reward.
        """
        total_clicks = 0
        total_impressions = 0
        history: list[dict[str, Any]] = []

        for impression in self._data_loader:
            candidates = impression["candidates"]
            rewards = impression["rewards"]
            user_id = impression["user_id"]

            selected_arm = self._algorithm.select_arm(candidates)
            reward = rewards[selected_arm]

            self._algorithm.update(selected_arm, reward)

            total_impressions += 1
            total_clicks += int(reward)

            history.append(
                {
                    "round": total_impressions,
                    "user_id": user_id,
                    "selected_arm": selected_arm,
                    "reward": reward,
                }
            )

        click_through_rate = (
            total_clicks / total_impressions if total_impressions > 0 else 0.0
        )

        return {
            "algorithm": self._algorithm.name,
            "total_impressions": total_impressions,
            "total_clicks": total_clicks,
            "click_through_rate": click_through_rate,
            "history": history,
        }
