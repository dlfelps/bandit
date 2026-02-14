"""Abstract base class for all bandit algorithms.

All bandit algorithms in this project must inherit from BanditAlgorithm
and implement the select_arm() and update() methods. This ensures that
algorithms are interchangeable within the simulation engine.

The interface supports both context-free (Multi-Armed Bandit) and
context-aware (Contextual Bandit) algorithms through the optional
context parameter.
"""

from abc import ABC, abstractmethod

import numpy as np


class BanditAlgorithm(ABC):
    """Abstract interface for bandit recommendation algorithms.

    All algorithms must implement two core methods:

    1. select_arm: Given a set of candidate arms (article IDs) and an
       optional context vector, choose which arm to pull (which article
       to recommend).

    2. update: After observing the reward (click=1 or no-click=0),
       update the algorithm's internal state to improve future
       selections.

    Attributes:
        name: Human-readable name of the algorithm (defaults to the
            class name).
    """

    @property
    def name(self) -> str:
        """Return the algorithm's class name as its identifier."""
        return self.__class__.__name__

    @abstractmethod
    def select_arm(
        self,
        arm_ids: list[str],
        context: np.ndarray | None = None,
    ) -> str:
        """Select which arm (article) to recommend.

        Args:
            arm_ids: List of candidate article IDs available for
                recommendation in this round.
            context: Optional feature vector representing the user
                and/or item context. Used by contextual algorithms
                like LinUCB; ignored by context-free algorithms.

        Returns:
            The ID of the selected article to recommend.
        """

    @abstractmethod
    def update(
        self,
        arm_id: str,
        reward: float,
        context: np.ndarray | None = None,
    ) -> None:
        """Update the algorithm's internal state after observing a reward.

        Args:
            arm_id: The ID of the article that was recommended.
            reward: The observed reward (1.0 for click, 0.0 for no
                click).
            context: Optional feature vector that was used when
                selecting this arm. Required by contextual algorithms
                to update their models.
        """
