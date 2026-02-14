"""Tests for the BanditAlgorithm abstract base class.

Verifies that the interface contract is correctly defined and that
concrete subclasses must implement all required abstract methods.
"""

import numpy as np
import pytest

from bandit.algorithms.base import BanditAlgorithm


class TestBanditAlgorithmInterface:
    """Test that BanditAlgorithm defines the correct abstract interface."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """BanditAlgorithm should not be directly instantiable."""
        with pytest.raises(TypeError):
            BanditAlgorithm()  # type: ignore[abstract]

    def test_subclass_missing_select_arm_raises(self) -> None:
        """A subclass that doesn't implement select_arm should raise."""

        class IncompleteAlgorithm(BanditAlgorithm):
            def update(
                self,
                arm_id: str,
                reward: float,
                context: np.ndarray | None = None,
            ) -> None:
                pass

        with pytest.raises(TypeError):
            IncompleteAlgorithm()  # type: ignore[abstract]

    def test_subclass_missing_update_raises(self) -> None:
        """A subclass that doesn't implement update should raise."""

        class IncompleteAlgorithm(BanditAlgorithm):
            def select_arm(
                self,
                arm_ids: list[str],
                context: np.ndarray | None = None,
            ) -> str:
                return arm_ids[0]

        with pytest.raises(TypeError):
            IncompleteAlgorithm()  # type: ignore[abstract]

    def test_concrete_subclass_can_be_instantiated(self) -> None:
        """A subclass implementing all methods should instantiate fine."""

        class ConcreteAlgorithm(BanditAlgorithm):
            def select_arm(
                self,
                arm_ids: list[str],
                context: np.ndarray | None = None,
            ) -> str:
                return arm_ids[0]

            def update(
                self,
                arm_id: str,
                reward: float,
                context: np.ndarray | None = None,
            ) -> None:
                pass

        algo = ConcreteAlgorithm()
        assert algo is not None

    def test_select_arm_returns_valid_arm_id(self) -> None:
        """select_arm should return one of the provided arm IDs."""

        class ConcreteAlgorithm(BanditAlgorithm):
            def select_arm(
                self,
                arm_ids: list[str],
                context: np.ndarray | None = None,
            ) -> str:
                return arm_ids[0]

            def update(
                self,
                arm_id: str,
                reward: float,
                context: np.ndarray | None = None,
            ) -> None:
                pass

        algo = ConcreteAlgorithm()
        candidates = ["N001", "N002", "N003"]
        selected = algo.select_arm(candidates)
        assert selected in candidates

    def test_select_arm_accepts_context_vector(self) -> None:
        """select_arm should accept an optional NumPy context vector."""

        class ConcreteAlgorithm(BanditAlgorithm):
            def select_arm(
                self,
                arm_ids: list[str],
                context: np.ndarray | None = None,
            ) -> str:
                return arm_ids[0]

            def update(
                self,
                arm_id: str,
                reward: float,
                context: np.ndarray | None = None,
            ) -> None:
                pass

        algo = ConcreteAlgorithm()
        context = np.array([1.0, 0.5, 0.3])
        selected = algo.select_arm(["N001", "N002"], context=context)
        assert selected in ["N001", "N002"]

    def test_update_accepts_reward_and_context(self) -> None:
        """update should accept arm_id, reward, and optional context."""

        class ConcreteAlgorithm(BanditAlgorithm):
            def __init__(self) -> None:
                self.last_update: tuple | None = None

            def select_arm(
                self,
                arm_ids: list[str],
                context: np.ndarray | None = None,
            ) -> str:
                return arm_ids[0]

            def update(
                self,
                arm_id: str,
                reward: float,
                context: np.ndarray | None = None,
            ) -> None:
                self.last_update = (arm_id, reward, context)

        algo = ConcreteAlgorithm()
        context = np.array([1.0, 0.5])
        algo.update("N001", 1.0, context=context)
        assert algo.last_update is not None
        assert algo.last_update[0] == "N001"
        assert algo.last_update[1] == 1.0
        assert np.array_equal(algo.last_update[2], context)

    def test_name_property_returns_class_name(self) -> None:
        """The name property should return the algorithm's class name."""

        class MyCustomAlgorithm(BanditAlgorithm):
            def select_arm(
                self,
                arm_ids: list[str],
                context: np.ndarray | None = None,
            ) -> str:
                return arm_ids[0]

            def update(
                self,
                arm_id: str,
                reward: float,
                context: np.ndarray | None = None,
            ) -> None:
                pass

        algo = MyCustomAlgorithm()
        assert algo.name == "MyCustomAlgorithm"
