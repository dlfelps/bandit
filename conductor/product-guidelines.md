# Product Guidelines

## Development Philosophy
- **Educational Clarity:** The primary goal is to teach. Code should be readable, well-commented, and use descriptive naming conventions that relate back to the underlying mathematical concepts.
- **Architectural Consistency:** Use a strict interface-driven approach for algorithms to ensure they are interchangeable and comparable within the simulation engine.
- **Robustness & Reproducibility:** While the MIND dataset is large, the implementation should focus on predictable, debuggable pipelines through strict typing and detailed logging.

## Code Style & Documentation
- **Descriptive Naming:** Favor clarity over brevity. (e.g., `expected_reward_upper_bound` instead of `ucb`).
- **Inline Explanations:** Comments should explain the mathematical logic of the bandit algorithms, especially for complex operations like matrix inversions in LinUCB or posterior updates in Thompson Sampling.
- **Type Safety:** Use Python's `typing` module (Type Hints) for all function signatures and complex data structures (context vectors, reward distributions).

## Architecture & Design
- **Interface-First:** All bandit algorithms must inherit from a common `BanditAlgorithm` base class/interface, implementing standard methods like `select_arm(context)` and `update(arm, reward, context)`.
- **Decoupled Simulation:** Keep the simulation logic (MIND data processing) strictly separated from the algorithm logic. The simulator should only "know" about the `BanditAlgorithm` interface.
- **Modular Utilities:** Common mathematical utilities (e.g., vector normalization, one-hot encoding) should be centralized for reuse across different algorithms.

## Verification & Monitoring
- **Pipeline Logging:** Record simulation events (round number, selected article ID, context features, reward received) to allow for detailed post-mortem analysis of algorithm behavior.
- **Progress Tracking:** Provide clear visual or console-based feedback during long-running simulations to monitor the accumulation of clicks and impressions.
