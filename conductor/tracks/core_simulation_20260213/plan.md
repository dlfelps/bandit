# Implementation Plan: Core Simulation & Baseline Bandits

## Phase 1: Environment & Foundational Interfaces
- [x] Task: Set up the Python environment using `uv` and install core dependencies (NumPy, Pandas, Pytest). `f60dba5`
- [ ] Task: Define the `BanditAlgorithm` abstract base class with `select_arm` and `update` methods.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Environment & Foundational Interfaces' (Protocol in workflow.md)

## Phase 2: Data Pipeline & Simulation Engine
- [ ] Task: Implement the `MINDDataLoader` to parse the MIND dataset impressions and news metadata.
- [ ] Task: Create the `SimulationEngine` to coordinate the flow between data and algorithms.
    - [ ] Write unit tests for the simulation loop logic.
    - [ ] Implement the core `run()` method.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Data Pipeline & Simulation Engine' (Protocol in workflow.md)

## Phase 3: Baseline Algorithm Implementation
- [ ] Task: Implement the `RandomChoice` algorithm.
    - [ ] Write unit tests for `RandomChoice`.
    - [ ] Implement `select_arm` logic.
- [ ] Task: Implement the `EpsilonGreedy` algorithm.
    - [ ] Write unit tests for `EpsilonGreedy`.
    - [ ] Implement exploration/exploitation logic.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Baseline Algorithm Implementation' (Protocol in workflow.md)

## Phase 4: Integration & Initial Comparison
- [ ] Task: Run a simulation with both `RandomChoice` and `EpsilonGreedy` on a data subset.
- [ ] Task: Implement basic metric logging and verify CSV output.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Integration & Initial Comparison' (Protocol in workflow.md)
