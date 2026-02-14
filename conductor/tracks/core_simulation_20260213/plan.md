# Implementation Plan: Core Simulation & Baseline Bandits

## Phase 1: Environment & Foundational Interfaces
- [x] Task: Set up the Python environment using `uv` and install core dependencies (NumPy, Pandas, Pytest). `f60dba5`
- [x] Task: Define the `BanditAlgorithm` abstract base class with `select_arm` and `update` methods. `ae4286a`
- [x] Task: Conductor - User Manual Verification 'Phase 1: Environment & Foundational Interfaces' (Protocol in workflow.md) ✓

## Phase 2: Data Pipeline & Simulation Engine
- [x] Task: Implement the `MINDDataLoader` to parse the MIND dataset impressions and news metadata. `133f0ab`
- [x] Task: Create the `SimulationEngine` to coordinate the flow between data and algorithms. `7a3994c`
    - [x] Write unit tests for the simulation loop logic.
    - [x] Implement the core `run()` method.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Data Pipeline & Simulation Engine' (Protocol in workflow.md) ✓

## Phase 3: Baseline Algorithm Implementation [checkpoint: 5c0417a]
- [x] Task: Implement the `RandomChoice` algorithm. `8634531`
    - [x] Write unit tests for `RandomChoice`.
    - [x] Implement `select_arm` logic.
- [x] Task: Implement the `EpsilonGreedy` algorithm. `9d966b7`
    - [x] Write unit tests for `EpsilonGreedy`.
    - [x] Implement exploration/exploitation logic.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Baseline Algorithm Implementation' (Protocol in workflow.md) ✓

## Phase 4: Integration & Initial Comparison
- [x] Task: Run a simulation with both `RandomChoice` and `EpsilonGreedy` on a data subset. `7f15c88`
- [ ] Task: Implement basic metric logging and verify CSV output.
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Integration & Initial Comparison' (Protocol in workflow.md)
