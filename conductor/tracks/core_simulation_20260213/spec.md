# Specification: Core Simulation & Baseline Bandits

## Overview
This track establishes the foundational infrastructure for the bandit recommendation system. It involves setting up the environment, creating a data pipeline for the MIND dataset, and implementing the first two comparison algorithms.

## Functional Requirements
- **MIND Data Pipeline:**
    - Load a subset of the MIND dataset (impressions and news metadata).
    - Parse impression logs into sequential rounds.
- **Simulation Engine:**
    - Implement a `Simulator` class that manages the recommendation loop.
    - Provide candidates to algorithms and record rewards based on actual user clicks.
- **Baseline Algorithms:**
    - Implement `RandomChoice` algorithm.
    - Implement `EpsilonGreedy` algorithm.
- **Metrics & Logging:**
    - Track cumulative clicks and impressions per algorithm.
    - Log results to a structured format (CSV).

## Technical Constraints
- All algorithms must inherit from a common `BanditAlgorithm` base class.
- Use `uv` for dependency management.
- Ensure compatibility with NumPy/Pandas for data handling.
