# Bandit Sim

Educational reference implementation comparing Contextual Bandits vs Multi-Armed Bandits using the [Microsoft News Dataset (MIND)](https://msnews.github.io/).

## Overview

Bandit Sim demonstrates how different bandit algorithms balance the explore-exploit tradeoff when selecting news articles to recommend to users. It implements four algorithms on a real-world news recommendation task and provides tooling to compare their performance side by side.

### Algorithms

| Algorithm | Type | Description |
|---|---|---|
| **RandomChoice** | Baseline | Selects articles uniformly at random (no learning) |
| **EpsilonGreedy** | Multi-Armed Bandit | Exploits the best-known article most of the time, explores randomly with probability epsilon |
| **ThompsonSampling** | Multi-Armed Bandit | Uses Bayesian inference (Beta-Bernoulli model) to balance exploration and exploitation |
| **LinUCB** | Contextual Bandit | Builds per-article linear models over 62-dimensional user-article context vectors |

## Project Structure

```
bandit/
├── src/bandit/
│   ├── algorithms/          # Bandit algorithm implementations
│   │   ├── base.py          # Abstract BanditAlgorithm interface
│   │   ├── random_choice.py
│   │   ├── epsilon_greedy.py
│   │   ├── thompson_sampling.py
│   │   └── linucb.py
│   ├── data/                # Dataset loading and parsing
│   ├── simulation/          # Simulation engine and comparison runner
│   ├── metrics/             # Plotting and CSV export
│   └── utils/
├── tests/                   # Unit tests (pytest)
├── scripts/                 # Executable entry points
│   ├── run_comparison.py
│   └── generate_sample_data.py
├── data/                    # MIND dataset files
├── docs/                    # Detailed methodology writeup
└── pyproject.toml
```

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

## Usage

### Run the full algorithm comparison

```bash
python scripts/run_comparison.py
```

This loads the MIND dataset, runs all four algorithms against the same sequence of impressions, prints a CTR summary, saves results to `results/`, and generates comparison charts.

### Generate sample data for development

```bash
python scripts/generate_sample_data.py
```

Creates a small synthetic MIND-format dataset in `data/MINDsmall_dev/` for quick iteration without the full dataset.

## Development

### Running tests

```bash
pytest tests/
```

With coverage (target: 80%):

```bash
pytest tests/ --cov=bandit --cov-report=html
```

### Linting

```bash
ruff check src/ tests/
```

## How It Works

The simulation engine processes MIND impressions sequentially. For each impression, every algorithm selects one article from the candidate set, and the engine records whether the user clicked on it. Over thousands of impressions, the algorithms that learn effectively converge on higher click-through rates (CTR).

LinUCB differs from the multi-armed bandit approaches by incorporating **context** — a 62-dimensional feature vector combining user subcategory preferences with article subcategory encodings — allowing it to personalize recommendations per user rather than learning a single global article ranking.

For a full explanation of the algorithms, feature engineering, and experimental results, see [`docs/blog_post.md`](docs/blog_post.md).
