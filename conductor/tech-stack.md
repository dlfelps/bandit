# Tech Stack

## Core Language & Runtime
- **Python 3.10+**: Leveraging modern features like type hinting and performance improvements.
- **uv**: Fast Python package installer and resolver for efficient dependency management.

## Data Processing & Mathematics
- **NumPy**: For efficient matrix operations required by LinUCB and Thompson Sampling.
- **Pandas**: For loading and manipulating the MIND dataset (impression logs, user history, metadata).
- **Scikit-learn**: For utilities like one-hot encoding and feature scaling.

## Visualization & Monitoring
- **Matplotlib**: For generating the final Cumulative CTR performance plots.
- **Seaborn**: For statistical data visualization and improved plot aesthetics.
- **tqdm**: For displaying real-time progress bars during long-running dataset simulations.

## Development & Testing
- **Pytest**: For unit testing the bandit algorithm implementations and simulation logic.
- **Black/Ruff**: For automated code formatting and linting to ensure a clean codebase.
