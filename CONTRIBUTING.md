# Contributing to Tidely

Thank you for your interest in contributing to Tidely! We welcome all contributions, including bug reports, feature requests, documentation improvements, and code submissions.

## Getting Started

1. **Fork the repository** and clone it locally.
2. **Install dependencies**: We use `pip` (or `poetry`) for managing dependencies. Install the development dependencies by running `pip install -r requirements-dev.txt` (or equivalent).
3. **Run the test suite**: Before making changes, ensure all tests pass by running `pytest tests/`.

## Development Workflow

1. **Create a branch**: `git checkout -b feature/my-awesome-feature`.
2. **Make your changes**: Ensure your code is clean, well-documented, and follows our style guidelines.
3. **Format and Lint**: Tidely enforces `black`, `ruff`, and `mypy`. Run these tools locally before committing.
4. **Write Tests**: If you are adding a new feature or fixing a bug, you *must* add a corresponding test.
5. **Submit a Pull Request**: Provide a clear and detailed description of your changes.

## Code Style

Tidely adheres to strictly typed Python.
- Always use type hints.
- Keep functions pure where possible.
- Ensure Docstrings follow the Google style guide.

Thank you for helping us build the best data cleaning engine in Python!
