# RL Bot Pro

Deterministic Rocket League bot scaffold for experimentation without requiring the official RLBot framework.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Run the CLI

```bash
rlbot-pro
```

This command builds a dummy game state, runs the handcrafted agent, and prints the resulting controls.

## Testing

```bash
pytest
ruff check .
black --check rlbot_pro tests cli.py
mypy rlbot_pro tests cli.py
```

All tooling targets Python 3.10 through 3.12. Continuous integration executes linting, static type checking, and the test suite on Ubuntu.
