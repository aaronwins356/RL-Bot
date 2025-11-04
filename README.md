# RL-Bot-Pro: Production-Ready Rocket League Bot Foundation

This repository provides a robust foundation for developing Rocket League bots with pro-style mechanics, a flexible planner, and comprehensive tooling for development, testing, and training. It's designed for offline use, focusing on deterministic behavior and a clean architecture.

## Features

- **Core Bot Logic**: `rlbot_pro/` package with core types, mechanics, planner, and RLBot adapter.
- **Pro-Style Mechanics**: Implementations for Aerials, Air Dribbles, Ceiling Shots, Flip Resets, Double Taps, and Recoveries.
- **Flexible Planner**: Behavior tree / options selector for dynamic decision-making based on game state.
- **RLBot Adapter**: Seamless integration with the RLBot framework (optional installation).
- **Simulation Mode**: Deterministic simulation harness for testing mechanics and planning without RLBot.
- **Minimal GUI**: A simple Tkinter-based GUI for live tweaking of bot behavior (thread-safe).
- **Telemetry**: CSV logging of game data and JSON summaries of attempts and Key Performance Indicators (KPIs).
- **Training Hooks**: RLGym-style environment abstractions for offline reinforcement learning (import-protected).
- **Comprehensive Testing**: Unit tests for all core components, ensuring determinism and correctness.
- **Continuous Integration (CI)**: GitHub Actions workflow for linting, type checking, and testing on Ubuntu with Python 3.11.

## Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/aaronwins356/RL-Bot.git
    cd RL-Bot
    ```

2.  **Create a Python Virtual Environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -e ".[dev]"
    ```
    This installs all core dependencies and development tools (pytest, mypy, ruff, black).
    If you plan to use the bot with RLBot, the `rlbot` package will also be installed.

## Quickstarts

### Run Simulation Demo

To see a quick simulation of the bot's control output without needing RLBot:

```bash
python cli.py
```
This will print a `Controls(...)` line, demonstrating the bot's basic control output.

### Run GUI

To launch the minimal GUI for live behavior tweaking (without RLBot):

```bash
python run_gui.py
```
The GUI will open in a separate window, allowing you to adjust bot parameters.

### Run Scenarios and Generate Report

To run a specific mechanic scenario in simulation, generate telemetry, and produce a report:

```bash
python -m scenarios.run --mechanic aerial --seed 0
python reports/generate_report.py
```
This will:
1.  Run the `aerial` mechanic scenario with a fixed seed, generating `telemetry.csv` and `metrics.json` in the `runs/` directory.
2.  Process the generated telemetry, printing a markdown summary to the console and saving plots to `reports/last_run/`.

### RLBot Integration (Optional)

If you have RLBot installed and want to run this bot within the game:

1.  Ensure RLBot is installed (`pip install rlbot` is included in `pyproject.toml`).
2.  Configure RLBot to use `main.py` as the bot's entry point.
    If RLBot is not installed, `main.py` will print a friendly message and exit.

## Configuration Reference

All core settings are managed via `config/settings.yaml`.

```yaml
# config/settings.yaml
log_level: INFO
enable_gui: false
telemetry_csv: "runs/telemetry.csv"
aggression: 0.6
mechanics:
  aerial: true
  air_dribble: true
  ceiling: true
  flip_reset: true
  double_tap: true
planner:
  pressure_level: 0.5
  safety_bias: 0.4
```

-   `log_level`: Logging verbosity (e.g., `DEBUG`, `INFO`, `WARNING`, `ERROR`).
-   `enable_gui`: Set to `true` to enable the GUI when running `main.py` or `cli.py`.
-   `telemetry_csv`: Path to the CSV file where telemetry data will be logged.
-   `aggression`: A float between 0 and 1, influencing the bot's offensive tendencies.
-   `mechanics`: A dictionary of booleans to enable/disable specific mechanics.
-   `planner`: Parameters for the behavior tree/options selector.
    -   `pressure_level`: How aggressively the bot pursues offensive plays.
    -   `safety_bias`: How much the bot prioritizes defensive plays.

## Testing & CI

### Run Tests

To run all unit tests:

```bash
pytest -q
```

### Linting and Type Checking

To check code style and types:

```bash
ruff check .
black --check .
mypy .
```

### Continuous Integration

The `.github/workflows/ci.yml` workflow automatically runs `ruff`, `black --check`, `mypy`, and `pytest` on every push and pull request to ensure code quality and correctness.

## Safety Disclaimer

This bot foundation is designed for **offline use and development only**. It does not include any code for online play or interaction with external network services. Use responsibly.
