# RLBot Pro Foundation ![CI](https://github.com/aaronwins356/RL-Bot/actions/workflows/ci.yml/badge.svg)

RLBot Pro is a production-grade Rocket League bot foundation showcasing deterministic mechanics,
planner integration, rich telemetry, and developer tooling. It is designed as a launchpad for
pro-level behaviors while remaining approachable for experimentation and research.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

### Run a single simulation tick
```bash
python cli.py
```

### Execute a focused scenario with telemetry output
```bash
python -m scenarios.run --mechanic aerial --seed 0
```

### Generate a telemetry report with plots
```bash
python reports/generate_report.py
```

### Launch the control panel GUI
```bash
python run_gui.py
```

### RLBot Integration
Running `python main.py` attempts to start the RLBot interface. If RLBot is not installed, a clear
message is printed and the process exits without error. Install `rlbot` once you are ready to run
in-game.

## Configuration
Configuration lives in `config/settings.yaml`. Key options include:

- `log_level`: Logging verbosity across modules.
- `enable_gui`: Launch the control panel GUI automatically.
- `telemetry_csv`: Output path for telemetry capture.
- `aggression`: Planner tuning between safety and offensive risk.
- `mechanics`: Toggle availability of each mechanic.
- `planner`: Parameters used to weigh safety versus aggression in the selector.

Logging configuration is stored separately in `config/logging.yaml` and adheres to Python logging
syntax.

## Telemetry & Reporting
Telemetry rows are captured via `telemetry/telemetry.py` and contain per-tick context such as
ball/car kinematics, selected option, and outcome flags. Metrics are derived with
`telemetry/metrics.py`, feeding `reports/generate_report.py` to create Markdown summaries and
matplotlib plots under `reports/last_run/`.

## Testing and Continuous Integration
Continuous integration is provided by `.github/workflows/ci.yml`. The pipeline installs
requirements, enforces linting (ruff, black), performs type checking (mypy), and runs pytest.

## Runbook
1. Create a virtual environment: `python -m venv .venv` then activate it.
2. Install dependencies: `pip install -e .`
3. Run a deterministic tick: `python cli.py`.
4. Execute a scenario and capture telemetry: `python -m scenarios.run --mechanic aerial --seed 0`.
5. Generate the latest report: `python reports/generate_report.py` and inspect `reports/last_run/`.
6. Enable the GUI by setting `enable_gui: true` in `config/settings.yaml` then running `python run_gui.py`.
7. To integrate RLBot later, install `rlbot` and run `python main.py`.
