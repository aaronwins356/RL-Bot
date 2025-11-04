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

### Use with RLBot GUI
1. Create and activate a virtual environment, then install runtime deps:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Launch the RLBot GUI (`python -m rlbot.gui` or start from the desktop shortcut).
3. In the GUI, choose **Open rlbot.cfg** and select the repository’s `rlbot.cfg` file.
4. Press **Start Match**. Blue spawns `WinYour1s`; Orange loads the mirrored configuration.
5. Tournament parameters (risk limits, kickoff variant, deterministic seed) live in
   `config/settings.yaml` under the `tournament` block. Toggle `enabled` or adjust thresholds to fit
   your event.
6. Loadout overrides for each team sit in `appearance/loadout_blue.cfg` and
   `appearance/loadout_orange.cfg`. Edit them to customize cosmetics.
7. If the bot does not appear, verify that Python is 64-bit, `rlbot` is installed inside the active
   environment, and `bot.cfg` points to `rlbot_agent/agent.py`.

### RLBot Entry Point
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
- `bot`: RLBot metadata such as the exported name and developer credit.
- `tournament`: Deterministic seed, kickoff variant preference, and style-risk limits.

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
2. Install editable deps for development: `pip install -e .`.
3. Capture a deterministic tick: `python cli.py`.
4. Practice mechanics with telemetry: `python -m scenarios.run --mechanic aerial --seed 0`.
5. Generate a performance report: `python reports/generate_report.py` and inspect `reports/last_run/`.
6. Install RLBot runtime deps: `pip install -r requirements.txt` (inside the active environment).
7. Launch RLBot GUI, open `rlbot.cfg`, and start a scrimmage.
8. Tune tournament safety in `config/settings.yaml` before competition.
