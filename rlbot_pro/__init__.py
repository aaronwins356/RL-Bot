import logging.config
import os
from pathlib import Path

import yaml

# Ensure the 'runs' directory exists for logs and telemetry
Path("runs").mkdir(exist_ok=True)

# Load logging configuration
_log_config_path = Path(__file__).parent.parent / "config" / "logging.yaml"
if _log_config_path.exists():
    with open(_log_config_path) as f:
        _log_config = yaml.safe_load(f)
    logging.config.dictConfig(_log_config)
else:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("rlbot_pro")

# Load settings
_settings_config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
SETTINGS: dict = {}
if _settings_config_path.exists():
    with open(_settings_config_path) as f:
        SETTINGS = yaml.safe_load(f)
else:
    logger.warning("config/settings.yaml not found. Using default empty settings.")

# Define a global variable for the GUI queue, initialized to None
# This will be set by run_gui.py if the GUI is enabled
GUI_QUEUE = None
