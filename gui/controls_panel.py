from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

CONFIG_QUEUE: queue.Queue[Mapping[str, Any]] = queue.Queue()


class ControlsPanel:
    def __init__(self, root: tk.Tk, config: Mapping[str, Any], config_path: Path):
        self.root = root
        self.config_path = config_path
        self.config = dict(config)
        self.root.title("RLBot Pro Control Panel")
        self._build()
        self._schedule_poll()

    def _build(self) -> None:
        aggression = tk.DoubleVar(value=float(self.config.get("aggression", 0.5)))
        tk.Label(self.root, text="Aggression").pack(fill=tk.X, padx=8, pady=4)
        slider = tk.Scale(
            self.root,
            from_=0.0,
            to=1.0,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=aggression,
        )
        slider.pack(fill=tk.X, padx=8)
        self.aggression_var = aggression

        mech_frame = tk.LabelFrame(self.root, text="Mechanics")
        mech_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.mechanic_vars: dict[str, tk.BooleanVar] = {}
        mechanics = self.config.get("mechanics", {})
        for key in ("aerial", "air_dribble", "ceiling", "flip_reset", "double_tap", "recoveries"):
            var = tk.BooleanVar(value=bool(mechanics.get(key, True)))
            cb = tk.Checkbutton(
                mech_frame,
                text=key.replace("_", " ").title(),
                variable=var,
                command=self._persist,
            )
            cb.pack(anchor=tk.W)
            self.mechanic_vars[key] = var

        save_btn = tk.Button(self.root, text="Save", command=self._persist)
        save_btn.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=8)

    def _schedule_poll(self) -> None:
        self.root.after(250, self._poll_updates)

    def _poll_updates(self) -> None:
        try:
            while True:
                update = CONFIG_QUEUE.get_nowait()
                self.config.update(update)
        except queue.Empty:
            pass
        self.root.after(250, self._poll_updates)

    def _persist(self) -> None:
        data = dict(self.config)
        data["aggression"] = float(self.aggression_var.get())
        mechanics = {key: bool(var.get()) for key, var in self.mechanic_vars.items()}
        data["mechanics"] = mechanics
        self.config = data
        self.config_path.write_text(yaml.safe_dump(data))


def launch_controls_panel(config_path: str | Path = "config/settings.yaml") -> None:
    path = Path(config_path)
    if not path.exists():
        message = f"Configuration file not found: {path}"
        raise FileNotFoundError(message)
    config_raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(config_raw, Mapping):
        message = "Configuration file must contain a mapping"
        raise TypeError(message)
    config: Mapping[str, Any] = dict(config_raw)
    root = tk.Tk()
    ControlsPanel(root, config, path)
    threading.Thread(target=root.mainloop, daemon=True).start()


__all__ = ["launch_controls_panel", "CONFIG_QUEUE", "ControlsPanel"]
