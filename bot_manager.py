"""Management layer coordinating the ProBot lifecycle and GUI communication."""
from __future__ import annotations

import threading
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

from core.pro_bot import ProBot
from telemetry import BotTelemetry


class BotManager(QObject):
    """Launches, stops, and monitors the ProBot instance."""

    telemetry_updated = pyqtSignal(BotTelemetry)
    status_changed = pyqtSignal(str)

    def __init__(self) -> None:
        super().__init__()
        self._bot: Optional[ProBot] = None
        self._lock = threading.Lock()

    def start_bot(self) -> None:
        """Start the bot thread if it is not already running."""

        with self._lock:
            if self._bot is None:
                self._bot = ProBot(telemetry_callback=self.telemetry_updated.emit)
                self._bot.start()
            self._bot.start_running()
        self.status_changed.emit("Running")

    def stop_bot(self) -> None:
        """Signal the bot thread to pause execution."""

        with self._lock:
            if self._bot is not None:
                self._bot.stop_running()
        self.status_changed.emit("Stopped")

    def shutdown(self) -> None:
        """Terminate the bot thread completely."""

        with self._lock:
            if self._bot is not None:
                self._bot.shutdown()
                self._bot = None
        self.status_changed.emit("Offline")

    def is_running(self) -> bool:
        with self._lock:
            return self._bot is not None and self._bot.is_running
