"""Main application window wiring controls to the bot manager."""
from __future__ import annotations

from PyQt6.QtWidgets import QMainWindow, QMessageBox, QVBoxLayout, QWidget

from bot_manager import BotManager
from controls_panel import ControlsPanel
from gui.status_display import StatusDisplay


class MainWindow(QMainWindow):
    """Primary window orchestrating the RLBot GUI."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RLBot Pro Controller")

        self.bot_manager = BotManager()
        self.controls_panel = ControlsPanel()
        self.status_display = StatusDisplay()

        central = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.controls_panel)
        layout.addWidget(self.status_display)
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.controls_panel.train_btn.setEnabled(False)
        self.controls_panel.start_btn.clicked.connect(self.bot_manager.start_bot)
        self.controls_panel.stop_btn.clicked.connect(self.bot_manager.stop_bot)

        self.bot_manager.telemetry_updated.connect(self.status_display.update_telemetry)
        self.bot_manager.status_changed.connect(self.status_display.update_status)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.bot_manager.is_running():
            if not self._confirm_shutdown():
                event.ignore()
                return
        self.bot_manager.shutdown()
        event.accept()

    def _confirm_shutdown(self) -> bool:
        message = QMessageBox(self)
        message.setWindowTitle("Stop Bot")
        message.setText("The bot is still running. Stop it before exiting?")
        message.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        message.setDefaultButton(QMessageBox.StandardButton.Yes)
        result = message.exec()
        if result == QMessageBox.StandardButton.Yes:
            self.bot_manager.stop_bot()
            return True
        return False
