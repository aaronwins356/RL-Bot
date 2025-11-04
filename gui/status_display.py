"""Widgets for presenting ProBot telemetry in the GUI."""
from __future__ import annotations

from typing import Iterable

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QTextEdit, QVBoxLayout, QWidget

from telemetry import BotTelemetry


class StatusDisplay(QWidget):
    """Simple read-only widget to display live telemetry."""

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.state_label = QLabel("Status: Offline")
        self.speed_label = QLabel("Speed: 0 uu/s")
        self.boost_label = QLabel("Boost: 100%")
        self.shot_label = QLabel("Shot Accuracy: 0%")
        self.mechanic_label = QLabel("Mechanic: Idle")

        self.ball_prediction_view = QTextEdit()
        self.ball_prediction_view.setReadOnly(True)
        self.ball_prediction_view.setMaximumHeight(120)

        layout.addWidget(self.state_label)
        layout.addWidget(self.speed_label)
        layout.addWidget(self.boost_label)
        layout.addWidget(self.shot_label)
        layout.addWidget(self.mechanic_label)
        layout.addWidget(QLabel("Ball Prediction (next 2s):"))
        layout.addWidget(self.ball_prediction_view)

        self.setLayout(layout)

    def update_status(self, status: str) -> None:
        self.state_label.setText(f"Status: {status}")

    def update_telemetry(self, telemetry: BotTelemetry) -> None:
        """Update all labels from telemetry data."""

        self.speed_label.setText(f"Speed: {telemetry.speed:.0f} uu/s")
        self.boost_label.setText(f"Boost: {telemetry.boost:.0f}%")
        self.shot_label.setText(f"Shot Accuracy: {telemetry.shot_accuracy * 100:.1f}%")
        self.mechanic_label.setText(f"Mechanic: {telemetry.mechanic}")
        self.ball_prediction_view.setPlainText(self._format_prediction(telemetry.ball_prediction))

    @staticmethod
    def _format_prediction(prediction: Iterable) -> str:
        preview = []
        for idx, slice_ in enumerate(prediction):
            if idx >= 5:
                break
            preview.append(
                f"t+{slice_.time:.2f}s -> ({slice_.position[0]:.0f}, {slice_.position[1]:.0f}, {slice_.position[2]:.0f})"
            )
        return "\n".join(preview)
