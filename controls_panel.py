from PyQt6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLabel

class ControlsPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout()

        self.start_btn = QPushButton("Start Bot")
        self.stop_btn = QPushButton("Stop Bot")
        self.train_btn = QPushButton("Train RL Model")

        layout.addWidget(QLabel("Controls:"))
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.train_btn)

        self.setLayout(layout)
