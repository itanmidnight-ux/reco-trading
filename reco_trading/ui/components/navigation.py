from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QPushButton, QToolButton, QVBoxLayout


class Sidebar(QFrame):
    tab_selected = Signal(int)

    def __init__(self, items: list[tuple[str, str]]) -> None:
        super().__init__()
        self.setObjectName("sidebar")
        self._buttons: list[QPushButton] = []
        self._expanded = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.toggle = QToolButton()
        self.toggle.setText("⟨")
        self.toggle.clicked.connect(self._toggle)
        layout.addWidget(self.toggle, alignment=Qt.AlignmentFlag.AlignLeft)

        for idx, (icon, label) in enumerate(items):
            btn = QPushButton(f"{icon}  {label}")
            btn.setCheckable(True)
            btn.clicked.connect(lambda _checked, i=idx: self.tab_selected.emit(i))
            btn.setObjectName("navButton")
            btn.setMinimumHeight(34)
            layout.addWidget(btn)
            self._buttons.append(btn)

        layout.addStretch(1)
        self.set_active(0)

    def set_active(self, index: int) -> None:
        for i, btn in enumerate(self._buttons):
            btn.setChecked(i == index)

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self.toggle.setText("⟨" if self._expanded else "⟩")
        self.setFixedWidth(190 if self._expanded else 64)
        for btn in self._buttons:
            text = btn.text()
            if self._expanded:
                if len(text.strip()) <= 2:
                    continue
            parts = text.split("  ")
            if len(parts) == 2:
                btn.setText(f"{parts[0]}  {parts[1]}" if self._expanded else parts[0])
