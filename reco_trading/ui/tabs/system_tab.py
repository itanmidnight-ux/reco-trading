from __future__ import annotations

from PySide6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget


class SystemTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        title = QLabel("System")
        title.setObjectName("sectionTitle")
        root.addWidget(title)

        self.cpu = QLabel("CPU usage: --")
        self.memory = QLabel("Memory usage: --")
        self.db = QLabel("Database: --")
        self.api = QLabel("Binance API: --")
        for w in [self.cpu, self.memory, self.db, self.api]:
            root.addWidget(w)

        self.cpu_gauge = QProgressBar()
        self.cpu_gauge.setRange(0, 100)
        self.mem_gauge = QProgressBar()
        self.mem_gauge.setRange(0, 100)
        root.addWidget(self.cpu_gauge)
        root.addWidget(self.mem_gauge)

    def update_state(self, state: dict) -> None:
        system = state.get("system", {}) if isinstance(state.get("system"), dict) else {}
        cpu = _to_float(system.get("cpu_usage", state.get("cpu_usage", 0)))
        mem = _to_float(system.get("memory_percent", state.get("memory_percent", 0)))
        self.cpu.setText(f"CPU usage: {cpu:.1f}%")
        self.memory.setText(f"Memory usage: {mem:.1f}%")
        self.db.setText(f"Database: {system.get('database_status', 'UNKNOWN')}")
        self.api.setText(f"Binance API: {system.get('exchange_status', 'UNKNOWN')}")
        self.cpu_gauge.setValue(int(max(0, min(100, cpu))))
        self.mem_gauge.setValue(int(max(0, min(100, mem))))


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
