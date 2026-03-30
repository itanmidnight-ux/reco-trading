from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class CacheTab(QWidget):
    def __init__(self, state_manager=None) -> None:
        super().__init__()
        self.state_manager = state_manager
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        header = QLabel("<h2>📊 Cache & Monitoring</h2>")
        layout.addWidget(header)

        cache_group = QGroupBox("Estado del Cache")
        cache_layout = QFormLayout()

        self.cache_enabled_label = QLabel("Activado")
        self.cache_size_label = QLabel("0")
        self.cache_hits_label = QLabel("0")
        self.cache_misses_label = QLabel("0")
        self.cache_hit_rate_label = QLabel("0%")
        self.cache_evictions_label = QLabel("0")

        cache_layout.addRow("Cache Enabled:", self.cache_enabled_label)
        cache_layout.addRow("Size:", self.cache_size_label)
        cache_layout.addRow("Hits:", self.cache_hits_label)
        cache_layout.addRow("Misses:", self.cache_misses_label)
        cache_layout.addRow("Hit Rate:", self.cache_hit_rate_label)
        cache_layout.addRow("Evictions:", self.cache_evictions_label)
        cache_group.setLayout(cache_layout)
        layout.addWidget(cache_group)

        ohlcv_group = QGroupBox("OHLCV Cache")
        ohlcv_layout = QFormLayout()

        self.ohlcv_ttl_label = QLabel("30s")
        self.ohlcv_symbols_label = QLabel("0")

        ohlcv_layout.addRow("TTL:", self.ohlcv_ttl_label)
        ohlcv_layout.addRow("Cached Symbols:", self.ohlcv_symbols_label)
        ohlcv_group.setLayout(ohlcv_layout)
        layout.addWidget(ohlcv_group)

        prefetch_group = QGroupBox("Pre-fetching")
        prefetch_layout = QFormLayout()

        self.prefetch_enabled_label = QLabel("Activado")
        self.prefetch_symbols_label = QLabel("0")
        self.prefetch_interval_label = QLabel("25s")

        prefetch_layout.addRow("Enabled:", self.prefetch_enabled_label)
        prefetch_layout.addRow("Symbols:", self.prefetch_symbols_label)
        prefetch_layout.addRow("Interval:", self.prefetch_interval_label)
        prefetch_group.setLayout(prefetch_layout)
        layout.addWidget(prefetch_group)

        data_group = QGroupBox("Data Providers")
        data_layout = QVBoxLayout()

        self.data_table = QTableWidget()
        self.data_table.setColumnCount(4)
        self.data_table.setHorizontalHeaderLabels(["Provider", "Status", "Last Update", "Errors"])
        self.data_table.setMaximumHeight(150)
        data_layout.addWidget(self.data_table)
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)

        buttons_layout = QHBoxLayout()
        self.clear_cache_button = QPushButton("Limpiar Cache")
        self.refresh_button = QPushButton("Refrescar")
        buttons_layout.addWidget(self.clear_cache_button)
        buttons_layout.addWidget(self.refresh_button)
        layout.addLayout(buttons_layout)

        layout.addStretch()

    def update_state(self, state: dict) -> None:
        cache_data = state.get("cache", {})

        if cache_data:
            self.cache_enabled_label.setText("Enabled" if cache_data.get("enabled", True) else "Disabled")
            self.cache_size_label.setText(str(cache_data.get("size", 0)))
            self.cache_hits_label.setText(str(cache_data.get("hits", 0)))
            self.cache_misses_label.setText(str(cache_data.get("misses", 0)))
            self.cache_hit_rate_label.setText(f"{cache_data.get('hit_rate', 0):.1%}")
            self.cache_evictions_label.setText(str(cache_data.get("evictions", 0)))

            ohlcv = cache_data.get("ohlcv", {})
            self.ohlcv_ttl_label.setText(f"{ohlcv.get('ttl', 30)}s")
            self.ohlcv_symbols_label.setText(str(ohlcv.get("symbols", 0)))

            prefetch = cache_data.get("prefetch", {})
            self.prefetch_enabled_label.setText("Enabled" if prefetch.get("enabled", True) else "Disabled")
            self.prefetch_symbols_label.setText(str(prefetch.get("symbols", 0)))
            self.prefetch_interval_label.setText(f"{prefetch.get('interval', 25)}s")

            providers = cache_data.get("providers", [])
            self.data_table.setRowCount(len(providers))
            for i, provider in enumerate(providers):
                self.data_table.setItem(i, 0, QTableWidgetItem(provider.get("name", "")))
                self.data_table.setItem(i, 1, QTableWidgetItem(provider.get("status", "")))
                self.data_table.setItem(i, 2, QTableWidgetItem(provider.get("last_update", "")))
                self.data_table.setItem(i, 3, QTableWidgetItem(str(provider.get("errors", 0))))
