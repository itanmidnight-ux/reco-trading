from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
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


class HealthTab(QWidget):
    def __init__(self, state_manager=None) -> None:
        super().__init__()
        self.state_manager = state_manager
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        header = QLabel("<h2>💚 Health Checks</h2>")
        layout.addWidget(header)

        summary_group = QGroupBox("Resumen de Salud")
        summary_layout = QFormLayout()

        self.overall_status_label = QLabel("OK")
        self.total_checks_label = QLabel("0")
        self.healthy_checks_label = QLabel("0")
        self.unhealthy_checks_label = QLabel("0")
        self.last_check_label = QLabel("-")

        summary_layout.addRow("Estado General:", self.overall_status_label)
        summary_layout.addRow("Total Checks:", self.total_checks_label)
        summary_layout.addRow("Healthy:", self.healthy_checks_label)
        summary_layout.addRow("Unhealthy:", self.unhealthy_checks_label)
        summary_layout.addRow("Último Check:", self.last_check_label)
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)

        components_group = QGroupBox("Componentes")
        components_layout = QVBoxLayout()

        self.health_table = QTableWidget()
        self.health_table.setColumnCount(4)
        self.health_table.setHorizontalHeaderLabels(["Componente", "Estado", "Mensaje", "Última Verificación"])
        self.health_table.setMaximumHeight(250)
        components_layout.addWidget(self.health_table)
        components_group.setLayout(components_layout)
        layout.addWidget(components_group)

        details_group = QGroupBox("Detalles")
        details_layout = QFormLayout()

        self.db_status_label = QLabel("-")
        self.exchange_status_label = QLabel("-")
        self.cache_status_label = QLabel("-")
        self.metrics_status_label = QLabel("-")

        details_layout.addRow("Database:", self.db_status_label)
        details_layout.addRow("Exchange:", self.exchange_status_label)
        details_layout.addRow("Cache:", self.cache_status_label)
        details_layout.addRow("Metrics Server:", self.metrics_status_label)
        details_group.setLayout(details_layout)
        layout.addWidget(details_group)

        buttons_layout = QHBoxLayout()
        self.refresh_button = QPushButton("Refrescar")
        buttons_layout.addWidget(self.refresh_button)
        layout.addLayout(buttons_layout)

        layout.addStretch()

    def update_state(self, state: dict) -> None:
        health_data = state.get("health", {})

        if health_data:
            is_healthy = health_data.get("healthy", True)
            self.overall_status_label.setText("OK" if is_healthy else "ERROR")
            self.overall_status_label.setStyleSheet(
                "color: green; font-weight: bold;" if is_healthy else "color: red; font-weight: bold;"
            )

            self.total_checks_label.setText(str(health_data.get("checks", 0)))
            self.healthy_checks_label.setText(str(health_data.get("healthy_checks", 0)))
            self.unhealthy_checks_label.setText(str(health_data.get("unhealthy_checks", 0)))
            self.last_check_label.setText(health_data.get("last_check", "-"))

            results = health_data.get("results", [])
            self.health_table.setRowCount(len(results))
            for i, result in enumerate(results):
                name = result.get("name", "")
                healthy = result.get("healthy", False)
                message = result.get("message", "")
                checked_at = result.get("checked_at", "")

                name_item = QTableWidgetItem(name)
                status_item = QTableWidgetItem("✓ OK" if healthy else "✗ ERROR")
                if healthy:
                    status_item.setBackground(QColor(0, 255, 0, 50))
                else:
                    status_item.setBackground(QColor(255, 0, 0, 50))
                message_item = QTableWidgetItem(message)
                time_item = QTableWidgetItem(checked_at)

                self.health_table.setItem(i, 0, name_item)
                self.health_table.setItem(i, 1, status_item)
                self.health_table.setItem(i, 2, message_item)
                self.health_table.setItem(i, 3, time_item)

            components = health_data.get("component_details", {})
            self.db_status_label.setText(components.get("database", "-"))
            self.exchange_status_label.setText(components.get("exchange", "-"))
            self.cache_status_label.setText(components.get("cache", "-"))
            self.metrics_status_label.setText(components.get("metrics_server", "-"))
