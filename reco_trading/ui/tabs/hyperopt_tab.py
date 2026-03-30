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


class HyperoptTab(QWidget):
    def __init__(self, state_manager=None) -> None:
        super().__init__()
        self.state_manager = state_manager
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        header = QLabel("<h2>⚡ Hyperopt - Optimización</h2>")
        layout.addWidget(header)

        status_group = QGroupBox("Estado de Optimización")
        status_layout = QFormLayout()

        self.optimizer_status_label = QLabel("Inactivo")
        self.current_trial_label = QLabel("0")
        self.total_trials_label = QLabel("0")
        self.best_score_label = QLabel("0.00")
        self.elapsed_time_label = QLabel("0s")

        status_layout.addRow("Estado:", self.optimizer_status_label)
        status_layout.addRow("Trial Actual:", self.current_trial_label)
        status_layout.addRow("Total Trials:", self.total_trials_label)
        status_layout.addRow("Mejor Score:", self.best_score_label)
        status_layout.addRow("Tiempo Transcurrido:", self.elapsed_time_label)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        best_params_group = QGroupBox("Mejores Parámetros")
        best_params_layout = QFormLayout()

        self.risk_fraction_label = QLabel("-")
        self.confidence_threshold_label = QLabel("-")
        self.maker_fee_label = QLabel("-")
        self.taker_fee_label = QLabel("-")

        best_params_layout.addRow("Risk Fraction:", self.risk_fraction_label)
        best_params_layout.addRow("Confidence Threshold:", self.confidence_threshold_label)
        best_params_layout.addRow("Maker Fee:", self.maker_fee_label)
        best_params_layout.addRow("Taker Fee:", self.taker_fee_label)
        best_params_group.setLayout(best_params_layout)
        layout.addWidget(best_params_group)

        trials_group = QGroupBox("Historial de Trials")
        trials_layout = QVBoxLayout()

        self.trials_table = QTableWidget()
        self.trials_table.setColumnCount(5)
        self.trials_table.setHorizontalHeaderLabels(["Trial", "Score", "Return", "Win Rate", "Sharpe"])
        self.trials_table.setMaximumHeight(200)
        trials_layout.addWidget(self.trials_table)
        trials_group.setLayout(trials_layout)
        layout.addWidget(trials_group)

        buttons_layout = QHBoxLayout()
        self.start_button = QPushButton("Iniciar Optimización")
        self.stop_button = QPushButton("Detener")
        self.export_button = QPushButton("Exportar Resultados")
        buttons_layout.addWidget(self.start_button)
        buttons_layout.addWidget(self.stop_button)
        buttons_layout.addWidget(self.export_button)
        layout.addLayout(buttons_layout)

        layout.addStretch()

    def update_state(self, state: dict) -> None:
        hyperopt_data = state.get("hyperopt", {})

        if hyperopt_data:
            self.optimizer_status_label.setText(hyperopt_data.get("status", "Inactivo"))
            self.current_trial_label.setText(str(hyperopt_data.get("current_trial", 0)))
            self.total_trials_label.setText(str(hyperopt_data.get("total_trials", 0)))
            self.best_score_label.setText(f"{hyperopt_data.get('best_score', 0):.4f}")
            self.elapsed_time_label.setText(f"{hyperopt_data.get('elapsed_seconds', 0):.1f}s")

            best_params = hyperopt_data.get("best_params", {})
            self.risk_fraction_label.setText(f"{best_params.get('risk_fraction', 0):.4f}")
            self.confidence_threshold_label.setText(f"{best_params.get('confidence_threshold', 0):.2f}")
            self.maker_fee_label.setText(f"{best_params.get('maker_fee_rate', 0):.4f}")
            self.taker_fee_label.setText(f"{best_params.get('taker_fee_rate', 0):.4f}")

            trials = hyperopt_data.get("trials", [])
            self.trials_table.setRowCount(len(trials))
            for i, trial in enumerate(trials):
                self.trials_table.setItem(i, 0, QTableWidgetItem(str(trial.get("id", i))))
                self.trials_table.setItem(i, 1, QTableWidgetItem(f"{trial.get('score', 0):.4f}"))
                self.trials_table.setItem(i, 2, QTableWidgetItem(f"{trial.get('return', 0):.2%}"))
                self.trials_table.setItem(i, 3, QTableWidgetItem(f"{trial.get('win_rate', 0):.2%}"))
                self.trials_table.setItem(i, 4, QTableWidgetItem(f"{trial.get('sharpe', 0):.4f}"))
