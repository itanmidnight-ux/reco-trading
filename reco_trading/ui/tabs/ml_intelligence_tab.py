from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class MLIntelligenceTab(QWidget):
    def __init__(self, state_manager=None) -> None:
        super().__init__()
        self.state_manager = state_manager
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)

        header = QLabel("<h2>🤖 RecoAI - Machine Learning</h2>")
        layout.addWidget(header)

        status_group = QGroupBox("Estado del Modelo")
        status_layout = QFormLayout()

        self.model_status_label = QLabel("Activo")
        self.model_type_label = QLabel("Ensemble (Momentum, Trend, Volume, Pattern, Sentiment)")
        self.trained_samples_label = QLabel("0")
        self.last_train_label = QLabel("En tiempo real")
        self.next_train_label = QLabel("Continuo")

        status_layout.addRow("Estado:", self.model_status_label)
        status_layout.addRow("Tipo de Modelo:", self.model_type_label)
        status_layout.addRow("Muestras de Entrenamiento:", self.trained_samples_label)
        status_layout.addRow("Último Entrenamiento:", self.last_train_label)
        status_layout.addRow("Próximo Entrenamiento:", self.next_train_label)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        prediction_group = QGroupBox("Predicciones en Vivo")
        prediction_layout = QFormLayout()

        self.ml_direction_label = QLabel("-")
        self.ml_confidence_label = QLabel("-")
        self.ml_predicted_move_label = QLabel("-")
        self.market_regime_label = QLabel("-")

        prediction_layout.addRow("Dirección:", self.ml_direction_label)
        prediction_layout.addRow("Confianza:", self.ml_confidence_label)
        prediction_layout.addRow("Movimiento Predicho:", self.ml_predicted_move_label)
        prediction_layout.addRow("Régimen de Mercado:", self.market_regime_label)
        prediction_group.setLayout(prediction_layout)
        layout.addWidget(prediction_group)

        metrics_group = QGroupBox("Métricas del Modelo")
        metrics_layout = QFormLayout()

        self.accuracy_label = QLabel("0.00")
        self.precision_label = QLabel("0.00")
        self.recall_label = QLabel("0.00")
        self.f1_label = QLabel("0.00")

        metrics_layout.addRow("Accuracy:", self.accuracy_label)
        metrics_layout.addRow("Precision:", self.precision_label)
        metrics_layout.addRow("Recall:", self.recall_label)
        metrics_layout.addRow("F1 Score:", self.f1_label)
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)

        features_group = QGroupBox("Características (Features)")
        features_layout = QVBoxLayout()

        self.features_table = QTableWidget()
        self.features_table.setColumnCount(3)
        self.features_table.setHorizontalHeaderLabels(["Nombre", "Tipo", "Importancia"])
        self.features_table.setMaximumHeight(150)
        features_layout.addWidget(self.features_table)
        features_group.setLayout(features_layout)
        layout.addWidget(features_group)

        buttons_layout = QHBoxLayout()
        self.train_button = QPushButton("Entrenar Modelo")
        self.predict_button = QPushButton("Nueva Predicción")
        self.save_button = QPushButton("Guardar Modelo")
        self.train_button.clicked.connect(self._on_train_clicked)
        self.predict_button.clicked.connect(self._on_predict_clicked)
        self.save_button.clicked.connect(self._on_save_clicked)
        buttons_layout.addWidget(self.train_button)
        buttons_layout.addWidget(self.predict_button)
        buttons_layout.addWidget(self.save_button)
        layout.addLayout(buttons_layout)

        layout.addStretch()

    def _on_train_clicked(self) -> None:
        if self.state_manager:
            self.state_manager.trigger_ml_train()

    def _on_predict_clicked(self) -> None:
        if self.state_manager:
            self.state_manager.trigger_ml_predict()

    def _on_save_clicked(self) -> None:
        if self.state_manager:
            self.state_manager.trigger_ml_save()

    def update_state(self, state: dict) -> None:
        ml_data = state.get("ml_intelligence", {})
        
        if ml_data:
            self.model_status_label.setText(ml_data.get("status", "Activo"))
            self.model_type_label.setText(ml_data.get("model_type", "Ensemble"))
            self.trained_samples_label.setText(str(ml_data.get("training_samples", 0)))
            self.last_train_label.setText(ml_data.get("last_train", "En tiempo real"))
            self.next_train_label.setText(ml_data.get("next_train", "Continuo"))
        
        direction = state.get("ml_direction", "-")
        confidence = state.get("ml_confidence", 0)
        predicted_move = state.get("ml_predicted_move", 0)
        regime = state.get("market_regime", "-")
        
        self.ml_direction_label.setText(str(direction))
        self.ml_confidence_label.setText(f"{float(confidence)*100:.1f}%" if confidence else "-")
        self.ml_predicted_move_label.setText(f"{float(predicted_move):.2f}%" if predicted_move else "-")
        self.market_regime_label.setText(str(regime))
        
        metrics = ml_data.get("metrics", {})
        if metrics:
            self.accuracy_label.setText(f"{metrics.get('accuracy', 0):.4f}")
            self.precision_label.setText(f"{metrics.get('precision', 0):.4f}")
            self.recall_label.setText(f"{metrics.get('recall', 0):.4f}")
            self.f1_label.setText(f"{metrics.get('f1', 0):.4f}")
        else:
            self.accuracy_label.setText("N/A (entrenando)")
            self.precision_label.setText("N/A (entrenando)")
            self.recall_label.setText("N/A (entrenando)")
            self.f1_label.setText("N/A (entrenando)")

        features = ml_data.get("features", [])
        if features:
            self.features_table.setRowCount(len(features))
            for i, feat in enumerate(features):
                self.features_table.setItem(i, 0, QTableWidgetItem(feat.get("name", "")))
                self.features_table.setItem(i, 1, QTableWidgetItem(feat.get("type", "")))
                self.features_table.setItem(i, 2, QTableWidgetItem(f"{feat.get('importance', 0):.4f}"))
        else:
            default_features = [
                {"name": "momentum", "type": "technical", "importance": 1.0},
                {"name": "trend", "type": "technical", "importance": 1.0},
                {"name": "volume", "type": "volume", "importance": 0.8},
                {"name": "pattern", "type": "candlestick", "importance": 0.8},
                {"name": "sentiment", "type": "market", "importance": 1.0},
            ]
            self.features_table.setRowCount(len(default_features))
            for i, feat in enumerate(default_features):
                self.features_table.setItem(i, 0, QTableWidgetItem(feat.get("name", "")))
                self.features_table.setItem(i, 1, QTableWidgetItem(feat.get("type", "")))
                self.features_table.setItem(i, 2, QTableWidgetItem(f"{feat.get('importance', 0):.4f}"))
