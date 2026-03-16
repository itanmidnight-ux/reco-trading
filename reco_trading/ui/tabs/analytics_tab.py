from __future__ import annotations

from statistics import mean
from typing import Any

from PySide6.QtCore import QEasingCurve, QPropertyAnimation
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from reco_trading.ui.widgets.pnl_chart import PnlChart
from reco_trading.ui.widgets.stat_card import StatCard


class AnalyticsTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(10)

        header = QFrame()
        header.setObjectName("panelCard")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(14, 10, 14, 10)

        title_wrap = QVBoxLayout()
        title = QLabel("Performance Analytics")
        title.setObjectName("sectionTitle")
        subtitle = QLabel("Execution quality, risk-adjusted metrics and live equity diagnostics")
        subtitle.setObjectName("metricLabel")
        title_wrap.addWidget(title)
        title_wrap.addWidget(subtitle)
        header_layout.addLayout(title_wrap, 1)

        self.score_badge = QLabel("Model Quality: N/A")
        self.score_badge.setObjectName("smallMetricValue")
        self.score_badge.setStyleSheet("padding: 8px 12px; border-radius: 10px; background:#12213d;")
        header_layout.addWidget(self.score_badge)

        self.quality_bar = QProgressBar()
        self.quality_bar.setRange(0, 100)
        self.quality_bar.setValue(0)
        self.quality_bar.setTextVisible(True)
        self.quality_bar.setFormat("Quality %p%")
        self.quality_bar.setFixedWidth(220)
        header_layout.addWidget(self.quality_bar)
        root.addWidget(header)

        metrics_panel = QFrame()
        metrics_panel.setObjectName("panelCard")
        metrics_layout = QVBoxLayout(metrics_panel)
        metrics_layout.setContentsMargins(12, 12, 12, 12)
        metrics_layout.setSpacing(10)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        self.cards: dict[str, StatCard] = {}
        keys = [
            "total_trades",
            "win_rate",
            "profit_factor",
            "average_win",
            "average_loss",
            "largest_win",
            "largest_loss",
            "max_drawdown",
        ]
        for i, key in enumerate(keys):
            card = StatCard(key.replace("_", " ").title(), compact=True)
            self.cards[key] = card
            grid.addWidget(card, i // 4, i % 4)
        metrics_layout.addLayout(grid)

        chart_and_table = QHBoxLayout()
        chart_and_table.setSpacing(10)

        chart_wrap = QFrame()
        chart_wrap.setObjectName("metricCard")
        chart_layout = QVBoxLayout(chart_wrap)
        chart_layout.setContentsMargins(10, 10, 10, 10)
        chart_title = QLabel("Equity / Balance Curve")
        chart_title.setObjectName("smallMetricValue")
        chart_layout.addWidget(chart_title)
        self.equity_curve = PnlChart("Performance")
        chart_layout.addWidget(self.equity_curve)

        table_wrap = QFrame()
        table_wrap.setObjectName("metricCard")
        table_layout = QVBoxLayout(table_wrap)
        table_layout.setContentsMargins(10, 10, 10, 10)
        table_title = QLabel("Advanced Ratios")
        table_title.setObjectName("smallMetricValue")
        table_layout.addWidget(table_title)
        self.analysis_table = QTableWidget(0, 2)
        self.analysis_table.setHorizontalHeaderLabels(["Analysis", "Value"])
        self.analysis_table.horizontalHeader().setStretchLastSection(True)
        table_layout.addWidget(self.analysis_table)

        chart_and_table.addWidget(chart_wrap, 3)
        chart_and_table.addWidget(table_wrap, 2)
        metrics_layout.addLayout(chart_and_table)

        insights_wrap = QFrame()
        insights_wrap.setObjectName("metricCard")
        insights_layout = QVBoxLayout(insights_wrap)
        insights_layout.setContentsMargins(10, 10, 10, 10)
        insights_title = QLabel("Live Insights")
        insights_title.setObjectName("smallMetricValue")
        insights_layout.addWidget(insights_title)

        self.insights = QListWidget()
        self.insights.addItems(["Waiting analytics signals..."])
        insights_layout.addWidget(self.insights)
        metrics_layout.addWidget(insights_wrap)

        root.addWidget(metrics_panel)

        self._badge_animation = QPropertyAnimation(self.score_badge, b"windowOpacity", self)
        self._badge_animation.setDuration(260)
        self._badge_animation.setStartValue(0.55)
        self._badge_animation.setEndValue(1.0)
        self._badge_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _build_analytics_from_trades(self, state: dict[str, Any], analytics: dict[str, Any]) -> dict[str, Any]:
        trade_history = state.get("trade_history", [])
        closed_pnls = [self._to_float(t.get("pnl"), 0.0) for t in trade_history if isinstance(t, dict) and t.get("pnl") is not None]
        wins = [p for p in closed_pnls if p > 0]
        losses = [p for p in closed_pnls if p < 0]

        computed: dict[str, Any] = dict(analytics)
        if closed_pnls:
            computed.setdefault("total_trades", len(closed_pnls))
            computed.setdefault("win_rate", (len(wins) / len(closed_pnls)) if closed_pnls else 0.0)
            computed.setdefault("average_win", mean(wins) if wins else 0.0)
            computed.setdefault("average_loss", mean(losses) if losses else 0.0)
            computed.setdefault("largest_win", max(wins) if wins else 0.0)
            computed.setdefault("largest_loss", min(losses) if losses else 0.0)
            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))
            computed.setdefault("profit_factor", (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0))
            computed.setdefault("expectancy", sum(closed_pnls) / len(closed_pnls))

        if not isinstance(computed.get("equity_curve"), list) or len(computed.get("equity_curve", [])) <= 1:
            equity = self._to_float(state.get("equity"), 0.0)
            session_pnl = self._to_float(state.get("session_pnl"), 0.0)
            base = max(equity - session_pnl, 0.0)
            computed["equity_curve"] = [base] + [base + sum(closed_pnls[: i + 1]) for i in range(len(closed_pnls))]
            if not computed["equity_curve"]:
                computed["equity_curve"] = [equity]

        curve = [self._to_float(v, 0.0) for v in computed.get("equity_curve", []) if isinstance(v, (int, float))]
        if curve:
            peak = curve[0]
            max_dd = 0.0
            for value in curve:
                peak = max(peak, value)
                if peak > 0:
                    max_dd = max(max_dd, (peak - value) / peak)
            computed.setdefault("max_drawdown", max_dd)

        return computed

    def update_state(self, state: dict) -> None:
        analytics = state.get("analytics", {})
        if not isinstance(analytics, dict):
            analytics = {}
        analytics = self._build_analytics_from_trades(state, analytics)

        for key, card in self.cards.items():
            val = analytics.get(key, "-")
            if key in {"win_rate", "max_drawdown"}:
                card.set_value(f"{self._to_float(val, 0.0) * 100:.2f}%")
            elif key in {"profit_factor", "average_win", "average_loss", "largest_win", "largest_loss"}:
                numeric = self._to_float(val, 0.0)
                if key == "profit_factor" and numeric > 1_000_000:
                    card.set_value("∞")
                else:
                    card.set_value(f"{numeric:.4f}")
            else:
                card.set_value(str(val))

        points = [self._to_float(v, 0.0) for v in analytics.get("equity_curve", []) if isinstance(v, (int, float))]
        self.equity_curve.plot(points)

        trade_history = state.get("trade_history", [])
        slippage_values: list[float] = []
        for trade in trade_history:
            if not isinstance(trade, dict):
                continue
            for key in ("entry_slippage_ratio", "exit_slippage_ratio"):
                value = trade.get(key)
                if isinstance(value, (int, float)):
                    slippage_values.append(float(value))

        metric_rows = [
            ("Sharpe Ratio", analytics.get("sharpe_ratio", "-")),
            ("Sortino Ratio", analytics.get("sortino_ratio", "-")),
            ("Max Drawdown", f"{self._to_float(analytics.get('max_drawdown'), 0.0):.4%}"),
            ("Expectancy", f"{self._to_float(analytics.get('expectancy'), 0.0):.4f}"),
            ("Avg Duration", analytics.get("avg_trade_duration", "-")),
        ]
        if slippage_values:
            avg_slippage = sum(slippage_values) / len(slippage_values)
            metric_rows.append(("Avg Slippage", f"{avg_slippage:.4%}"))

        self.analysis_table.setRowCount(0)
        for row, (name, value) in enumerate(metric_rows):
            self.analysis_table.insertRow(row)
            self.analysis_table.setItem(row, 0, QTableWidgetItem(name))
            self.analysis_table.setItem(row, 1, QTableWidgetItem(str(value)))

        win_rate = self._to_float(analytics.get("win_rate"), 0.0)
        confidence = self._to_float(state.get("confidence"), 0.0)
        score = int(((win_rate * 0.65) + (confidence * 0.35)) * 100)
        score = max(0, min(score, 100))
        self.score_badge.setText(f"Model Quality: {score}%")
        accent = "#16c784" if score >= 65 else "#f0b90b" if score >= 45 else "#ea3943"
        self.score_badge.setStyleSheet(f"padding: 8px 12px; border-radius: 10px; color: {accent}; background:#12213d;")
        self.quality_bar.setValue(score)
        self._badge_animation.stop()
        self._badge_animation.start()

        self.insights.clear()
        self.insights.addItem(f"Signal: {state.get('signal', '-')}")
        self.insights.addItem(f"Confidence: {confidence:.2f}")
        self.insights.addItem(f"Open position: {state.get('open_position', '-')}")
        self.insights.addItem(f"Trades today: {state.get('trades_today', 0)}")
        self.insights.addItem(f"Session PnL: {self._to_float(state.get('session_pnl'), 0.0):.4f}")
