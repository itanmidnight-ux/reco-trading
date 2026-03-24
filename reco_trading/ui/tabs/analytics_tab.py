from __future__ import annotations

from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QListWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from reco_trading.ui.widgets.pnl_chart import PnlChart
from reco_trading.ui.widgets.stat_card import StatCard


class AnalyticsTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._last_signature: tuple[object, ...] | None = None
        layout = QVBoxLayout(self)
        title = QLabel("Performance Analytics")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        subtitle = QLabel("Detailed strategy metrics, quality score and equity behavior")
        subtitle.setObjectName("metricLabel")
        layout.addWidget(subtitle)

        panel = QFrame()
        panel.setObjectName("panelCard")
        layout.addWidget(panel)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(12, 12, 12, 12)

        self.score_badge = QLabel("Model Quality: N/A")
        self.score_badge.setObjectName("smallMetricValue")
        panel_layout.addWidget(self.score_badge)

        grid = QGridLayout()
        self.cards = {}
        keys = ["total_trades", "win_rate", "profit_factor", "average_win", "average_loss", "largest_win", "largest_loss"]
        for i, key in enumerate(keys):
            card = StatCard(key.replace("_", " ").title(), compact=True)
            self.cards[key] = card
            grid.addWidget(card, i // 4, i % 4)
        panel_layout.addLayout(grid)

        self.equity_curve = PnlChart("Performance")
        panel_layout.addWidget(self.equity_curve)

        self.analysis_table = QTableWidget(0, 2)
        self.analysis_table.setHorizontalHeaderLabels(["Analysis", "Value"])
        self.analysis_table.horizontalHeader().setStretchLastSection(True)
        panel_layout.addWidget(self.analysis_table)

        self.insights = QListWidget()
        self.insights.addItems(["Waiting analytics signals..."])
        panel_layout.addWidget(self.insights)

    def update_state(self, state: dict) -> None:
        trade_history = state.get("trade_history", [])
        closed_pnls = [float(t.get("pnl", 0) or 0) for t in trade_history if t.get("pnl") not in {None, "-"}]
        wins = [p for p in closed_pnls if p > 0]
        losses = [p for p in closed_pnls if p < 0]
        analytics = dict(state.get("analytics", {}) or {})
        analytics.setdefault("total_trades", len(trade_history))
        analytics.setdefault("win_rate", (len(wins) / len(closed_pnls)) if closed_pnls else 0.0)
        analytics.setdefault("profit_factor", (sum(wins) / abs(sum(losses))) if losses else (float("inf") if wins else 0.0))
        analytics.setdefault("average_win", (sum(wins) / len(wins)) if wins else 0.0)
        analytics.setdefault("average_loss", (sum(losses) / len(losses)) if losses else 0.0)
        analytics.setdefault("largest_win", max(wins) if wins else 0.0)
        analytics.setdefault("largest_loss", min(losses) if losses else 0.0)
        points = [float(v) for v in analytics.get("equity_curve", []) if isinstance(v, (int, float))]
        if not points:
            current_equity = float(state.get("equity", 0) or 0)
            running = current_equity - sum(closed_pnls)
            points = [running]
            for pnl in reversed(closed_pnls):
                running += pnl
                points.append(running)
        signature = (
            tuple(sorted((k, str(analytics.get(k, "-"))) for k in self.cards)),
            tuple(round(v, 8) for v in points[-240:]),
            tuple(
                (
                    str(trade.get("trade_id", "")),
                    str(trade.get("status", "")),
                    str(trade.get("pnl", "")),
                    str(trade.get("entry_slippage_ratio", "")),
                    str(trade.get("exit_slippage_ratio", "")),
                )
                for trade in trade_history[-50:]
            ),
            str(state.get("signal", "-")),
            str(state.get("open_position", "-")),
            str(state.get("trades_today", 0)),
            str(state.get("session_pnl", 0)),
            str(state.get("confidence", 0)),
        )
        if signature == self._last_signature:
            return
        self._last_signature = signature

        for key, card in self.cards.items():
            val = analytics.get(key, "-")
            if key == "win_rate":
                try:
                    card.set_value(f"{float(val) * 100:.2f}%")
                except (TypeError, ValueError):
                    card.set_value("-")
            elif key == "profit_factor" and val == float("inf"):
                card.set_value("∞")
            else:
                card.set_value(str(val))

        self.equity_curve.plot(points)

        metric_rows = [
            ("Sharpe Ratio", analytics.get("sharpe_ratio", "-")),
            ("Sortino Ratio", analytics.get("sortino_ratio", "-")),
            ("Max Drawdown", analytics.get("max_drawdown", "-")),
            ("Expectancy", analytics.get("expectancy", "-")),
            ("Avg Duration", analytics.get("avg_trade_duration", "-")),
        ]
        slippage_values: list[float] = []
        for trade in trade_history:
            for key in ("entry_slippage_ratio", "exit_slippage_ratio"):
                value = trade.get(key)
                if isinstance(value, (int, float)):
                    slippage_values.append(float(value))
        if slippage_values:
            avg_slippage = sum(slippage_values) / len(slippage_values)
            metric_rows.append(("Avg Slippage", f"{avg_slippage:.4%}"))
        expectancy_rolling = sum(closed_pnls[-20:]) / max(len(closed_pnls[-20:]), 1) if closed_pnls else 0.0
        metric_rows.append(("Expectancy Rolling(20)", f"{expectancy_rolling:.4f}"))
        self.analysis_table.setRowCount(0)
        for row, (name, value) in enumerate(metric_rows):
            self.analysis_table.insertRow(row)
            self.analysis_table.setItem(row, 0, QTableWidgetItem(name))
            self.analysis_table.setItem(row, 1, QTableWidgetItem(str(value)))

        win_rate = float(analytics.get("win_rate", 0) or 0)
        confidence = float(state.get("confidence", 0) or 0)
        score = int(((win_rate * 0.6) + (confidence * 0.4)) * 100)
        self.score_badge.setText(f"Model Quality: {score}%")
        self.score_badge.setStyleSheet(f"color: {'#16c784' if score >= 60 else '#f0b90b'};")

        self.insights.clear()
        self.insights.addItem(f"Signal: {state.get('signal', '-')}")
        self.insights.addItem(f"Confidence: {confidence:.2f}")
        self.insights.addItem(f"Open position: {state.get('open_position', '-')}")
        self.insights.addItem(f"Trades today: {state.get('trades_today', 0)}")
        self.insights.addItem(f"Session PnL: {state.get('session_pnl', 0)}")
        session_stats = (
            state.get("session_stats")
            or (state.get("analytics") or {}).get("session_stats", {})
        )
        if session_stats:
            streak = int(session_stats.get("streak", 0))
            rec = str(session_stats.get("recommendation", "NORMAL"))
            pf_sess = float(session_stats.get("profit_factor", 0) or 0)
            sharpe = float(session_stats.get("sharpe", 0) or 0)
            self.insights.addItem(f"Session streak: {'+' if streak > 0 else ''}{streak}")
            self.insights.addItem(f"Recommendation: {rec}")
            self.insights.addItem(f"Session PF: {pf_sess:.2f}")
            self.insights.addItem(f"Sharpe est.: {sharpe:.2f}")
        regimes = {"trending": [], "ranging": []}
        hourly: dict[int, list[float]] = {}
        for trade in trade_history:
            pnl = float(trade.get("pnl", 0) or 0)
            regime = str(trade.get("regime", "ranging")).lower()
            bucket = "trending" if "trend" in regime else "ranging"
            regimes[bucket].append(pnl)
            ts = str(trade.get("timestamp", ""))
            hour = int(ts[11:13]) if len(ts) >= 13 and ts[11:13].isdigit() else 0
            hourly.setdefault(hour, []).append(pnl)
        for regime_name, pnls in regimes.items():
            if pnls:
                wins_reg = len([p for p in pnls if p > 0])
                ratio = wins_reg / len(pnls)
                self.insights.addItem(f"{regime_name.title()} W/L: {ratio:.1%}")
        best_hours = sorted(((h, sum(v)) for h, v in hourly.items()), key=lambda x: x[1], reverse=True)[:3]
        if best_hours:
            self.insights.addItem("Heatmap hora top: " + ", ".join(f\"{h:02d}h={p:+.2f}\" for h, p in best_hours))
