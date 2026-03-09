from __future__ import annotations

from typing import Any

from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QScrollArea, QSplitter, QVBoxLayout, QWidget

from reco_trading.ui.chart_widget import CandlestickChartWidget
from reco_trading.ui.state_manager import StateManager
from reco_trading.ui.widgets.stat_card import StatCard


class DashboardTab(QWidget):
    def __init__(self, state_manager: StateManager | None = None) -> None:
        super().__init__()
        self.state_manager = state_manager

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        title = QLabel("Dashboard")
        title.setObjectName("sectionTitle")
        root.addWidget(title)

        splitter = QSplitter()
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.chart_panel = self._panel("Realtime Candlestick Chart")
        self.chart = CandlestickChartWidget()
        self.chart_panel.layout().addWidget(self.chart)
        left_layout.addWidget(self.chart_panel, 2)

        self.activity_panel = self._panel("System Activity")
        self.feed = QLabel("[--:--] Waiting for events")
        self.feed.setObjectName("smallMetricValue")
        self.feed.setWordWrap(True)
        self.activity_panel.layout().addWidget(self.feed)
        left_layout.addWidget(self.activity_panel, 1)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_content = QWidget()
        right_grid = QGridLayout(right_content)
        right_grid.setSpacing(8)

        self.market_cards = self._make_cards(
            "Market Overview",
            ["pair", "price", "spread", "trend", "adx", "volatility_regime", "order_flow"],
            ["Pair", "Price", "Spread", "Trend", "ADX", "Volatility Regime", "Order Flow"],
        )
        self.account_cards = self._make_cards(
            "Account Performance",
            ["usdt_balance", "btc_balance", "btc_value", "total_equity", "daily_pnl", "trades_today", "win_rate"],
            ["USDT Balance", "BTC Balance", "BTC Value", "Total Equity", "Daily PnL", "Trades Today", "Win Rate"],
        )
        self.position_cards = self._make_cards(
            "Position Status",
            ["position_side", "entry_price", "current_price", "position_size", "position_value", "unrealized_pnl"],
            ["Position Side", "Entry Price", "Current Price", "Position Size", "Position Value", "Unrealized PnL"],
        )
        self.exposure_cards = self._make_cards(
            "Portfolio Exposure", ["btc_exposure", "usdt_exposure"], ["BTC Exposure %", "USDT Exposure %"]
        )
        self.risk_cards = self._make_cards(
            "Risk Metrics",
            ["risk_per_trade", "current_exposure", "max_concurrent_trades", "daily_drawdown"],
            ["Risk Per Trade", "Current Exposure", "Max Concurrent Trades", "Daily Drawdown"],
        )
        self.system_cards = self._make_cards(
            "System Status",
            ["bot_mode", "engine_state", "exchange_status", "database_status", "api_latency"],
            ["Bot Mode", "Engine State", "Exchange Status", "Database Status", "API Latency"],
        )

        panels = [
            self.market_cards,
            self.account_cards,
            self.position_cards,
            self.exposure_cards,
            self.risk_cards,
            self.system_cards,
        ]
        for i, group in enumerate(panels):
            right_grid.addWidget(group["panel"], i // 2, i % 2)

        right_grid.setColumnStretch(0, 1)
        right_grid.setColumnStretch(1, 1)
        right_grid.setRowStretch(3, 1)
        right_scroll.setWidget(right_content)

        splitter.addWidget(left_container)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

    def _panel(self, name: str) -> QFrame:
        panel = QFrame()
        panel.setObjectName("panelCard")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        head = QLabel(name)
        head.setObjectName("metricLabel")
        layout.addWidget(head)
        return panel

    def _make_cards(self, title: str, keys: list[str], labels: list[str]) -> dict[str, Any]:
        panel = self._panel(title)
        grid = QGridLayout()
        cards: dict[str, StatCard] = {}
        for i, (key, label) in enumerate(zip(keys, labels)):
            card = StatCard(label, compact=True)
            cards[key] = card
            grid.addWidget(card, i // 2, i % 2)
        panel.layout().addLayout(grid)
        return {"panel": panel, "cards": cards}

    def update_state(self, state: dict[str, Any]) -> None:
        try:
            self._update_market(state)
            self._update_account(state)
            self._update_position(state)
            self._update_exposure(state)
            self._update_risk(state)
            self._update_system(state)
            logs = state.get("logs", [])[-6:]
            self.feed.setText("\n".join(f"[{log.get('time', '--:--')}] {log.get('message', '-')}" for log in logs) or "-")
            self.chart.update_from_snapshot(state)
        except Exception:
            return

    def _set(self, group: dict[str, Any], key: str, value: Any) -> None:
        group["cards"][key].set_value("-" if value in (None, "") else str(value))

    def _update_market(self, state: dict[str, Any]) -> None:
        self._set(self.market_cards, "pair", state.get("pair", "BTC/USDT"))
        self._set(self.market_cards, "price", _fmt(state.get("current_price", state.get("price")), 2))
        self._set(self.market_cards, "spread", _fmt(state.get("spread"), 6))
        self._set(self.market_cards, "trend", state.get("trend", "-"))
        self._set(self.market_cards, "adx", _fmt(state.get("adx"), 2))
        self._set(self.market_cards, "volatility_regime", state.get("volatility_regime", "-"))
        self._set(self.market_cards, "order_flow", state.get("order_flow", "-"))

    def _update_account(self, state: dict[str, Any]) -> None:
        self._set(self.account_cards, "usdt_balance", _fmt(state.get("balance", state.get("usdt_balance")), 2))
        self._set(self.account_cards, "btc_balance", _fmt(state.get("btc_balance"), 6))
        self._set(self.account_cards, "btc_value", _fmt(state.get("btc_value"), 2))
        self._set(self.account_cards, "total_equity", _fmt(state.get("total_equity", state.get("equity")), 2))
        self._set(self.account_cards, "daily_pnl", _fmt(state.get("daily_pnl"), 2))
        self._set(self.account_cards, "trades_today", state.get("trades_today", "-"))
        win_rate = _to_float(state.get("win_rate"))
        self._set(self.account_cards, "win_rate", f"{win_rate * 100:.2f}%" if win_rate is not None else "-")

    def _update_position(self, state: dict[str, Any]) -> None:
        self._set(self.position_cards, "position_side", state.get("position_side", state.get("open_position", "-")))
        entry = _to_float(state.get("entry_price"))
        current = _to_float(state.get("current_price", state.get("price")))
        size = _to_float(state.get("position_size", state.get("position_size_btc")))
        value = size * current if size is not None and current is not None else state.get("position_value")
        upnl = (current - entry) * size if entry is not None and current is not None and size is not None else None
        self._set(self.position_cards, "entry_price", _fmt(entry, 2))
        self._set(self.position_cards, "current_price", _fmt(current, 2))
        self._set(self.position_cards, "position_size", _fmt(size, 6))
        self._set(self.position_cards, "position_value", _fmt(value, 2))
        self._set(self.position_cards, "unrealized_pnl", _fmt(upnl, 2))

    def _update_exposure(self, state: dict[str, Any]) -> None:
        total = _to_float(state.get("total_equity", state.get("equity")))
        btc = _to_float(state.get("btc_value"))
        usdt = _to_float(state.get("balance", state.get("usdt_balance")))
        btc_pct = (btc / total) * 100 if total and btc is not None else None
        usdt_pct = (usdt / total) * 100 if total and usdt is not None else None
        self._set(self.exposure_cards, "btc_exposure", _fmt(btc_pct, 2))
        self._set(self.exposure_cards, "usdt_exposure", _fmt(usdt_pct, 2))

    def _update_risk(self, state: dict[str, Any]) -> None:
        risk = state.get("risk_metrics", {}) or {}
        self._set(self.risk_cards, "risk_per_trade", risk.get("risk_per_trade", "-"))
        self._set(self.risk_cards, "current_exposure", risk.get("current_exposure", "-"))
        self._set(self.risk_cards, "max_concurrent_trades", risk.get("max_concurrent_trades", "-"))
        self._set(self.risk_cards, "daily_drawdown", risk.get("daily_drawdown", "-"))

    def _update_system(self, state: dict[str, Any]) -> None:
        system = state.get("system", {}) or {}
        self._set(self.system_cards, "bot_mode", state.get("bot_mode", "TESTNET" if state.get("testnet", True) else "LIVE"))
        self._set(self.system_cards, "engine_state", state.get("status", state.get("engine_state", "-")))
        self._set(self.system_cards, "exchange_status", system.get("exchange_status", "-"))
        self._set(self.system_cards, "database_status", system.get("database_status", "-"))
        self._set(self.system_cards, "api_latency", _fmt(system.get("api_latency_ms", state.get("api_latency_ms")), 2))


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int) -> str:
    if value in (None, ""):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)
