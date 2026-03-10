from __future__ import annotations

from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget

from reco_trading.ui.chart_widget import CandlestickChartWidget
from reco_trading.ui.components.animated import FlashValueLabel
from reco_trading.ui.components.formatting import as_text, fmt_number, fmt_pct, fmt_price, trend_arrow
from reco_trading.ui.state_manager import StateManager


class MetricPanel(QFrame):
    def __init__(self, title: str, fields: list[str]) -> None:
        super().__init__()
        self.setObjectName("panelCard")
        self.values: dict[str, FlashValueLabel] = {}
        layout = QVBoxLayout(self)
        header = QLabel(title)
        header.setObjectName("sectionTitle")
        layout.addWidget(header)

        grid = QGridLayout()
        for i, field in enumerate(fields):
            label = QLabel(field)
            label.setObjectName("metricLabel")
            value = FlashValueLabel("--")
            self.values[field] = value
            grid.addWidget(label, i, 0)
            grid.addWidget(value, i, 1)
        layout.addLayout(grid)


class DashboardTab(QWidget):
    def __init__(self, state_manager: StateManager | None = None) -> None:
        super().__init__()
        self.state_manager = state_manager

        root = QVBoxLayout(self)
        root.addWidget(QLabel("Dashboard"), alignment=None)

        content = QGridLayout()
        content.setSpacing(10)
        root.addLayout(content)

        self.account = MetricPanel("Account Panel", ["Balance", "Equity", "Daily PnL", "Total PnL", "Trades Today", "Win Rate"])
        self.market = MetricPanel("Market Panel", ["BTC/USDT Price", "Spread", "Trend", "ADX", "Volatility Regime", "Order Flow"])
        self.signal = MetricPanel("Signal Panel", ["Current Signal", "Confidence", "Signal Strength", "Last Signal Time"])
        self.bot = MetricPanel("Bot State Panel", ["Bot State", "Cooldown", "Last Trade", "System Health"])

        self.confidence = QProgressBar()
        self.confidence.setRange(0, 100)
        self.signal.layout().addWidget(self.confidence)

        self.chart_panel = QFrame()
        self.chart_panel.setObjectName("panelCard")
        chart_layout = QVBoxLayout(self.chart_panel)
        chart_title = QLabel("BTC/USDT Candles · 5m")
        chart_title.setObjectName("sectionTitle")
        chart_layout.addWidget(chart_title)
        self.chart = CandlestickChartWidget()
        chart_layout.addWidget(self.chart)

        content.addWidget(self.account, 0, 0)
        content.addWidget(self.market, 0, 1)
        content.addWidget(self.chart_panel, 1, 0, 1, 2)
        content.addWidget(self.signal, 2, 0)
        content.addWidget(self.bot, 2, 1)

    def update_state(self, state: dict) -> None:
        self.account.values["Balance"].set_value(f"{fmt_number(state.get('balance'), 2)} USDT")
        self.account.values["Equity"].set_value(f"{fmt_number(state.get('equity'), 2)} USDT")
        daily = float(state.get("daily_pnl", 0) or 0)
        self.account.values["Daily PnL"].set_value(fmt_number(daily, 2), positive=daily >= 0)
        total = float(state.get("total_pnl", state.get("analytics", {}).get("total_pnl", 0)) or 0)
        self.account.values["Total PnL"].set_value(fmt_number(total, 2), positive=total >= 0)
        self.account.values["Trades Today"].set_value(str(state.get("trades_today", "--")))
        self.account.values["Win Rate"].set_value(fmt_pct(state.get("win_rate", 0)))

        trend = as_text(state.get("trend"))
        self.market.values["BTC/USDT Price"].set_value(fmt_price(state.get("current_price", state.get("price", 0))))
        self.market.values["Spread"].set_value(fmt_number(state.get("spread", 0), 4))
        self.market.values["Trend"].set_value(f"{trend_arrow(trend)} {trend}")
        self.market.values["ADX"].set_value(fmt_number(state.get("adx", 0), 2))
        self.market.values["Volatility Regime"].set_value(as_text(state.get("volatility_regime")))
        self.market.values["Order Flow"].set_value(as_text(state.get("order_flow")))

        confidence = float(state.get("confidence", 0) or 0)
        self.signal.values["Current Signal"].set_value(as_text(state.get("signal")))
        self.signal.values["Confidence"].set_value(fmt_pct(confidence))
        self.signal.values["Signal Strength"].set_value(as_text(state.get("signal_strength", "NORMAL")))
        self.signal.values["Last Signal Time"].set_value(as_text(state.get("last_signal_time")))
        self.confidence.setValue(max(0, min(100, int(confidence * 100 if confidence <= 1 else confidence))))

        self.bot.values["Bot State"].set_value(as_text(state.get("bot_state", state.get("status"))))
        self.bot.values["Cooldown"].set_value(as_text(state.get("cooldown_timer", state.get("cooldown"))))
        self.bot.values["Last Trade"].set_value(as_text(state.get("last_trade")))
        system = state.get("system", {}) if isinstance(state.get("system"), dict) else {}
        self.bot.values["System Health"].set_value(as_text(system.get("exchange_status", "UNKNOWN")))

        self.chart.update_from_snapshot(state)
