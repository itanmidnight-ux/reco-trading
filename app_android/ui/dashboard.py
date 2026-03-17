from __future__ import annotations

from typing import Any

from kivy.clock import Clock
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem

from config import API_URL, REFRESH_INTERVAL_SECONDS
from services.api_client import APIClient
from ui.components import Card, GradientSection, LabeledField, PrimaryButton, VerticalScroll


class Dashboard(BoxLayout):
    def __init__(self, **kwargs) -> None:
        super().__init__(orientation="vertical", padding=dp(10), spacing=dp(8), **kwargs)
        self.client = APIClient()
        self._latest_runtime: dict[str, Any] = {}

        self.header = Label(
            text="Reco Trading Mobile · Advanced Institutional Console",
            font_size="20sp",
            bold=True,
            size_hint_y=None,
            height=dp(36),
            color=(0.92, 0.96, 1, 1),
        )
        self.connection = Label(
            text="DISCONNECTED",
            font_size="13sp",
            size_hint_y=None,
            height=dp(22),
            color=(1, 0.45, 0.45, 1),
        )

        self.tabs = TabbedPanel(do_default_tab=False, tab_height=dp(44), background_color=(0.08, 0.11, 0.18, 1))
        self.overview_tab = TabbedPanelItem(text="Overview")
        self.market_tab = TabbedPanelItem(text="Market")
        self.controls_tab = TabbedPanelItem(text="Controls")
        self.settings_tab = TabbedPanelItem(text="Settings")

        self.overview_tab.content = self._build_overview_tab()
        self.market_tab.content = self._build_market_tab()
        self.controls_tab.content = self._build_controls_tab()
        self.settings_tab.content = self._build_settings_tab()

        self.tabs.add_widget(self.overview_tab)
        self.tabs.add_widget(self.market_tab)
        self.tabs.add_widget(self.controls_tab)
        self.tabs.add_widget(self.settings_tab)

        self.add_widget(self.header)
        self.add_widget(self.connection)
        self.add_widget(self.tabs)

        Clock.schedule_interval(lambda *_: self.refresh(), REFRESH_INTERVAL_SECONDS)
        self.refresh()

    def _build_overview_tab(self):
        scroll = VerticalScroll()
        body = BoxLayout(orientation="vertical", spacing=dp(10), size_hint_y=None, padding=[0, dp(4), 0, dp(14)])
        body.bind(minimum_height=body.setter("height"))

        self.kpi_grid = GridLayout(cols=2, spacing=dp(8), size_hint_y=None, height=dp(456))
        self.cards = {
            "status": Card("Bot Status", "INITIALIZING"),
            "pair": Card("Pair", "-"),
            "price": Card("Price", "-"),
            "signal": Card("Signal", "NEUTRAL"),
            "confidence": Card("Confidence", "0%"),
            "daily_pnl": Card("Daily PnL", "0.00 USDT"),
            "equity": Card("Equity", "0.00 USDT"),
            "balance": Card("Balance", "0.00 USDT"),
        }
        for card in self.cards.values():
            self.kpi_grid.add_widget(card)

        runtime = GradientSection()
        runtime.add_widget(Label(text="Runtime & Protections", size_hint_y=None, height=dp(24), bold=True, color=(0.88, 0.93, 1, 1)))
        self.runtime_cards = {
            "open_positions": Card("Open Positions", "0", height=dp(84)),
            "restart_count": Card("Restart Count", "0", height=dp(84)),
            "heartbeat": Card("Heartbeat Age", "0s", height=dp(84)),
            "manual_pause": Card("Manual Pause", "No", height=dp(84)),
            "kill_switch": Card("Kill Switch", "No", height=dp(84)),
            "cooldown": Card("Cooldown", "READY", height=dp(84)),
        }
        runtime_grid = GridLayout(cols=2, spacing=dp(8), size_hint_y=None, height=dp(274))
        for card in self.runtime_cards.values():
            runtime_grid.add_widget(card)
        runtime.add_widget(runtime_grid)

        self.log_card = Card("Latest Event", "Waiting events", subtitle="-")

        body.add_widget(self.kpi_grid)
        body.add_widget(runtime)
        body.add_widget(self.log_card)
        scroll.add_widget(body)
        return scroll

    def _build_market_tab(self):
        scroll = VerticalScroll()
        body = BoxLayout(orientation="vertical", spacing=dp(10), size_hint_y=None, padding=[0, dp(4), 0, dp(14)])
        body.bind(minimum_height=body.setter("height"))

        section = GradientSection()
        section.add_widget(Label(text="Market Intelligence", size_hint_y=None, height=dp(24), bold=True, color=(0.88, 0.93, 1, 1)))
        self.market_cards = {
            "trend": Card("Trend", "-", height=dp(92)),
            "adx": Card("ADX", "-", height=dp(92)),
            "spread": Card("Spread", "-", height=dp(92)),
            "volatility_regime": Card("Volatility Regime", "-", height=dp(92)),
            "order_flow": Card("Order Flow", "-", height=dp(92)),
            "market_regime": Card("Market Regime", "-", height=dp(92)),
            "volatility_state": Card("Volatility State", "-", height=dp(92)),
            "change_24h": Card("24h Change", "-", height=dp(92)),
            "atr": Card("ATR", "-", height=dp(92)),
            "distance_to_support": Card("Distance Support", "-", height=dp(92)),
            "distance_to_resistance": Card("Distance Resistance", "-", height=dp(92)),
            "timeframe": Card("Timeframe", "-", height=dp(92)),
        }
        grid = GridLayout(cols=2, spacing=dp(8), size_hint_y=None, height=dp(612))
        for card in self.market_cards.values():
            grid.add_widget(card)
        section.add_widget(grid)
        body.add_widget(section)
        scroll.add_widget(body)
        return scroll

    def _build_controls_tab(self):
        container = BoxLayout(orientation="vertical", spacing=dp(10), padding=[0, dp(6), 0, dp(12)])
        action_section = GradientSection()
        action_section.add_widget(Label(text="Bot Control Actions", size_hint_y=None, height=dp(24), bold=True, color=(0.88, 0.93, 1, 1)))
        grid = GridLayout(cols=2, spacing=dp(8), size_hint_y=None, height=dp(160))

        self.start_btn = PrimaryButton("Start Bot", tone="primary")
        self.start_btn.bind(on_release=lambda *_: self._run_action(self.client.start))
        self.pause_btn = PrimaryButton("Pause Bot", tone="warning")
        self.pause_btn.bind(on_release=lambda *_: self._run_action(self.client.pause))
        self.resume_btn = PrimaryButton("Resume Bot", tone="positive")
        self.resume_btn.bind(on_release=lambda *_: self._run_action(self.client.resume))
        self.kill_btn = PrimaryButton("Kill Switch", tone="danger")
        self.kill_btn.bind(on_release=lambda *_: self._run_action(self.client.kill_switch))
        self.close_btn = PrimaryButton("Close Active Position", tone="primary")
        self.close_btn.bind(on_release=lambda *_: self._close_active_position())

        for btn in (self.start_btn, self.pause_btn, self.resume_btn, self.kill_btn, self.close_btn):
            grid.add_widget(btn)

        action_section.add_widget(grid)
        container.add_widget(action_section)
        return container

    def _build_settings_tab(self):
        scroll = VerticalScroll()
        body = BoxLayout(orientation="vertical", spacing=dp(10), size_hint_y=None, padding=[0, dp(4), 0, dp(14)])
        body.bind(minimum_height=body.setter("height"))

        read_section = GradientSection()
        read_section.add_widget(Label(text="Core Program Settings", size_hint_y=None, height=dp(24), bold=True, color=(0.88, 0.93, 1, 1)))
        self.settings_fields = {
            "environment": LabeledField("Environment", "-", readonly=True),
            "runtime_profile": LabeledField("Runtime Profile", "-", readonly=True),
            "symbol": LabeledField("Trading Symbol", "-", readonly=True),
            "timeframe": LabeledField("Timeframe", "-", readonly=True),
            "risk_per_trade_fraction": LabeledField("Risk per trade", "-", readonly=True),
            "max_trade_balance_fraction": LabeledField("Max trade fraction", "-", readonly=True),
            "daily_loss_limit_fraction": LabeledField("Daily loss limit", "-", readonly=True),
            "max_drawdown_fraction": LabeledField("Max drawdown", "-", readonly=True),
        }
        read_grid = GridLayout(cols=1, spacing=dp(6), size_hint_y=None, height=dp(640))
        for field in self.settings_fields.values():
            read_grid.add_widget(field)
        read_section.add_widget(read_grid)

        edit_section = GradientSection()
        edit_section.add_widget(Label(text="Live Runtime Configuration", size_hint_y=None, height=dp(24), bold=True, color=(0.88, 0.93, 1, 1)))
        self.runtime_fields = {
            "investment_mode": LabeledField("Investment mode", "Balanced"),
            "risk_per_trade_fraction": LabeledField("Risk per trade fraction", "0.01"),
            "max_trade_balance_fraction": LabeledField("Max trade balance fraction", "0.20"),
            "capital_limit_usdt": LabeledField("Capital limit (USDT, 0=off)", "0"),
            "symbol_capital_limits": LabeledField("Symbol limits (BTCUSDT=1200,ETHUSDT=600)", ""),
        }
        edit_grid = GridLayout(cols=1, spacing=dp(6), size_hint_y=None, height=dp(420))
        for field in self.runtime_fields.values():
            edit_grid.add_widget(field)

        self.save_runtime_btn = PrimaryButton("Apply Runtime Configuration", tone="positive")
        self.save_runtime_btn.bind(on_release=lambda *_: self._apply_runtime_settings())

        edit_section.add_widget(edit_grid)
        edit_section.add_widget(self.save_runtime_btn)

        body.add_widget(read_section)
        body.add_widget(edit_section)
        scroll.add_widget(body)
        return scroll

    def set_connection(self, connected: bool, base_url: str = "") -> None:
        if connected:
            self.connection.text = f"CONNECTED · {base_url}"
            self.connection.color = (0.45, 0.92, 0.55, 1)
            return

        fallback = API_URL or "auto-discovery"
        self.connection.text = f"DISCONNECTED · target={fallback}"
        self.connection.color = (1, 0.4, 0.4, 1)

    def _run_action(self, action) -> None:
        result = action()
        if result.get("error"):
            self.header.text = f"Reco Trading Mobile · Error: {result['error']}"
            return
        self.header.text = "Reco Trading Mobile · Action executed"
        Clock.schedule_once(lambda *_: self.refresh(), 0.2)

    def _close_active_position(self) -> None:
        symbol = str(self._latest_runtime.get("snapshot", {}).get("pair") or "")
        if not symbol:
            self.header.text = "Reco Trading Mobile · No active symbol detected"
            return
        self._run_action(lambda: self.client.close_position(symbol))

    def _apply_runtime_settings(self) -> None:
        symbol_limits: dict[str, float] = {}
        raw_limits = self.runtime_fields["symbol_capital_limits"].input.text.strip()
        if raw_limits:
            for chunk in raw_limits.split(","):
                if "=" not in chunk:
                    continue
                key, raw_value = chunk.split("=", 1)
                try:
                    parsed = float(raw_value.strip())
                except ValueError:
                    continue
                if parsed > 0:
                    symbol_limits[key.strip().upper()] = parsed

        payload = {
            "investment_mode": self.runtime_fields["investment_mode"].input.text.strip() or "Balanced",
            "risk_per_trade_fraction": self._float_or_default(self.runtime_fields["risk_per_trade_fraction"].input.text, 0.01),
            "max_trade_balance_fraction": self._float_or_default(
                self.runtime_fields["max_trade_balance_fraction"].input.text, 0.20
            ),
            "capital_limit_usdt": self._float_or_default(self.runtime_fields["capital_limit_usdt"].input.text, 0.0),
            "symbol_capital_limits": symbol_limits,
        }
        self._run_action(lambda: self.client.apply_runtime_settings(payload))

    def refresh(self) -> None:
        runtime = self.client.runtime()
        settings = self.client.settings()

        if runtime.get("error"):
            self.cards["status"].set_value(f"OFFLINE ({runtime['error']})", color=(1, 0.45, 0.45, 1))
            return

        self._latest_runtime = runtime
        snapshot = runtime.get("snapshot", {})

        self.cards["status"].set_value(str(runtime.get("bot_status", "UNKNOWN")), color=(0.86, 0.92, 1, 1))
        self.cards["pair"].set_value(str(snapshot.get("pair", "-")))
        self.cards["price"].set_value(_num(snapshot.get("price"), 2))

        signal = str(snapshot.get("signal", "NEUTRAL")).upper()
        signal_color = (0.2, 0.85, 0.58, 1) if signal == "BUY" else (0.93, 0.37, 0.37, 1) if signal == "SELL" else (0.78, 0.82, 0.91, 1)
        self.cards["signal"].set_value(signal, color=signal_color)
        self.cards["confidence"].set_value(f"{_pct(snapshot.get('confidence'))}%")
        self.cards["daily_pnl"].set_value(f"{_num(snapshot.get('daily_pnl'), 2)} USDT")
        self.cards["equity"].set_value(f"{_num(snapshot.get('equity'), 2)} USDT")
        self.cards["balance"].set_value(f"{_num(snapshot.get('balance'), 2)} USDT")

        for key in self.market_cards:
            value = snapshot.get(key)
            if key in {"adx", "spread", "change_24h", "atr", "distance_to_support", "distance_to_resistance"}:
                self.market_cards[key].set_value(_num(value, 4 if key == "spread" else 2))
            else:
                self.market_cards[key].set_value(str(value or "-"))

        self.runtime_cards["open_positions"].set_value(str(runtime.get("open_positions", 0)))
        self.runtime_cards["restart_count"].set_value(str(runtime.get("restart_count", 0)))
        self.runtime_cards["heartbeat"].set_value(f"{_num(runtime.get('heartbeat_age_seconds'), 1)}s")
        self.runtime_cards["manual_pause"].set_value("Yes" if runtime.get("manual_pause") else "No")
        self.runtime_cards["kill_switch"].set_value("Yes" if runtime.get("kill_switch") else "No")
        self.runtime_cards["cooldown"].set_value(str(snapshot.get("cooldown", "READY")))

        logs = snapshot.get("logs") or []
        if logs:
            last = logs[-1]
            self.log_card.set_value(str(last.get("message", "-")), subtitle=str(last.get("time", "--:--")))

        if not settings.get("error"):
            for key, field in self.settings_fields.items():
                field.input.text = str(settings.get(key, "-"))

    @staticmethod
    def _float_or_default(raw: str, default: float) -> float:
        try:
            return float(raw)
        except ValueError:
            return default


def _num(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.1f}"
    except (TypeError, ValueError):
        return "0.0"
