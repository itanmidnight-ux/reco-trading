from __future__ import annotations

from reco_trading.core.bot_engine import _sanitize_runtime_settings_payload


def test_runtime_settings_sanitization_normalizes_symbol_limits_and_strips_unknowns() -> None:
    payload = _sanitize_runtime_settings_payload(
        {
            "investment_mode": "Custom",
            "theme": "Neon",
            "language": "Deutsch",
            "capital_limit_usdt": 500,
            "capital_reserve_ratio": 1.5,
            "min_cash_buffer_usdt": -10,
            "risk_per_trade_fraction": 0.5,
            "max_trade_balance_fraction": 2.0,
            "symbol_capital_limits": {"BTCUSDT": 125.0, "ETH/USDT": -1, "": 10},
            "binance_api_key": "should_not_pass",
            "binance_api_secret": "should_not_pass",
        }
    )

    assert payload["investment_mode"] == "Custom"
    assert payload["theme"] == "Dark"
    assert payload["language"] == "English"
    assert payload["capital_limit_usdt"] == 500.0
    assert payload["capital_reserve_ratio"] == 0.90
    assert payload["min_cash_buffer_usdt"] == 0.0
    assert payload["risk_per_trade_fraction"] == 0.10
    assert payload["max_trade_balance_fraction"] == 1.0
    assert payload["symbol_capital_limits"] == {"BTC/USDT": 125.0}
    assert "binance_api_key" not in payload
    assert "binance_api_secret" not in payload
