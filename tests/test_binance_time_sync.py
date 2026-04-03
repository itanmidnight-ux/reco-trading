from __future__ import annotations

import asyncio

import ccxt

from reco_trading.exchange.binance_client import BinanceClient


def test_exchange_initialization_has_time_safety_options() -> None:
    client = BinanceClient(api_key="", api_secret="", testnet=True)
    assert client.exchange.enableRateLimit is True
    assert client.exchange.options.get("adjustForTimeDifference") is True
    assert client.exchange.options.get("recvWindow") == 10000


def test_safe_exchange_call_resyncs_and_retries_once_on_timestamp_error() -> None:
    client = BinanceClient(api_key="", api_secret="", testnet=True)
    calls = {"fn": 0, "sync": 0}

    async def fake_sync(*, reason: str = "manual") -> None:
        calls["sync"] += 1

    client.sync_exchange_time = fake_sync  # type: ignore[assignment]

    def flaky() -> str:
        calls["fn"] += 1
        if calls["fn"] == 1:
            raise ccxt.InvalidNonce('binance {"code":-1021,"msg":"Timestamp for this request is outside of the recvWindow."}')
        return "ok"

    result = asyncio.run(client.safe_exchange_call(flaky, operation="unit_test", retries=3))
    assert result == "ok"
    assert calls["fn"] == 2
    assert calls["sync"] == 1


def test_safe_exchange_call_raises_if_timestamp_error_repeats() -> None:
    client = BinanceClient(api_key="", api_secret="", testnet=True)
    calls = {"sync": 0}

    async def fake_sync(*, reason: str = "manual") -> None:
        calls["sync"] += 1

    client.sync_exchange_time = fake_sync  # type: ignore[assignment]

    def always_bad() -> None:
        raise ccxt.ExchangeError('binance {"code":-1021,"msg":"Timestamp for this request is outside of the recvWindow."}')

    try:
        asyncio.run(client.safe_exchange_call(always_bad, operation="unit_test", retries=3))
    except ccxt.ExchangeError:
        pass
    else:
        raise AssertionError("Expected ccxt.ExchangeError")

    # Only one forced resync/retry for timestamp errors.
    assert calls["sync"] == 1


def test_safe_exchange_call_cancels_when_thread_executor_is_shutdown() -> None:
    client = BinanceClient(api_key="", api_secret="", testnet=True)

    def executor_shutdown() -> None:
        raise RuntimeError("cannot schedule new futures after shutdown")

    try:
        asyncio.run(client.safe_exchange_call(executor_shutdown, operation="unit_test", retries=3))
    except asyncio.CancelledError as exc:
        assert str(exc) == "executor_shutdown"
    else:
        raise AssertionError("Expected asyncio.CancelledError")
