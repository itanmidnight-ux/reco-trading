from __future__ import annotations

from pathlib import Path

from trading_system.app.backtesting.runner import BacktestExecutionConfig, HistoricalBacktestRunner
from trading_system.app.config.settings import Settings
from trading_system.app.services.decision_engine.service import Decision


def _synthetic_candles(size: int = 240) -> list[dict[str, float]]:
    candles: list[dict[str, float]] = []
    price = 100.0
    for i in range(size):
        drift = 0.0008 if i % 5 != 0 else -0.0003
        open_price = price
        close_price = price * (1 + drift)
        high = max(open_price, close_price) * 1.0012
        low = min(open_price, close_price) * 0.9988
        candles.append(
            {
                'timestamp': i,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': 1000 + i * 2,
                'bid_qty': 450 + i,
                'ask_qty': 470 + i,
            }
        )
        price = close_price
    return candles


def test_backtesting_runner_is_deterministic_with_fixed_seed(tmp_path: Path):
    candles = _synthetic_candles()
    cfg = BacktestExecutionConfig(seed=7, warmup_bars=50, hold_bars=2)

    r1 = HistoricalBacktestRunner(Settings(), config=cfg, output_dir=tmp_path / 'run1').run(candles, run_id='seeded')
    r2 = HistoricalBacktestRunner(Settings(), config=cfg, output_dir=tmp_path / 'run2').run(candles, run_id='seeded')

    assert r1.returns == r2.returns
    assert r1.trades == r2.trades
    assert r1.report.expectancy == r2.report.expectancy


def test_backtesting_runner_writes_audit_files(tmp_path: Path):
    candles = _synthetic_candles()
    runner = HistoricalBacktestRunner(
        Settings(),
        config=BacktestExecutionConfig(seed=11, warmup_bars=45, hold_bars=1),
        output_dir=tmp_path,
    )
    runner.decision_engine.decide = lambda *args, **kwargs: Decision(  # type: ignore[assignment]
        signal='LONG',
        confidence=0.9,
        score=0.9,
        expected_value=0.3,
        reason='forced-for-audit-test',
    )
    result = runner.run(candles, run_id='audit')

    assert result.signals
    assert result.trades
    assert (tmp_path / 'audit_signals.csv').exists()
    assert (tmp_path / 'audit_trades.csv').exists()
    assert (tmp_path / 'audit_config.csv').exists()
