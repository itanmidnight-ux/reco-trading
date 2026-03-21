import pandas as pd

from reco_trading.strategy.confluence import TimeframeConfluence


def _frame(*, close: float, ema20: float, ema50: float, rsi: float, atr: float, size: int = 25) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": [close] * size,
            "ema20": [ema20] * size,
            "ema50": [ema50] * size,
            "rsi": [rsi] * size,
            "atr": [atr] * size,
        }
    )


def test_aligned_timeframes_score_above_threshold() -> None:
    confluence = TimeframeConfluence()
    df5m = _frame(close=105.0, ema20=103.0, ema50=100.0, rsi=60.0, atr=1.0)
    df15m = _frame(close=210.0, ema20=206.0, ema50=200.0, rsi=58.0, atr=2.0)

    result = confluence.evaluate(df5m, df15m)

    assert result.score > 0.75
    assert result.aligned is True


def test_divergent_timeframes_reduce_score() -> None:
    confluence = TimeframeConfluence()
    df5m = _frame(close=105.0, ema20=103.0, ema50=100.0, rsi=60.0, atr=1.0)
    df15m = _frame(close=195.0, ema20=198.0, ema50=200.0, rsi=45.0, atr=2.0)

    result = confluence.evaluate(df5m, df15m)

    assert result.score < 0.75
    assert result.aligned is False


def test_insufficient_data_returns_safe_default() -> None:
    confluence = TimeframeConfluence()
    df5m = _frame(close=105.0, ema20=103.0, ema50=100.0, rsi=60.0, atr=1.0, size=5)
    df15m = _frame(close=210.0, ema20=206.0, ema50=200.0, rsi=58.0, atr=2.0, size=5)

    result = confluence.evaluate(df5m, df15m)

    assert result.score == 0.70
    assert result.aligned is True
    assert result.dominant_side == "MIXED"


def test_confluence_trend_divergence_penalty() -> None:
    """Verifica penalización -35% para divergencia trend."""
    confluence = TimeframeConfluence()
    df5m = _frame(close=105.0, ema20=103.0, ema50=100.0, rsi=60.0, atr=1.0)
    df15m = _frame(close=195.0, ema20=198.0, ema50=200.0, rsi=45.0, atr=2.0)

    result = confluence.evaluate(df5m, df15m)

    assert result.score < 0.60
    assert result.aligned is False
    assert result.dominant_side == "MIXED"
    assert all(isinstance(note, str) for note in result.notes)
