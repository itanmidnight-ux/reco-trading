from __future__ import annotations

from typing import Any


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_text(value: Any, default: str = "--") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def fmt_price(value: Any, digits: int = 2) -> str:
    return f"{to_float(value):,.{digits}f}"


def fmt_number(value: Any, digits: int = 2, default: str = "--") -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return default


def fmt_pct(value: Any, digits: int = 2) -> str:
    val = to_float(value)
    if val <= 1:
        val *= 100
    return f"{val:.{digits}f}%"


def trend_arrow(trend: str) -> str:
    t = (trend or "").upper()
    if t in {"BUY", "UP", "BULLISH", "LONG"}:
        return "▲"
    if t in {"SELL", "DOWN", "BEARISH", "SHORT"}:
        return "▼"
    return "■"
