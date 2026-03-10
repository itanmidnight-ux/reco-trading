from __future__ import annotations


def normalize_symbol(symbol: str) -> str:
    """Normalize exchange symbols to CCXT's pair format when possible."""
    if "/" in symbol:
        return symbol
    if symbol.endswith("USDT"):
        base = symbol[:-4]
        return f"{base}/USDT"
    return symbol
