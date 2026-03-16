from __future__ import annotations


def normalize_symbol(symbol: str) -> str:
    """Normalize exchange symbols to CCXT's pair format when possible."""
    if "/" in symbol:
        return symbol
    if symbol.endswith("USDT"):
        base = symbol[:-4]
        return f"{base}/USDT"
    return symbol


def split_symbol(symbol: str) -> tuple[str, str | None]:
    """Split normalized symbol into base/quote assets."""
    normalized = normalize_symbol(symbol)
    if "/" in normalized:
        base, quote = normalized.split("/", 1)
        return base.upper(), quote.upper()
    return normalized.upper(), None
