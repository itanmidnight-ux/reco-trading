from __future__ import annotations


def normalize_symbol(symbol: str) -> str:
    """Normalize exchange symbols to CCXT's pair format when possible."""
    cleaned = str(symbol).strip().upper()
    if "/" in cleaned:
        base, quote = cleaned.split("/", 1)
        return f"{base}/{quote}"
    if cleaned.endswith("USDT"):
        base = cleaned[:-4]
        return f"{base}/USDT"
    return cleaned


def split_symbol(symbol: str) -> tuple[str, str | None]:
    """Split normalized symbol into base/quote assets."""
    normalized = normalize_symbol(symbol)
    if "/" in normalized:
        base, quote = normalized.split("/", 1)
        return base.upper(), quote.upper()
    return normalized.upper(), None
