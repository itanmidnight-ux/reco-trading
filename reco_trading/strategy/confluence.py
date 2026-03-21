from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class ConfluenceResult:
    score: float
    aligned: bool
    dominant_side: str
    notes: list[str]


class TimeframeConfluence:
    """Calcula confluencia 5m/15m como ajuste suave del confidence."""

    def evaluate(self, df5m: pd.DataFrame, df15m: pd.DataFrame) -> ConfluenceResult:
        notes: list[str] = []

        if len(df5m) < 20 or len(df15m) < 20:
            return ConfluenceResult(score=0.70, aligned=True, dominant_side="MIXED", notes=["insufficient_data"])

        row5 = df5m.iloc[-1]
        row15 = df15m.iloc[-1]

        trend5_bull = float(row5["ema20"]) > float(row5["ema50"])
        trend15_bull = float(row15["ema20"]) > float(row15["ema50"])
        trend_aligned = trend5_bull == trend15_bull

        rsi5 = float(row5["rsi"])
        rsi15 = float(row15["rsi"])
        rsi5_bull = rsi5 > 50
        rsi15_bull = rsi15 > 50
        rsi_aligned = rsi5_bull == rsi15_bull

        mom5_bull = float(row5["close"]) > float(row5["ema20"])
        mom15_bull = float(row15["close"]) > float(row15["ema20"])
        mom_aligned = mom5_bull == mom15_bull

        atr5_ratio = float(row5["atr"]) / max(float(row5["close"]), 1e-9)
        atr15_ratio = float(row15["atr"]) / max(float(row15["close"]), 1e-9)
        vol_compat = abs(atr5_ratio - atr15_ratio) < 0.01

        points = sum([trend_aligned, rsi_aligned, mom_aligned, vol_compat])
        score = 0.50 + (points / 4) * 0.50

        # Sistema de penalización diferenciada por divergencia
        divergence_penalty = 1.0

        # Trend divergence = penalización principal (35% = más agresiva)
        if not trend_aligned:
            notes.append("trend_divergence")
            dominant_side = "MIXED"
            divergence_penalty *= 0.65

            # Compensación parcial si RSI alineado (añade 15% recuperación)
            if rsi_aligned:
                divergence_penalty *= 1.15
        else:
            notes.append("trend_aligned")
            dominant_side = "BUY" if trend5_bull else "SELL"

        # RSI no alineado = penalización adicional
        if rsi_aligned:
            notes.append("rsi_aligned")
        else:
            divergence_penalty *= 0.90

        # Momentum no alineado = penalización leve
        if mom_aligned:
            notes.append("momentum_aligned")
        else:
            divergence_penalty *= 0.95

        # Volatilidad no compatible = penalización leve
        if not vol_compat:
            notes.append("volatility_divergence")
            divergence_penalty *= 0.95

        # Aplicar penalización combinada y normalizar resultado
        score *= divergence_penalty
        score = round(min(max(score, 0.0), 1.0), 4)

        return ConfluenceResult(
            score=round(min(max(score, 0.0), 1.0), 4),
            aligned=trend_aligned and rsi_aligned,
            dominant_side=dominant_side,
            notes=notes,
        )
