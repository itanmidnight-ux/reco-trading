from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class OrderFlowDecision:
    signal: str
    buy_pressure: float
    sell_pressure: float


class OrderFlowAnalyzer:
    """Approximates order flow pressure from candle direction and volume."""

    def __init__(self, lookback: int = 20) -> None:
        self.lookback = lookback

    def evaluate(self, frame: pd.DataFrame) -> OrderFlowDecision:
        """
        Calcula presión de order flow ponderada por magnitud del move.

        ALGORITMO:
        1. Extraer últimas N velas (lookback)
        2. Separar en barras UP (close > open) y DOWN (close < open)
        3. Para cada tipo: calcular (magnitud × volumen) normalizado por ATR
        4. Escalar presiones a rango [0, 1] usando fórmula normalizada
        5. Aplicar threshold 65% (más conservador que antes)
        6. Retornar OrderFlowDecision con signal y presiones

        Returns:
            OrderFlowDecision: señal + presiones de compra/venta
        """
        recent = frame.tail(self.lookback)
        if recent.empty:
            return OrderFlowDecision(signal="NEUTRAL", buy_pressure=0.5, sell_pressure=0.5)

        # Separar barras alcistas y bajistas
        up_bars = recent[recent["close"] > recent["open"]]
        down_bars = recent[recent["close"] < recent["open"]]

        # Obtener ATR actual para normalización (fallback a 1.0 si no existe)
        atr_current = float(recent["atr"].iloc[-1]) if "atr" in recent.columns else 1.0
        atr_safe = max(atr_current, 1e-9)

        # Calcular presión ponderada para barras UP
        if len(up_bars) > 0:
            up_size = up_bars["close"] - up_bars["open"]
            normalized_up_size = up_size / atr_safe
            up_pressure_weighted = float((normalized_up_size * up_bars["volume"]).sum())
        else:
            up_pressure_weighted = 0.0

        # Calcular presión ponderada para barras DOWN
        if len(down_bars) > 0:
            down_size = down_bars["open"] - down_bars["close"]
            normalized_down_size = down_size / atr_safe
            down_pressure_weighted = float((normalized_down_size * down_bars["volume"]).sum())
        else:
            down_pressure_weighted = 0.0

        total_pressure = up_pressure_weighted + down_pressure_weighted

        # Si no hay presión detectable, retornar NEUTRAL
        if total_pressure <= 0:
            return OrderFlowDecision(signal="NEUTRAL", buy_pressure=0.5, sell_pressure=0.5)

        # Escalar a rango [0, 1] usando fórmula de normalización
        net_pressure = up_pressure_weighted - down_pressure_weighted
        buy_pressure = 0.5 + (net_pressure / (2.0 * max(total_pressure, 1e-9)))

        # Clamp a rango válido [0, 1]
        buy_pressure = max(0.0, min(1.0, buy_pressure))
        sell_pressure = 1.0 - buy_pressure

        # Threshold balanceado: requiere 60% para signal fuerte.
        if buy_pressure > 0.60:
            signal = "BUY"
        elif buy_pressure < 0.40:
            signal = "SELL"
        else:
            signal = "NEUTRAL"

        return OrderFlowDecision(signal=signal, buy_pressure=buy_pressure, sell_pressure=sell_pressure)
