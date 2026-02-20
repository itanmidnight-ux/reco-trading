from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class ExecutionQualityMetrics:
    route: str
    filled_qty: float
    requested_qty: float
    fill_ratio: float
    avg_fill_price: float
    slippage_bps: float
    adverse_selection_cost_bps: float
    avg_confirmation_latency_ms: float


class DarkPoolSimulator:
    """Modelo simplificado de liquidez oculta con API de comparación de rutas para SOR."""

    def __init__(
        self,
        base_fill_sensitivity: float = 0.85,
        volatility_penalty: float = 3.0,
        tif_scale_seconds: float = 90.0,
        adverse_drift_scale: float = 12.0,
        spread_decay_halflife_ms: float = 60.0,
    ) -> None:
        self.base_fill_sensitivity = base_fill_sensitivity
        self.volatility_penalty = volatility_penalty
        self.tif_scale_seconds = max(tif_scale_seconds, 1.0)
        self.adverse_drift_scale = adverse_drift_scale
        self.spread_decay_halflife_ms = max(spread_decay_halflife_ms, 1.0)
        self._history: list[dict[str, float | str]] = []

    def hidden_fill_probability(
        self,
        order_qty: float,
        visible_volume: float,
        volatility: float,
        time_in_force_seconds: float,
    ) -> float:
        volume_ratio = max(visible_volume, 1e-9) / max(order_qty, 1e-9)
        tif_factor = 1.0 - np.exp(-max(time_in_force_seconds, 0.0) / self.tif_scale_seconds)
        raw_score = (
            self.base_fill_sensitivity * np.log1p(volume_ratio)
            - self.volatility_penalty * max(volatility, 0.0)
            + 1.4 * tif_factor
        )
        return float(np.clip(1.0 / (1.0 + np.exp(-raw_score)), 0.01, 0.99))

    def compare_routes(
        self,
        order_qty: float,
        side: str,
        mid_price: float,
        visible_volume: float,
        volatility: float,
        spread_bps: float,
        time_in_force_seconds: float,
        hybrid_dark_ratio: float = 0.5,
        seed: int | None = None,
    ) -> dict[str, ExecutionQualityMetrics]:
        rng = np.random.default_rng(seed)
        side = side.upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")

        dark = self._simulate_dark_route(
            route="dark-only",
            order_qty=order_qty,
            side=side,
            mid_price=mid_price,
            visible_volume=visible_volume,
            volatility=volatility,
            spread_bps=spread_bps,
            time_in_force_seconds=time_in_force_seconds,
            rng=rng,
        )

        visible = self._simulate_visible_route(
            route="visible-only",
            order_qty=order_qty,
            side=side,
            mid_price=mid_price,
            visible_volume=visible_volume,
            volatility=volatility,
            spread_bps=spread_bps,
            rng=rng,
        )

        dark_ratio = float(np.clip(hybrid_dark_ratio, 0.0, 1.0))
        dark_leg = self._simulate_dark_route(
            route="hybrid-dark-leg",
            order_qty=order_qty * dark_ratio,
            side=side,
            mid_price=mid_price,
            visible_volume=visible_volume,
            volatility=volatility,
            spread_bps=spread_bps,
            time_in_force_seconds=time_in_force_seconds,
            rng=rng,
        )
        visible_leg = self._simulate_visible_route(
            route="hybrid-visible-leg",
            order_qty=order_qty * (1.0 - dark_ratio),
            side=side,
            mid_price=mid_price,
            visible_volume=visible_volume,
            volatility=volatility,
            spread_bps=spread_bps,
            rng=rng,
        )

        hybrid = self._combine_metrics("hybrid-split", order_qty, dark_leg, visible_leg)
        routes = {"visible-only": visible, "dark-only": dark, "hybrid-split": hybrid}

        for route_name, metrics in routes.items():
            record = asdict(metrics)
            record.update(
                {
                    "side": side,
                    "mid_price": float(mid_price),
                    "volatility": float(volatility),
                    "spread_bps": float(spread_bps),
                    "time_in_force_seconds": float(time_in_force_seconds),
                    "route": route_name,
                }
            )
            self._history.append(record)

        return routes

    def metrics_history(self) -> pd.DataFrame:
        if not self._history:
            return pd.DataFrame(
                columns=[
                    "route",
                    "side",
                    "filled_qty",
                    "requested_qty",
                    "fill_ratio",
                    "slippage_bps",
                    "adverse_selection_cost_bps",
                    "avg_confirmation_latency_ms",
                ]
            )
        return pd.DataFrame(self._history)

    def _simulate_dark_route(
        self,
        route: str,
        order_qty: float,
        side: str,
        mid_price: float,
        visible_volume: float,
        volatility: float,
        spread_bps: float,
        time_in_force_seconds: float,
        rng: np.random.Generator,
    ) -> ExecutionQualityMetrics:
        if order_qty <= 0:
            return self._empty_metrics(route, order_qty, mid_price)

        fill_prob = self.hidden_fill_probability(order_qty, visible_volume, volatility, time_in_force_seconds)
        target_fill = order_qty * fill_prob

        partials: list[float] = []
        remaining = target_fill
        while remaining > 1e-9:
            clip = float(min(remaining, max(order_qty * rng.uniform(0.05, 0.35), 1e-9)))
            partials.append(clip)
            remaining -= clip

        filled_qty = float(sum(partials))
        fill_ratio = float(np.clip(filled_qty / order_qty, 0.0, 1.0))

        sign = 1.0 if side == "BUY" else -1.0
        spread_component = spread_bps * 0.35
        slippage_noise = rng.normal(0.0, max(0.2, spread_bps * 0.02))
        slippage_bps = sign * spread_component + slippage_noise

        latencies = rng.lognormal(mean=3.3, sigma=0.25, size=max(len(partials), 1))
        avg_latency_ms = float(latencies.mean())
        spread_capture_decay = np.exp(-avg_latency_ms / self.spread_decay_halflife_ms)

        drift_post_fill_bps = abs(rng.normal(loc=volatility * self.adverse_drift_scale, scale=0.4))
        adverse_selection_cost_bps = float(drift_post_fill_bps + spread_component * (1.0 - spread_capture_decay))

        avg_fill_price = float(mid_price * (1.0 + (slippage_bps / 10000.0)))
        return ExecutionQualityMetrics(
            route=route,
            filled_qty=filled_qty,
            requested_qty=order_qty,
            fill_ratio=fill_ratio,
            avg_fill_price=avg_fill_price,
            slippage_bps=float(slippage_bps),
            adverse_selection_cost_bps=adverse_selection_cost_bps,
            avg_confirmation_latency_ms=avg_latency_ms,
        )

    def _simulate_visible_route(
        self,
        route: str,
        order_qty: float,
        side: str,
        mid_price: float,
        visible_volume: float,
        volatility: float,
        spread_bps: float,
        rng: np.random.Generator,
    ) -> ExecutionQualityMetrics:
        if order_qty <= 0:
            return self._empty_metrics(route, order_qty, mid_price)

        liquidity_ratio = np.clip(visible_volume / max(order_qty, 1e-9), 0.0, 1.5)
        fill_ratio = float(np.clip(0.6 + 0.4 * liquidity_ratio - 1.8 * volatility, 0.05, 1.0))
        filled_qty = float(order_qty * fill_ratio)

        sign = 1.0 if side == "BUY" else -1.0
        slippage_bps = float(sign * spread_bps * 0.9 + abs(rng.normal(0.0, spread_bps * 0.04)))
        adverse_selection_cost_bps = float(max(0.1, volatility * self.adverse_drift_scale * 0.75))
        avg_latency_ms = float(max(1.0, rng.normal(8.0, 1.5)))
        avg_fill_price = float(mid_price * (1.0 + slippage_bps / 10000.0))

        return ExecutionQualityMetrics(
            route=route,
            filled_qty=filled_qty,
            requested_qty=order_qty,
            fill_ratio=fill_ratio,
            avg_fill_price=avg_fill_price,
            slippage_bps=slippage_bps,
            adverse_selection_cost_bps=adverse_selection_cost_bps,
            avg_confirmation_latency_ms=avg_latency_ms,
        )

    def _combine_metrics(
        self,
        route: str,
        total_order_qty: float,
        a: ExecutionQualityMetrics,
        b: ExecutionQualityMetrics,
    ) -> ExecutionQualityMetrics:
        filled_qty = a.filled_qty + b.filled_qty
        weight_a = a.filled_qty / max(filled_qty, 1e-9)
        weight_b = 1.0 - weight_a

        return ExecutionQualityMetrics(
            route=route,
            filled_qty=filled_qty,
            requested_qty=total_order_qty,
            fill_ratio=float(np.clip(filled_qty / max(total_order_qty, 1e-9), 0.0, 1.0)),
            avg_fill_price=float(a.avg_fill_price * weight_a + b.avg_fill_price * weight_b),
            slippage_bps=float(a.slippage_bps * weight_a + b.slippage_bps * weight_b),
            adverse_selection_cost_bps=float(
                a.adverse_selection_cost_bps * weight_a + b.adverse_selection_cost_bps * weight_b
            ),
            avg_confirmation_latency_ms=float(
                a.avg_confirmation_latency_ms * weight_a + b.avg_confirmation_latency_ms * weight_b
            ),
        )

    @staticmethod
    def _empty_metrics(route: str, requested_qty: float, mid_price: float) -> ExecutionQualityMetrics:
        return ExecutionQualityMetrics(
            route=route,
            filled_qty=0.0,
            requested_qty=float(requested_qty),
            fill_ratio=0.0,
            avg_fill_price=float(mid_price),
            slippage_bps=0.0,
            adverse_selection_cost_bps=0.0,
            avg_confirmation_latency_ms=0.0,
        )
