from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reco_trading.kernel.capital_governor import CapitalGovernor, CapitalTicket


@dataclass(slots=True)
class VenueSnapshot:
    venue: str
    spread_bps: float
    depth: float
    latency_ms: float
    fee_bps: float
    fill_ratio: float
    liquidity: float


class VenueScoreModel:
    def __init__(
        self,
        spread_weight: float = 0.30,
        depth_weight: float = 0.20,
        latency_weight: float = 0.20,
        fee_weight: float = 0.15,
        fill_ratio_weight: float = 0.15,
    ) -> None:
        self.weights = {
            'spread': spread_weight,
            'depth': depth_weight,
            'latency': latency_weight,
            'fees': fee_weight,
            'fill_ratio': fill_ratio_weight,
        }

    @staticmethod
    def _normalize(values: list[float], inverse: bool = False) -> list[float]:
        if not values:
            return []
        arr = np.asarray(values, dtype=float)
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        if hi - lo < 1e-12:
            norm = np.ones_like(arr) * 0.5
        else:
            norm = (arr - lo) / (hi - lo)
        if inverse:
            norm = 1.0 - norm
        return norm.tolist()

    def score(self, venues: list[VenueSnapshot], epsilon: float = 0.02) -> dict[str, float]:
        spreads = self._normalize([v.spread_bps for v in venues], inverse=True)
        depths = self._normalize([v.depth for v in venues], inverse=False)
        latencies = self._normalize([v.latency_ms for v in venues], inverse=True)
        fees = self._normalize([v.fee_bps for v in venues], inverse=True)
        fills = self._normalize([v.fill_ratio for v in venues], inverse=False)

        scores: dict[str, float] = {}
        for idx, venue in enumerate(venues):
            raw_score = (
                self.weights['spread'] * spreads[idx]
                + self.weights['depth'] * depths[idx]
                + self.weights['latency'] * latencies[idx]
                + self.weights['fees'] * fees[idx]
                + self.weights['fill_ratio'] * fills[idx]
            )
            scores[venue.venue] = float(raw_score)

        if not scores:
            return scores

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if len(sorted_scores) > 1 and (sorted_scores[0][1] - sorted_scores[1][1]) <= epsilon:
            winner = min(venues, key=lambda v: v.latency_ms)
            scores[winner.venue] += epsilon / 2
        return scores


class OrderSplitter:
    def split(
        self,
        amount: float,
        strategy: str,
        slices: int = 5,
        expected_volume_profile: list[float] | None = None,
        iceberg_peak_ratio: float = 0.20,
    ) -> list[float]:
        if amount <= 0:
            return []
        strategy = strategy.upper()
        if strategy == 'TWAP':
            return [amount / slices for _ in range(slices)]
        if strategy == 'VWAP':
            profile = expected_volume_profile or [1.0 for _ in range(slices)]
            norm = np.asarray(profile, dtype=float)
            norm = np.clip(norm, 1e-9, None)
            norm /= np.sum(norm)
            return (norm * amount).tolist()
        if strategy == 'ICEBERG':
            peak = max(amount * iceberg_peak_ratio, amount / slices)
            chunks: list[float] = []
            remaining = amount
            while remaining > 1e-9:
                child = min(peak, remaining)
                chunks.append(float(child))
                remaining -= child
            return chunks
        raise ValueError(f'Estrategia de split desconocida: {strategy}')


class ImpactModel:
    def __init__(self, lambda_impact: float = 0.5) -> None:
        self.lambda_impact = lambda_impact

    def estimate(self, order_size: float, liquidity: float) -> float:
        safe_liquidity = max(liquidity, 1e-9)
        return float(self.lambda_impact * (order_size / safe_liquidity))


class SmartOrderRouter:
    def __init__(
        self,
        score_model: VenueScoreModel | None = None,
        splitter: OrderSplitter | None = None,
        impact_model: ImpactModel | None = None,
        epsilon: float = 0.02,
        capital_governor: CapitalGovernor | None = None,
    ) -> None:
        self.score_model = score_model or VenueScoreModel()
        self.splitter = splitter or OrderSplitter()
        self.impact_model = impact_model or ImpactModel()
        self.epsilon = epsilon
        self.capital_governor = capital_governor

    def _solve_allocation(
        self,
        child_amount: float,
        venues: list[VenueSnapshot],
        fee_weight: float,
        risk_limit: float,
    ) -> dict[str, float]:
        try:
            import cvxpy as cp

            liquidities = np.asarray([max(v.liquidity, 1e-9) for v in venues], dtype=float)
            fees = np.asarray([max(v.fee_bps, 0.0) / 10_000.0 for v in venues], dtype=float)
            alloc = cp.Variable(len(venues), nonneg=True)
            impact_cost = cp.sum(cp.multiply(self.impact_model.lambda_impact / liquidities, cp.square(alloc)))
            fee_cost = fee_weight * cp.sum(cp.multiply(fees, alloc))
            risk_penalty = risk_limit * cp.sum(cp.square(alloc / np.maximum(liquidities, 1e-9)))
            objective = cp.Minimize(impact_cost + fee_cost + risk_penalty)
            constraints = [
                cp.sum(alloc) == child_amount,
                alloc <= liquidities,
            ]
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.SCS, verbose=False)
            if alloc.value is None:
                raise RuntimeError('Solver sin solución')
            solved = np.clip(np.asarray(alloc.value, dtype=float), 0.0, None)
        except Exception:
            weights = np.asarray([max(v.liquidity, 1e-9) / (1 + v.fee_bps) for v in venues], dtype=float)
            weights /= max(np.sum(weights), 1e-9)
            solved = weights * child_amount

        solved *= child_amount / max(np.sum(solved), 1e-9)
        return {venue.venue: float(qty) for venue, qty in zip(venues, solved)}

    def route_order(
        self,
        amount: float,
        venues: list[VenueSnapshot],
        strategy: str = 'TWAP',
        slices: int = 5,
        expected_volume_profile: list[float] | None = None,
        fee_weight: float = 1.0,
        risk_limit: float = 0.25,
        capital_ticket: CapitalTicket | None = None,
    ) -> list[dict[str, float | str | int]]:
        if amount <= 0:
            return []
        if self.capital_governor is not None:
            valid_ticket, reason = self.capital_governor.validate_ticket(capital_ticket, min_notional=amount)
            if not valid_ticket:
                raise ValueError(f'capital_ticket_invalid:{reason}')
        if not venues:
            raise ValueError('Se requieren venues para rutear la orden')

        venue_scores = self.score_model.score(venues, epsilon=self.epsilon)
        sorted_venues = sorted(venues, key=lambda v: venue_scores.get(v.venue, 0.0), reverse=True)

        child_orders = self.splitter.split(
            amount=amount,
            strategy=strategy,
            slices=slices,
            expected_volume_profile=expected_volume_profile,
        )

        routed: list[dict[str, float | str | int]] = []
        for idx, child_amount in enumerate(child_orders, start=1):
            allocation = self._solve_allocation(child_amount, sorted_venues, fee_weight=fee_weight, risk_limit=risk_limit)
            for venue in sorted_venues:
                qty = allocation.get(venue.venue, 0.0)
                if qty <= 1e-9:
                    continue
                routed.append(
                    {
                        'child_id': idx,
                        'venue': venue.venue,
                        'amount': qty,
                        'score': float(venue_scores.get(venue.venue, 0.0)),
                        'estimated_impact': self.impact_model.estimate(qty, venue.liquidity),
                    }
                )
        return routed
