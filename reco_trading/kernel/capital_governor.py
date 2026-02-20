from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import numpy as np

from reco_trading.monitoring.alert_manager import AlertManager
from reco_trading.monitoring.metrics import TradingMetrics


@dataclass(slots=True)
class CapitalTicket:
    ticket_id: str
    issued_at: datetime
    expires_at: datetime
    strategy: str
    exchange: str
    symbol: str
    requested_notional: float
    approved_notional: float
    status: str = 'approved'
    reason: str = 'ok'


@dataclass(slots=True)
class CapitalGovernorState:
    capital_by_strategy: dict[str, float] = field(default_factory=dict)
    capital_by_exchange: dict[str, float] = field(default_factory=dict)
    total_exposure: float = 0.0
    exposure_by_asset: dict[str, float] = field(default_factory=dict)
    hard_cap_global: float = float('inf')


@dataclass(slots=True)
class TailRiskSnapshot:
    extreme_loss_count: int
    volatility_cluster_ratio: float
    breached: bool
    reason: str


@dataclass(slots=True)
class IntradayStressResult:
    spread_shock_loss: float
    liquidity_shock_loss: float
    price_gap_loss: float
    total_stress_loss: float
    breached: bool


class CapitalGovernor:
    def __init__(
        self,
        *,
        hard_cap_global: float,
        ticket_ttl_seconds: int = 30,
        metrics: TradingMetrics | None = None,
        alert_manager: AlertManager | None = None,
        labels: dict[str, str] | None = None,
    ) -> None:
        self.state = CapitalGovernorState(hard_cap_global=float(max(hard_cap_global, 0.0)))
        self.ticket_ttl_seconds = max(int(ticket_ttl_seconds), 1)
        self.metrics = metrics
        self.alert_manager = alert_manager
        self.labels = labels or {}
        self._tickets: dict[str, CapitalTicket] = {}

    def update_state(
        self,
        *,
        strategy: str,
        exchange: str,
        symbol: str,
        capital_by_strategy: float,
        capital_by_exchange: float,
        total_exposure: float,
        asset_exposure: float,
    ) -> None:
        self.state.capital_by_strategy[strategy] = float(capital_by_strategy)
        self.state.capital_by_exchange[exchange] = float(capital_by_exchange)
        self.state.total_exposure = float(total_exposure)
        self.state.exposure_by_asset[symbol] = float(asset_exposure)

    @staticmethod
    def rolling_var_cvar(pnl_or_returns: list[float] | np.ndarray, alpha: float = 0.95) -> tuple[float, float]:
        arr = np.asarray(pnl_or_returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 0.0, 0.0

        losses = -arr
        var = float(np.quantile(losses, alpha))
        tail = losses[losses > var]
        cvar = var if tail.size == 0 else float(np.mean(tail))
        return max(var, 0.0), max(cvar, 0.0)

    def monitor_tail_risk(
        self,
        pnl_or_returns: list[float] | np.ndarray,
        *,
        extreme_loss_sigma: float = 2.5,
        cluster_window: int = 20,
        cluster_threshold: float = 0.60,
    ) -> TailRiskSnapshot:
        arr = np.asarray(pnl_or_returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return TailRiskSnapshot(0, 0.0, False, 'insufficient_data')

        mu = float(np.mean(arr))
        sigma = float(np.std(arr))
        loss_cutoff = mu - extreme_loss_sigma * max(sigma, 1e-12)
        extreme_count = int(np.sum(arr < loss_cutoff))

        if arr.size < max(cluster_window, 3):
            cluster_ratio = 0.0
        else:
            rolling_vol = np.array(
                [
                    float(np.std(arr[idx - cluster_window : idx]))
                    for idx in range(cluster_window, len(arr) + 1)
                ],
                dtype=float,
            )
            vol_cutoff = float(np.quantile(rolling_vol, 0.80)) if rolling_vol.size else 0.0
            cluster_ratio = float(np.mean(rolling_vol >= vol_cutoff)) if rolling_vol.size else 0.0

        breached = extreme_count > 0 and cluster_ratio >= cluster_threshold
        reason = 'tail_risk_clustered' if breached else 'ok'
        return TailRiskSnapshot(extreme_count, cluster_ratio, breached, reason)

    @staticmethod
    def run_intraday_stress_test(
        *,
        notional: float,
        spread_bps: float,
        available_liquidity: float,
        price_gap_pct: float,
        spread_shock_factor: float = 2.0,
        liquidity_shock_factor: float = 0.35,
    ) -> IntradayStressResult:
        safe_notional = max(float(notional), 0.0)
        spread_loss = safe_notional * ((spread_bps / 10_000.0) * max(spread_shock_factor, 0.0))

        safe_liquidity = max(float(available_liquidity), 1e-9)
        liquidity_shortfall = max(safe_notional - safe_liquidity * (1.0 - liquidity_shock_factor), 0.0)
        liquidity_loss = liquidity_shortfall * 0.01

        gap_loss = safe_notional * max(price_gap_pct, 0.0)
        total = spread_loss + liquidity_loss + gap_loss
        breached = total > safe_notional * 0.05
        return IntradayStressResult(spread_loss, liquidity_loss, gap_loss, total, breached)

    def issue_ticket(
        self,
        *,
        strategy: str,
        exchange: str,
        symbol: str,
        requested_notional: float,
        pnl_or_returns: list[float] | np.ndarray,
        spread_bps: float,
        available_liquidity: float,
        price_gap_pct: float,
    ) -> CapitalTicket:
        request = max(float(requested_notional), 0.0)
        var_95, cvar_95 = self.rolling_var_cvar(pnl_or_returns, alpha=0.95)
        tail = self.monitor_tail_risk(pnl_or_returns)
        stress = self.run_intraday_stress_test(
            notional=request,
            spread_bps=spread_bps,
            available_liquidity=available_liquidity,
            price_gap_pct=price_gap_pct,
        )

        current_total = abs(self.state.total_exposure)
        hard_cap_remaining = max(self.state.hard_cap_global - current_total, 0.0)
        risk_scale = float(np.clip(1.0 - (cvar_95 * 10.0), 0.1, 1.0))
        stress_scale = 0.5 if stress.breached else 1.0
        tail_scale = 0.4 if tail.breached else 1.0
        approved = min(request * risk_scale * stress_scale * tail_scale, hard_cap_remaining)

        reason = 'ok'
        status = 'approved'
        if approved <= 0:
            status = 'rejected'
            if hard_cap_remaining <= 0:
                reason = 'hard_cap_global_hit'
            elif tail.breached:
                reason = tail.reason
            elif stress.breached:
                reason = 'intraday_stress_breached'
            else:
                reason = 'risk_scaled_to_zero'

        now = datetime.now(timezone.utc)
        ticket = CapitalTicket(
            ticket_id=str(uuid4()),
            issued_at=now,
            expires_at=now + timedelta(seconds=self.ticket_ttl_seconds),
            strategy=strategy,
            exchange=exchange,
            symbol=symbol,
            requested_notional=request,
            approved_notional=float(approved),
            status=status,
            reason=reason,
        )
        self._tickets[ticket.ticket_id] = ticket
        self._record_decision(ticket=ticket, var_95=var_95, cvar_95=cvar_95, tail=tail, stress=stress)
        return ticket

    def validate_ticket(self, ticket: CapitalTicket | None, *, min_notional: float = 0.0) -> tuple[bool, str]:
        if ticket is None:
            return False, 'missing_ticket'
        stored = self._tickets.get(ticket.ticket_id)
        if stored is None:
            return False, 'unknown_ticket'
        if stored.status != 'approved':
            return False, f'invalid_ticket_status:{stored.status}'
        if datetime.now(timezone.utc) > stored.expires_at:
            return False, 'ticket_expired'
        if stored.approved_notional < max(min_notional, 0.0):
            return False, 'insufficient_approved_notional'
        return True, 'ok'

    def _record_decision(
        self,
        *,
        ticket: CapitalTicket,
        var_95: float,
        cvar_95: float,
        tail: TailRiskSnapshot,
        stress: IntradayStressResult,
    ) -> None:
        payload: dict[str, Any] = {
            'ticket_id': ticket.ticket_id,
            'strategy': ticket.strategy,
            'exchange': ticket.exchange,
            'symbol': ticket.symbol,
            'requested_notional': ticket.requested_notional,
            'approved_notional': ticket.approved_notional,
            'status': ticket.status,
            'reason': ticket.reason,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'tail_risk_breached': tail.breached,
            'tail_cluster_ratio': tail.volatility_cluster_ratio,
            'stress_total_loss': stress.total_stress_loss,
            'stress_breached': stress.breached,
        }

        if self.metrics:
            if ticket.status == 'approved':
                self.metrics.observe_request('capital_governor', **self.labels)
            else:
                self.metrics.observe_error('capital_governor', ticket.reason, **self.labels)

        if self.alert_manager and ticket.status != 'approved':
            self.alert_manager.emit(
                'capital governor rejection',
                f"Ticket {ticket.ticket_id} rechazado: {ticket.reason}",
                severity='warning',
                exchange=ticket.exchange,
                payload=payload,
            )
