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
    status: str
    reason: str
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class CapitalGovernorState:
    equity: float = 0.0
    daily_pnl: float = 0.0
    total_exposure: float = 0.0
    exposure_by_asset: dict[str, float] = field(default_factory=dict)
    exposure_by_exchange: dict[str, float] = field(default_factory=dict)
    capital_by_strategy: dict[str, float] = field(default_factory=dict)
    hard_cap_global: float = float('inf')


class CapitalGovernor:
    """Gatekeeper de capital para trading real.

    Reglas duras:
    - riesgo por trade <= 0.5% del equity
    - pérdida diaria <= 3% del equity
    - exposición total <= max_total_exposure_ratio * equity
    - CVaR diario <= max_cvar_ratio * equity
    """

    def __init__(
        self,
        *,
        hard_cap_global: float,
        max_risk_per_trade_ratio: float = 0.005,
        max_daily_loss_ratio: float = 0.03,
        max_total_exposure_ratio: float = 0.80,
        max_cvar_ratio: float = 0.025,
        ticket_ttl_seconds: int = 30,
        metrics: TradingMetrics | None = None,
        alert_manager: AlertManager | None = None,
        labels: dict[str, str] | None = None,
    ) -> None:
        self.state = CapitalGovernorState(hard_cap_global=max(float(hard_cap_global), 0.0))
        self.max_risk_per_trade_ratio = float(max(max_risk_per_trade_ratio, 0.0))
        self.max_daily_loss_ratio = float(max(max_daily_loss_ratio, 0.0))
        self.max_total_exposure_ratio = float(max(max_total_exposure_ratio, 0.0))
        self.max_cvar_ratio = float(max(max_cvar_ratio, 0.0))
        self.ticket_ttl_seconds = max(int(ticket_ttl_seconds), 1)
        self.metrics = metrics
        self.alert_manager = alert_manager
        self.labels = labels or {}

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
        equity: float | None = None,
        daily_pnl: float | None = None,
    ) -> None:
        self.state.capital_by_strategy[strategy] = float(capital_by_strategy)
        self.state.exposure_by_exchange[exchange] = float(capital_by_exchange)
        self.state.total_exposure = float(total_exposure)
        self.state.exposure_by_asset[symbol] = float(asset_exposure)
        if equity is not None:
            self.state.equity = float(equity)
        if daily_pnl is not None:
            self.state.daily_pnl = float(daily_pnl)

    @staticmethod
    def rolling_var_cvar(pnl_or_returns: list[float] | np.ndarray, alpha: float = 0.95) -> tuple[float, float]:
        arr = np.asarray(pnl_or_returns, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 0.0, 0.0
        losses = -arr
        var = float(np.quantile(losses, alpha))
        tail = losses[losses >= var]
        cvar = var if tail.size == 0 else float(np.mean(tail))
        return max(var, 0.0), max(cvar, 0.0)

    def _reject(self, strategy: str, exchange: str, symbol: str, requested_notional: float, reason: str, metrics_payload: dict[str, float]) -> CapitalTicket:
        if self.metrics:
            self.metrics.observe_error('capital_governor', reason, exchange=exchange, strategy=strategy)
        if self.alert_manager:
            self.alert_manager.emit(
                'CapitalGovernor rechazó orden',
                f'{symbol} bloqueado: {reason}',
                severity='warning',
                exchange=exchange,
                payload=metrics_payload,
            )
        now = datetime.now(timezone.utc)
        return CapitalTicket(
            ticket_id=f'reject-{uuid4()}',
            issued_at=now,
            expires_at=now,
            strategy=strategy,
            exchange=exchange,
            symbol=symbol,
            requested_notional=float(requested_notional),
            approved_notional=0.0,
            status='rejected',
            reason=reason,
            metrics=metrics_payload,
        )

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
        estimated_trade_risk: float | None = None,
    ) -> CapitalTicket:
        request = max(float(requested_notional), 0.0)
        equity = max(float(self.state.equity), float(self.state.hard_cap_global), 1.0)
        daily_loss = max(-float(self.state.daily_pnl), 0.0)
        var95, cvar95 = self.rolling_var_cvar(pnl_or_returns, alpha=0.95)

        max_trade_risk = equity * self.max_risk_per_trade_ratio
        trade_risk = max(float(estimated_trade_risk if estimated_trade_risk is not None else request * 0.01), 0.0)
        projected_total_exposure = self.state.total_exposure + request
        hard_cap = min(self.state.hard_cap_global, equity * self.max_total_exposure_ratio)
        projected_daily_loss = daily_loss + trade_risk
        cvar_notional = cvar95 * equity

        metrics_payload = {
            'equity': equity,
            'trade_risk': trade_risk,
            'max_trade_risk': max_trade_risk,
            'daily_loss': daily_loss,
            'projected_daily_loss': projected_daily_loss,
            'projected_total_exposure': projected_total_exposure,
            'hard_cap': hard_cap,
            'var95': var95,
            'cvar95': cvar95,
            'cvar_notional': cvar_notional,
            'spread_bps': float(spread_bps),
            'available_liquidity': float(available_liquidity),
            'price_gap_pct': float(price_gap_pct),
        }

        if request <= 0.0:
            return self._reject(strategy, exchange, symbol, request, 'invalid_notional', metrics_payload)
        if trade_risk > max_trade_risk:
            return self._reject(strategy, exchange, symbol, request, 'risk_per_trade_limit', metrics_payload)
        if projected_daily_loss > equity * self.max_daily_loss_ratio:
            return self._reject(strategy, exchange, symbol, request, 'daily_loss_limit', metrics_payload)
        if projected_total_exposure > hard_cap:
            return self._reject(strategy, exchange, symbol, request, 'total_exposure_limit', metrics_payload)
        if len(np.asarray(pnl_or_returns, dtype=float)) >= 20 and cvar_notional > equity * self.max_cvar_ratio:
            return self._reject(strategy, exchange, symbol, request, 'cvar_limit', metrics_payload)
        if available_liquidity <= 0.0:
            return self._reject(strategy, exchange, symbol, request, 'liquidity_unavailable', metrics_payload)

        approved_notional = min(request, max(available_liquidity * 0.15, 0.0), max(hard_cap - self.state.total_exposure, 0.0))
        if approved_notional <= 0:
            return self._reject(strategy, exchange, symbol, request, 'no_capacity', metrics_payload)

        now = datetime.now(timezone.utc)
        ticket = CapitalTicket(
            ticket_id=f'cap-{uuid4()}',
            issued_at=now,
            expires_at=now + timedelta(seconds=self.ticket_ttl_seconds),
            strategy=strategy,
            exchange=exchange,
            symbol=symbol,
            requested_notional=request,
            approved_notional=float(approved_notional),
            status='approved',
            reason='ok',
            metrics=metrics_payload,
        )
        if self.metrics:
            self.metrics.observe_request('capital_governor', exchange=exchange, strategy=strategy)
        return ticket


    def validate_ticket(self, ticket: CapitalTicket | None, min_notional: float | None = None) -> tuple[bool, str]:
        if ticket is None:
            return False, 'capital_ticket_missing'
        if ticket.status != 'approved':
            return False, 'capital_ticket_invalid'
        if ticket.expires_at <= datetime.now(timezone.utc):
            return False, 'capital_ticket_expired'
        if min_notional is not None and float(ticket.approved_notional) < float(min_notional):
            return False, 'capital_ticket_insufficient'
        return True, 'ok'

    @staticmethod
    def ticket_is_valid(ticket: CapitalTicket | None) -> bool:
        return bool(ticket and ticket.status == 'approved' and ticket.expires_at > datetime.now(timezone.utc))

    def register_fill(self, *, symbol: str, exchange: str, notional: float, pnl: float = 0.0) -> None:
        self.state.total_exposure += max(float(notional), 0.0)
        self.state.daily_pnl += float(pnl)
        self.state.exposure_by_asset[symbol] = self.state.exposure_by_asset.get(symbol, 0.0) + max(float(notional), 0.0)
        self.state.exposure_by_exchange[exchange] = self.state.exposure_by_exchange.get(exchange, 0.0) + max(float(notional), 0.0)
