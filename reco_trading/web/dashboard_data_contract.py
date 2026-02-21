from __future__ import annotations

"""Contrato de datos del dashboard.

Fuente de verdad (runtime):
- Señales: ``trade_signals``.
- Ejecución: ``order_executions`` (resumen por orden) y ``fills`` (detalle de ejecución).
- Estado de portafolio: ``portfolio_state.snapshot`` (JSON serializado desde RuntimeState).

Este módulo centraliza queries y convenciones para evitar divergencias entre writers
del kernel/database y readers del dashboard.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DashboardDataContract:
    last_signal_query: str = """
        SELECT signal, score, reason, created_at
        FROM trade_signals
        ORDER BY created_at DESC
        LIMIT 1
    """

    daily_execution_pnls_query: str = """
        SELECT pnl
        FROM order_executions
        WHERE pnl IS NOT NULL
          AND created_at::date = CURRENT_DATE
    """

    daily_fill_aggregate_query: str = """
        SELECT
            COALESCE(SUM(CASE WHEN side = 'BUY' THEN fill_price * fill_amount ELSE 0 END), 0) AS buy_notional,
            COALESCE(SUM(CASE WHEN side = 'SELL' THEN fill_price * fill_amount ELSE 0 END), 0) AS sell_notional,
            COALESCE(SUM(fee), 0) AS fees
        FROM fills
        WHERE created_at::date = CURRENT_DATE
    """

    current_operation_query: str = """
        SELECT side, qty, price, status, created_at
        FROM order_executions
        ORDER BY created_at DESC
        LIMIT 1
    """

    latest_portfolio_snapshot_query: str = """
        SELECT snapshot, created_at
        FROM portfolio_state
        ORDER BY created_at DESC
        LIMIT 1
    """


CONTRACT = DashboardDataContract()

