from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(slots=True)
class OperationalReport:
    report_type: str
    generated_at: str
    payload: dict


class InstitutionalReportingService:
    def build_operational_pack(
        self,
        *,
        symbol: str,
        session_realized_pnl: float,
        lifetime_realized_pnl: float,
        total_equity: float,
        exchange_qty: float,
        db_qty: float,
    ) -> list[OperationalReport]:
        now = datetime.now(timezone.utc).isoformat()
        discrepancy = abs(exchange_qty - db_qty)
        return [
            OperationalReport(
                report_type='session_vs_lifetime_traceability',
                generated_at=now,
                payload={
                    'symbol': symbol,
                    'session_realized_pnl': float(session_realized_pnl),
                    'lifetime_realized_pnl': float(lifetime_realized_pnl),
                    'delta': float(lifetime_realized_pnl - session_realized_pnl),
                },
            ),
            OperationalReport(
                report_type='db_vs_exchange_discrepancy',
                generated_at=now,
                payload={
                    'symbol': symbol,
                    'db_qty': float(db_qty),
                    'exchange_qty': float(exchange_qty),
                    'abs_discrepancy': float(discrepancy),
                    'has_discrepancy': bool(discrepancy > 1e-8),
                },
            ),
            OperationalReport(
                report_type='eod_ready_snapshot',
                generated_at=now,
                payload={
                    'symbol': symbol,
                    'total_equity': float(total_equity),
                    'session_realized_pnl': float(session_realized_pnl),
                    'lifetime_realized_pnl': float(lifetime_realized_pnl),
                    'sealed': True,
                },
            ),
        ]
