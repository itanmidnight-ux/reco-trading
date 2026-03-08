from reco_trading.kernel.risk_verdict_engine import RiskSignal, RiskVerdictEngine
from reco_trading.reporting.operational_reporting import InstitutionalReportingService


def test_risk_verdict_engine_prioritizes_blocking_signal() -> None:
    engine = RiskVerdictEngine()
    verdict = engine.evaluate(
        [
            RiskSignal(blocked=False, reason='none', source='ok', priority=100),
            RiskSignal(blocked=True, reason='kill_switch', source='kill_switch', priority=10),
            RiskSignal(blocked=True, reason='risk_layer', source='risk', priority=20),
        ]
    )
    assert verdict.blocked is True
    assert verdict.reason == 'kill_switch'
    assert verdict.source == 'kill_switch'


def test_reporting_service_builds_operational_pack() -> None:
    service = InstitutionalReportingService()
    reports = service.build_operational_pack(
        symbol='BTC/USDT',
        session_realized_pnl=10.0,
        lifetime_realized_pnl=50.0,
        total_equity=200.0,
        exchange_qty=0.20,
        db_qty=0.10,
    )
    assert len(reports) == 3
    discrepancy = [item for item in reports if item.report_type == 'db_vs_exchange_discrepancy'][0]
    assert discrepancy.payload['has_discrepancy'] is True
