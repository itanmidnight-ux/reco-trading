from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reco_trading.monitoring.alert_manager import AlertManager


@dataclass(slots=True)
class QuantKernelState:
    conservative_mode: bool = False
    kill_switch: bool = False
    firewall_rejections: int = 0


class QuantKernel:
    def __init__(self, alert_manager: AlertManager | None = None, rejection_threshold_for_conservative: int = 3) -> None:
        self.alert_manager = alert_manager or AlertManager()
        self.state = QuantKernelState()
        self.rejection_threshold_for_conservative = max(int(rejection_threshold_for_conservative), 1)

    def should_block_trading(self) -> bool:
        return self.state.kill_switch

    def on_firewall_rejection(self, reason: str, risk_snapshot: dict[str, Any]) -> None:
        self.state.firewall_rejections += 1
        severity = 'warning'

        if reason in {'daily_loss_limit', 'total_exposure_limit'}:
            self.state.kill_switch = True
            severity = 'critical'
        elif self.state.firewall_rejections >= self.rejection_threshold_for_conservative:
            self.state.conservative_mode = True

        self.alert_manager.emit(
            title='execution firewall rejection',
            detail=f'Orden bloqueada por firewall: {reason}',
            severity=severity,
            payload={
                'reason': reason,
                'firewall_rejections': self.state.firewall_rejections,
                'conservative_mode': self.state.conservative_mode,
                'kill_switch': self.state.kill_switch,
                'risk_snapshot': risk_snapshot,
            },
        )
