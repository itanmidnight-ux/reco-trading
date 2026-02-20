from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from loguru import logger


@dataclass(slots=True)
class EvolutionState:
    generation: int = 0
    active_profile: str = 'baseline'
    last_health_score: float = 1.0
    last_mutation_reason: str = 'bootstrap'
    metadata: dict[str, Any] = field(default_factory=dict)


class EvolutionEngine:
    """Motor de evolución incremental para estrategia/configuración live."""

    def __init__(self, *, min_health_score: float = 0.65) -> None:
        self.min_health_score = float(min(max(min_health_score, 0.0), 1.0))
        self.state = EvolutionState()

    async def evaluate_system_health(
        self,
        *,
        daily_pnl: float,
        consecutive_losses: int,
        kill_switch: bool,
        latency_ms: float | None = None,
    ) -> dict[str, Any]:
        latency_penalty = 0.0 if latency_ms is None else min(max(latency_ms, 0.0) / 2000.0, 0.35)
        pnl_penalty = min(abs(min(daily_pnl, 0.0)) * 8.0, 0.4)
        loss_penalty = min(max(consecutive_losses, 0) * 0.08, 0.4)
        kill_penalty = 0.5 if kill_switch else 0.0

        score = float(max(0.0, min(1.0, 1.0 - latency_penalty - pnl_penalty - loss_penalty - kill_penalty)))
        healthy = score >= self.min_health_score
        self.state.last_health_score = score

        report = {
            'healthy': healthy,
            'health_score': score,
            'generation': self.state.generation,
            'active_profile': self.state.active_profile,
            'evaluated_at': datetime.now(timezone.utc).isoformat(),
            'penalties': {
                'latency': latency_penalty,
                'pnl': pnl_penalty,
                'losses': loss_penalty,
                'kill_switch': kill_penalty,
            },
        }
        logger.info('Evolution health evaluated', report=report)
        return report

    async def mutate_strategies(self, health_report: dict[str, Any]) -> dict[str, Any]:
        if health_report.get('healthy', False):
            mutation = {
                'mutated': False,
                'reason': 'system_healthy',
                'candidate_profile': self.state.active_profile,
                'generation': self.state.generation,
            }
            logger.info('Evolution mutation skipped', mutation=mutation)
            return mutation

        self.state.generation += 1
        candidate = f"adaptive_profile_{self.state.generation}"
        reason = f"health_score={health_report.get('health_score', 0.0):.3f}"
        self.state.last_mutation_reason = reason

        mutation = {
            'mutated': True,
            'candidate_profile': candidate,
            'reason': reason,
            'generation': self.state.generation,
            'health_reference': health_report,
        }
        logger.warning('Evolution mutation generated', mutation=mutation)
        return mutation

    async def deploy_new_configuration(self, mutation_plan: dict[str, Any]) -> dict[str, Any]:
        if not mutation_plan.get('mutated', False):
            return {
                'deployed': False,
                'profile': self.state.active_profile,
                'generation': self.state.generation,
                'reason': mutation_plan.get('reason', 'no_mutation'),
            }

        profile = str(mutation_plan.get('candidate_profile', self.state.active_profile))
        self.state.active_profile = profile
        self.state.metadata['last_deploy_at'] = datetime.now(timezone.utc).isoformat()

        deployment = {
            'deployed': True,
            'profile': profile,
            'generation': self.state.generation,
            'reason': mutation_plan.get('reason', 'mutation'),
            'deployed_at': self.state.metadata['last_deploy_at'],
        }
        logger.success('Evolution deployment applied', deployment=deployment)
        return deployment
