from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reco_trading.infra.database import Database


@dataclass(frozen=True)
class RollbackPolicy:
    min_sharpe: float = 1.0
    max_drawdown: float = 0.12
    max_execution_error_rate: float = 0.03


class EvolutionEngine:
    def __init__(self, database: Database, rollback_policy: RollbackPolicy | None = None) -> None:
        self.database = database
        self.rollback_policy = rollback_policy or RollbackPolicy()

    def should_trigger_rollback(self, metrics: dict[str, float]) -> tuple[bool, str]:
        sharpe = float(metrics.get('sharpe', 0.0))
        drawdown = float(metrics.get('drawdown', 0.0))
        execution_error_rate = float(metrics.get('execution_error_rate', 0.0))

        if sharpe < self.rollback_policy.min_sharpe:
            return True, f'Sharpe degradado: {sharpe:.4f} < {self.rollback_policy.min_sharpe:.4f}'
        if drawdown > self.rollback_policy.max_drawdown:
            return True, f'Drawdown excedido: {drawdown:.4f} > {self.rollback_policy.max_drawdown:.4f}'
        if execution_error_rate > self.rollback_policy.max_execution_error_rate:
            return True, (
                'Errores de ejecución elevados: '
                f'{execution_error_rate:.4f} > {self.rollback_policy.max_execution_error_rate:.4f}'
            )
        return False, 'Métricas saludables para mantener la versión activa'

    async def deploy_new_configuration(
        self,
        *,
        version: str,
        configuration: dict[str, Any],
        signature: str,
        validation_metrics: dict[str, float],
        reason: str | None = None,
    ) -> dict[str, Any]:
        version_id = await self.database.create_config_version(
            version,
            configuration,
            signature,
            reason=reason,
            metadata_payload={'validation_metrics': validation_metrics},
        )
        deployment_id = await self.database.register_deployment(version_id, status='pending', reason=reason)

        try:
            await self.database.activate_config_version(version, reason='Configuración desplegada y activada')
            rollback_needed, rollback_reason = self.should_trigger_rollback(validation_metrics)

            if rollback_needed:
                await self.database.register_config_failure(version, rollback_reason)
                rolled_back_to = await self.database.execute_rollback(
                    from_version=version,
                    reason=rollback_reason,
                    deployment_id=deployment_id,
                )
                await self.database.complete_deployment(deployment_id, status='rolled_back', reason=rollback_reason)
                return {
                    'version': version,
                    'status': 'rolled_back',
                    'reason': rollback_reason,
                    'rolled_back_to': rolled_back_to,
                }

            await self.database.complete_deployment(deployment_id, status='active', reason='Deployment exitoso')
            return {
                'version': version,
                'status': 'active',
                'reason': 'Deployment aplicado sin degradación de métricas',
            }
        except Exception as exc:
            failure_reason = f'Deployment fallido: {exc}'
            await self.database.register_config_failure(version, failure_reason)
            rolled_back_to = await self.database.execute_rollback(
                from_version=version,
                reason=failure_reason,
                deployment_id=deployment_id,
            )
            await self.database.complete_deployment(deployment_id, status='failed', reason=failure_reason)
            return {
                'version': version,
                'status': 'failed',
                'reason': failure_reason,
                'rolled_back_to': rolled_back_to,
            }
