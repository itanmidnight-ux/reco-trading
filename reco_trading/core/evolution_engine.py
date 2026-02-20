from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reco_trading.infra.database import Database
from reco_trading.security.rbac import CriticalOperation, CriticalRBAC, SecurityError
from reco_trading.security.signing import ConfigurationSigner


@dataclass(frozen=True)
class RollbackPolicy:
    min_sharpe: float = 1.0
    max_drawdown: float = 0.12
    max_execution_error_rate: float = 0.03


class EvolutionEngine:
    def __init__(
        self,
        database: Database,
        rollback_policy: RollbackPolicy | None = None,
        signer: ConfigurationSigner | None = None,
        rbac: CriticalRBAC | None = None,
    ) -> None:
        self.database = database
        self.rollback_policy = rollback_policy or RollbackPolicy()
        self.signer = signer or ConfigurationSigner('reco_trading')
        self.rbac = rbac or CriticalRBAC()

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
        actor: str = 'system',
        actor_role: str = 'platform_admin',
    ) -> dict[str, Any]:
        self.rbac.require(actor_role, CriticalOperation.DEPLOY)
        if not self.signer.verify(configuration, signature):
            raise SecurityError('Firma inválida para la configuración solicitada')

        version_id = await self.database.create_config_version(
            version,
            configuration,
            signature,
            reason=reason,
            metadata_payload={'validation_metrics': validation_metrics},
            actor=actor,
        )
        deployment_payload = {
            'version': version,
            'version_id': version_id,
            'validation_metrics': validation_metrics,
            'reason': reason,
        }
        deployment_envelope = self.signer.sign(deployment_payload)
        deployment_id = await self.database.register_deployment(
            version_id,
            status='pending',
            reason=reason,
            actor=actor,
            signature=deployment_envelope.signature,
            deployment_hash=deployment_envelope.payload_hash,
        )

        try:
            await self.database.activate_config_version(version, reason='Configuración desplegada y activada', actor=actor)
            rollback_needed, rollback_reason = self.should_trigger_rollback(validation_metrics)

            if rollback_needed:
                self.rbac.require(actor_role, CriticalOperation.ROLLBACK)
                await self.database.register_config_failure(version, rollback_reason, actor=actor)
                rolled_back_to = await self.database.execute_rollback(
                    from_version=version,
                    reason=rollback_reason,
                    deployment_id=deployment_id,
                    actor=actor,
                )
                await self.database.complete_deployment(deployment_id, status='rolled_back', reason=rollback_reason, actor=actor)
                return {
                    'version': version,
                    'status': 'rolled_back',
                    'reason': rollback_reason,
                    'rolled_back_to': rolled_back_to,
                }

            await self.database.complete_deployment(deployment_id, status='active', reason='Deployment exitoso', actor=actor)
            return {
                'version': version,
                'status': 'active',
                'reason': 'Deployment aplicado sin degradación de métricas',
            }
        except Exception as exc:
            failure_reason = f'Deployment fallido: {exc}'
            await self.database.register_config_failure(version, failure_reason, actor=actor)
            rolled_back_to = await self.database.execute_rollback(
                from_version=version,
                reason=failure_reason,
                deployment_id=deployment_id,
                actor=actor,
            )
            await self.database.complete_deployment(deployment_id, status='failed', reason=failure_reason, actor=actor)
            return {
                'version': version,
                'status': 'failed',
                'reason': failure_reason,
                'rolled_back_to': rolled_back_to,
            }

    async def activate_kill_switch(self, *, reason: str, actor: str, actor_role: str) -> dict[str, str]:
        self.rbac.require(actor_role, CriticalOperation.KILL_SWITCH)
        await self.database.append_audit_event(
            event_type='security_event',
            actor=actor,
            target='kill_switch',
            payload={'action': 'activate_kill_switch', 'reason': reason},
        )
        return {'status': 'kill_switch_activated', 'reason': reason}
