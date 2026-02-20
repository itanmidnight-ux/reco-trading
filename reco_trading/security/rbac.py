from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SecurityError(RuntimeError):
    pass


class CriticalOperation(str, Enum):
    DEPLOY = 'deploy'
    ROLLBACK = 'rollback'
    KILL_SWITCH = 'kill_switch'


@dataclass(frozen=True, slots=True)
class RolePolicy:
    role: str
    allowed_operations: set[CriticalOperation]


class CriticalRBAC:
    """Role-based authorization for critical production actions."""

    def __init__(self, policies: dict[str, RolePolicy] | None = None) -> None:
        self._policies = policies or {
            'viewer': RolePolicy('viewer', set()),
            'operator': RolePolicy('operator', {CriticalOperation.DEPLOY}),
            'security_admin': RolePolicy('security_admin', {CriticalOperation.DEPLOY, CriticalOperation.ROLLBACK}),
            'platform_admin': RolePolicy('platform_admin', set(CriticalOperation)),
        }

    def authorize(self, role: str, operation: CriticalOperation) -> bool:
        policy = self._policies.get(role)
        return bool(policy and operation in policy.allowed_operations)

    def require(self, role: str, operation: CriticalOperation) -> None:
        if not self.authorize(role, operation):
            raise SecurityError(f'RBAC denied for role={role} operation={operation.value}')
