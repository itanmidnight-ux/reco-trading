from __future__ import annotations

from typing import Any

from loguru import logger


class AlertManager:
    def emit(self, title: str, detail: str, *, severity: str = 'error', exchange: str | None = None, payload: dict[str, Any] | None = None) -> None:
        bound = logger.bind(component='alert_manager', title=title, severity=severity, exchange=exchange, payload=payload or {})
        log_fn = getattr(bound, severity, bound.error)
        log_fn(f'ALERT | {title} | {detail}')
