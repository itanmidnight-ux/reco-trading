from __future__ import annotations

from loguru import logger


class AlertManager:
    def emit(self, title: str, detail: str) -> None:
        logger.error(f'ALERT | {title} | {detail}')
