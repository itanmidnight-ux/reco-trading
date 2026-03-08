from __future__ import annotations

import sys
from loguru import logger


def configure_logger() -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        format='<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}',
        level='INFO',
        enqueue=True,
    )


__all__ = ['configure_logger', 'logger']
