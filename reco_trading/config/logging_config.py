from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

from loguru import logger


def _json_sink(message) -> None:
    record = message.record
    payload = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'level': record['level'].name,
        'module': record['module'],
        'function': record['function'],
        'line': record['line'],
        'message': record['message'],
    }
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + '\n')


def configure_logging() -> None:
    logger.remove()
    logger.add(
        _json_sink,
        level='INFO',
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
