from __future__ import annotations

import logging
from typing import Any


class _CompatLogger:
    """Minimal loguru-compatible adapter over stdlib logging.

    Supports the subset of API used in this repository:
    - logger.info/debug/warning/error/exception(..., **extra)
    - logger.bind(**context) returning a contextual logger
    """

    def __init__(self, context: dict[str, Any] | None = None) -> None:
        self._logger = logging.getLogger('loguru_compat')
        self._context = context or {}

    def bind(self, **kwargs: Any) -> '_CompatLogger':
        merged = dict(self._context)
        merged.update(kwargs)
        return _CompatLogger(context=merged)

    def _log(self, level: int, *args: Any, **kwargs: Any) -> None:
        if not args:
            message = ''
        else:
            message = str(args[0])

        if len(args) > 1:
            try:
                message = message % args[1:]
            except Exception:
                message = ' '.join(str(part) for part in args)

        payload: dict[str, Any] = dict(self._context)
        payload.update(kwargs)
        if payload:
            message = f"{message} | {payload}"

        self._logger.log(level, message)

    def debug(self, *args: Any, **kwargs: Any) -> None:
        self._log(logging.DEBUG, *args, **kwargs)

    def info(self, *args: Any, **kwargs: Any) -> None:
        self._log(logging.INFO, *args, **kwargs)

    def warning(self, *args: Any, **kwargs: Any) -> None:
        self._log(logging.WARNING, *args, **kwargs)

    def error(self, *args: Any, **kwargs: Any) -> None:
        self._log(logging.ERROR, *args, **kwargs)

    def critical(self, *args: Any, **kwargs: Any) -> None:
        self._log(logging.CRITICAL, *args, **kwargs)

    def success(self, *args: Any, **kwargs: Any) -> None:
        self._log(logging.INFO, *args, **kwargs)

    def exception(self, *args: Any, **kwargs: Any) -> None:
        self._log(logging.ERROR, *args, **kwargs)


logger = _CompatLogger()
