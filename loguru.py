from __future__ import annotations

import logging


class _CompatLogger:
    def __init__(self) -> None:
        self._logger = logging.getLogger('loguru_compat')

    def debug(self, *args, **kwargs):
        self._logger.debug(*args, **kwargs)

    def info(self, *args, **kwargs):
        self._logger.info(*args, **kwargs)

    def warning(self, *args, **kwargs):
        self._logger.warning(*args, **kwargs)

    def error(self, *args, **kwargs):
        self._logger.error(*args, **kwargs)

    def exception(self, *args, **kwargs):
        self._logger.exception(*args, **kwargs)


logger = _CompatLogger()
