from loguru import logger
import sys


def configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        format='{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}',
        level='INFO',
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
