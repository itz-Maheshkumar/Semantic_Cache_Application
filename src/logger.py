"""
logger.py — Structured logging utility for the Semantic Cache application.

Usage:
    from src.logger import get_logger
    log = get_logger(__name__)
    log.info("Cache hit", extra={"similarity": 0.92})
"""

import logging
import sys
from logging.handlers import RotatingFileHandler

from src.config import LOG_FILE, LOG_LEVEL, LOG_BACKUP_COUNT, LOG_MAX_BYTES

# ── Formatter ────────────────────────────────────────────────────────────────
_CONSOLE_FMT = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
_FILE_FMT = "%(asctime)s  %(levelname)-8s  %(name)s  [%(filename)s:%(lineno)d] — %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def _build_console_handler() -> logging.StreamHandler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_CONSOLE_FMT, datefmt=_DATE_FMT))
    return handler


def _build_file_handler() -> RotatingFileHandler:
    handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_DATE_FMT))
    return handler


# ── Root logger setup (called once at import time) ───────────────────────────
def _configure_root_logger() -> None:
    root = logging.getLogger()
    if root.handlers:
        return  # Already configured (prevents duplicate handlers in reload scenarios)

    root.setLevel(LOG_LEVEL)
    root.addHandler(_build_console_handler())
    root.addHandler(_build_file_handler())


_configure_root_logger()


# ── Public API ────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger inheriting the root configuration.

    Args:
        name: Typically pass ``__name__`` from the calling module.

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    return logging.getLogger(name)
