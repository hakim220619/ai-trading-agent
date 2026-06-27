"""Centralised loguru logger configuration."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_CONFIGURED = False


def setup_logger(level: str = "INFO", log_dir: str = "logs") -> "logger.__class__":
    """Configure the global loguru logger once.

    Logs to stderr (colorised) and to a rotating file under ``log_dir``.
    Safe to call multiple times; configuration only happens once.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return logger

    logger.remove()

    logger.add(
        sys.stderr,
        level=level,
        colorize=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
    )

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger.add(
        Path(log_dir) / "trading_{time:YYYY-MM-DD}.log",
        level=level,
        rotation="00:00",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )

    _CONFIGURED = True
    return logger


# Default-configured logger ready for import: ``from app.utils.logger import logger``
setup_logger()
