"""Application logging helpers."""

from __future__ import annotations

import logging
from logging import Logger

from constants import LOG_DIR, LOG_FILE_PATH

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure console + file logging once for the process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(LOG_FILE_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # Avoid duplicate handlers if Streamlit reloads modules oddly.
    # FileHandler subclasses StreamHandler, so check console separately.
    has_file = any(isinstance(h, logging.FileHandler) for h in root.handlers)
    has_console = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    )
    if not has_file:
        root.addHandler(file_handler)
    if not has_console:
        root.addHandler(console_handler)

    _CONFIGURED = True


def get_logger(name: str) -> Logger:
    """Return a named logger after ensuring logging is configured."""
    setup_logging()
    return logging.getLogger(name)
