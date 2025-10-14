"""Centralised logging helpers for the malware traffic detector."""

from __future__ import annotations

import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Dict

_LOGGERS: Dict[str, logging.Logger] = {}


def _resolve_log_dir() -> Path:
    base = Path(
        os.getenv("MALDET_DATA_DIR", Path.home() / "maldet_data")
    ).expanduser().resolve()
    log_dir = Path(os.getenv("MALDET_LOG_DIR", base / "logs")).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_logger(name: str) -> logging.Logger:
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    if logger.handlers:
        _LOGGERS[name] = logger
        return logger

    logger.setLevel(logging.INFO)
    log_dir = _resolve_log_dir()
    safe_name = name.replace("/", "_").replace(".", "_")
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = TimedRotatingFileHandler(
        log_dir / f"{safe_name}.log",
        when="D",
        interval=1,
        backupCount=7,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.propagate = False

    _LOGGERS[name] = logger
    return logger
