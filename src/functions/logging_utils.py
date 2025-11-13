"""Centralised logging helpers for the malware traffic detector.

This module not only exposes :func:`get_logger` but also structured logging
utilities so that long-running training jobs can persist metadata that is later
consumed by other services (CLI/REST) or the UI.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Dict, Optional

_LOGGERS: Dict[str, logging.Logger] = {}
_EVENT_LOG_LOCK = threading.Lock()
_EVENT_LOG_NAME = "model_events.jsonl"


def _resolve_log_dir() -> Path:
    base = Path(
        os.getenv("MALDET_DATA_DIR", Path.home() / "maldet_data")
    ).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    base = base.resolve()
    log_dir = Path(os.getenv("MALDET_LOG_DIR", base / "logs")).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_log_dir() -> Path:
    """Return the directory where application logs are stored."""

    return _resolve_log_dir()


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


def _event_log_path() -> Path:
    return _resolve_log_dir() / _EVENT_LOG_NAME


def _write_event(payload: Dict[str, object]) -> None:
    path = _event_log_path()
    try:
        serialised = json.dumps(payload, ensure_ascii=False)
    except TypeError:
        # Fallback to string representation for non-serialisable payloads
        serialised = json.dumps({"raw": str(payload)}, ensure_ascii=False)
    with _EVENT_LOG_LOCK:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(serialised)
                handle.write("\n")
        except Exception:
            # Logging errors should never crash the main workflow.
            logger = get_logger(__name__)
            logger.debug("Failed to persist structured log entry", exc_info=True)


def log_model_event(event_type: str, data: Optional[Dict[str, object]] = None) -> None:
    """Persist a structured model lifecycle event to a JSONL log file."""

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": str(event_type),
        "data": dict(data or {}),
    }
    _write_event(entry)


def log_training_run(payload: Dict[str, object]) -> None:
    """Persist a single training run summary for later auditing."""

    enriched = dict(payload)
    enriched.setdefault("event_type", "training")
    log_model_event("training", enriched)