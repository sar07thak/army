"""Centralized logging configuration.

Every module in the project obtains its logger through :func:`get_logger`.
:func:`setup_logging` wires a console handler and a rotating file handler
(``logs/project.log`` by default) onto the root logger. Repeated calls are
deterministic: pre-existing root handlers are replaced, never duplicated.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
DEFAULT_LOG_FILE = "project.log"
DEFAULT_MAX_BYTES = 5 * 1024 * 1024  # 5 MB per log file
DEFAULT_BACKUP_COUNT = 3
DEFAULT_CONSOLE_LEVEL = logging.INFO
DEFAULT_FILE_LEVEL = logging.DEBUG


def _make_console_handler(level: int = DEFAULT_CONSOLE_LEVEL) -> logging.Handler:
    """Return a console handler that prints formatted records to stdout."""
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    return handler


def _make_file_handler(
    log_dir: Path,
    log_file: str = DEFAULT_LOG_FILE,
    level: int = DEFAULT_FILE_LEVEL,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> logging.Handler:
    """Return a rotating file handler writing to ``log_dir / log_file``.

    Args:
        log_dir: Directory the log file lives in (created if missing).
        log_file: Log file name inside ``log_dir``.
        level: Minimum level written to the file.
        max_bytes: Size at which the file rotates.
        backup_count: Number of rotated backups to retain.

    Raises:
        OSError: if the log directory cannot be created.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_dir / log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    return handler


def setup_logging(
    log_dir: Path | None = None,
    log_file: str = DEFAULT_LOG_FILE,
    console_level: int = DEFAULT_CONSOLE_LEVEL,
    file_level: int = DEFAULT_FILE_LEVEL,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> None:
    """Configure root logging with console + rotating file handlers.

    Replaces any pre-existing root handlers so repeated calls are
    deterministic (important for tests). Call once at process start.

    Args:
        log_dir: Directory for log files (defaults to ``<repo>/logs``).
        log_file: Log file name inside ``log_dir``.
        console_level: Minimum level emitted to the console.
        file_level: Minimum level written to the file.
        max_bytes: Rotation size for the file handler.
        backup_count: Number of rotated backup files to keep.
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    root.addHandler(_make_console_handler(console_level))
    root.addHandler(
        _make_file_handler(
            Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR,
            log_file,
            file_level,
            max_bytes,
            backup_count,
        )
    )


def get_logger(name: str) -> logging.Logger:
    """Return the module logger for ``name``.

    Loggers inherit the root configuration created by :func:`setup_logging`.
    """
    return logging.getLogger(name)
