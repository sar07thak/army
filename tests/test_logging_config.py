"""Tests for ``src.logging_config``."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src import logging_config


@pytest.fixture(autouse=True)
def _reset_root_logger() -> None:
    """Isolate each test by clearing root handlers before and after."""
    logging.getLogger().handlers.clear()
    yield
    logging.getLogger().handlers.clear()


def test_get_logger_returns_named_logger() -> None:
    logger = logging_config.get_logger("test.module")
    assert logger.name == "test.module"


def test_setup_creates_log_dir_and_writes_info(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    logging_config.setup_logging(log_dir=log_dir)
    logging_config.get_logger("test").info("hello m1")
    files = list(log_dir.glob("*.log"))
    assert len(files) == 1
    assert "hello m1" in files[0].read_text(encoding="utf-8")


def test_setup_is_idempotent(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    logging_config.setup_logging(log_dir=log_dir)
    logging_config.setup_logging(log_dir=log_dir)
    assert len(logging.getLogger().handlers) == 2  # console + file, no duplicates


def test_default_log_dir_is_repo_logs_and_writes() -> None:
    logging_config.setup_logging()  # uses the real default <repo>/logs
    logging_config.get_logger("test").info("default dir m1")
    assert logging_config.DEFAULT_LOG_DIR.is_dir()
    log_file = logging_config.DEFAULT_LOG_DIR / logging_config.DEFAULT_LOG_FILE
    assert log_file.is_file()
    assert "default dir m1" in log_file.read_text(encoding="utf-8")


def test_warning_reaches_file_below_console_level(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    logging_config.setup_logging(
        log_dir=log_dir, console_level=logging.ERROR, file_level=logging.DEBUG
    )
    logging_config.get_logger("test").warning("warn m1")
    files = list(log_dir.glob("*.log"))
    assert len(files) == 1
    assert "warn m1" in files[0].read_text(encoding="utf-8")
