"""Tests for the Tidely logging infrastructure."""

import logging

from tidely.core.logging import logger, setup_logging


def test_default_logger_state() -> None:
    """Default logger should have a NullHandler and not propagate."""
    # Reset logger to default state for testing
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    assert not logger.propagate
    assert len(logger.handlers) >= 1
    assert any(isinstance(h, logging.NullHandler) for h in logger.handlers)


def test_setup_logging_console() -> None:
    """setup_logging should configure a RichHandler."""
    setup_logging(level="DEBUG")
    assert logger.level == logging.DEBUG
    # Verify we added the RichHandler
    assert len(logger.handlers) >= 1
    # Check handlers
    from rich.logging import RichHandler

    assert any(isinstance(h, RichHandler) for h in logger.handlers)


def test_setup_logging_file(tmp_path) -> None:  # type: ignore
    """setup_logging should support file handlers."""
    log_file = tmp_path / "test.log"
    setup_logging(level="INFO", log_file=str(log_file))

    logger.info("Test file logging message")

    # Force flushing handlers
    for h in logger.handlers:
        h.flush()

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Test file logging message" in content
