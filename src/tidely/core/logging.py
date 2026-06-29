"""Logging configuration for Tidely using Rich for structured console output."""

import logging

from rich.logging import RichHandler

# Root logger for the package
logger = logging.getLogger("tidely")
# Default to NullHandler so the library is silent by default
logger.addHandler(logging.NullHandler())
logger.propagate = False


def setup_logging(
    level: str = "INFO",
    show_time: bool = True,
    show_level: bool = True,
    show_path: bool = False,
    log_file: str | None = None,
) -> logging.Logger:
    """Configures the logging for the tidely library with Rich console output.

    Args:
        level: The logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR").
        show_time: Whether to show timestamp in logs.
        show_level: Whether to show log level (e.g. INFO) in logs.
        show_path: Whether to show file name and line number in logs.
        log_file: Optional path to write plain text logs to.

    Returns:
        logging.Logger: The configured tidely logger.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Clear existing handlers
    logger.handlers.clear()

    # Rich console handler
    console_handler = RichHandler(
        level=numeric_level,
        show_time=show_time,
        show_level=show_level,
        show_path=show_path,
        markup=True,
    )
    console_handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    logger.addHandler(console_handler)

    # Optional file handler for persistent logs
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)

    logger.setLevel(numeric_level)
    logger.debug(f"Tidely logging initialized at level {level}")
    return logger
