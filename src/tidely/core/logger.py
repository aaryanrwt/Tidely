"""Standardized logging configuration for Tidely."""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Returns a configured logger for the Tidely package.

    Args:
        name: The name of the module requesting the logger.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Only configure if it doesn't already have handlers to avoid duplicate logs
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)

        # Consistent format: [tidely] INFO - message
        formatter = logging.Formatter(fmt="[%(name)s] %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Prevent propagation to the root logger to keep output clean
        logger.propagate = False

    return logger
