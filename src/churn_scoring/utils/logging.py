"""Logging configuration for the project."""

import logging


def setup_logging(log_level: str = "INFO") -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
