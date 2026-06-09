"""Centralized logging configuration for the RAG pipeline."""

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    format_str: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for the pipeline.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format_str: Custom log format string

    Returns:
        Configured logger instance
    """
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger("rag_pipeline")


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger instance.

    Args:
        name: Logger name (typically __name__ from calling module)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
