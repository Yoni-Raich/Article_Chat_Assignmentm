"""
Logging configuration module.

This module sets up and configures the logger for the application.
"""

# Standard library imports
import logging

def setup_logger(name, level="INFO"):
    """Create a simple logger for the project"""
    log = logging.getLogger(name)

    # Prevent duplicate handlers
    if log.handlers:
        return log

    # Set log level
    log.setLevel(getattr(logging, level.upper()))

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    log.addHandler(console_handler)

    return log

# Create default logger
logger = setup_logger("article_chat")
