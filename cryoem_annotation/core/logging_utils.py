"""Logging utilities for consistent status messages."""

import logging
import sys
from typing import Optional


class StatusLogger:
    """Simple status logger with consistent formatting.

    Provides methods for success, error, info, and progress messages
    with consistent prefixes for easy parsing and reading.
    """

    def __init__(self, name: str = "cryoem", verbose: bool = True):
        """Initialize the status logger.

        Args:
            name: Logger name for Python logging integration
            verbose: If False, suppresses info messages
        """
        self.verbose = verbose
        self._logger = logging.getLogger(name)

        # Configure handler if not already configured
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter('%(message)s'))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.DEBUG)

    def success(self, message: str, indent: int = 0) -> None:
        """Log a success message with [OK] prefix.

        Args:
            message: The message to log
            indent: Number of spaces to indent
        """
        prefix = " " * indent
        self._logger.info(f"{prefix}[OK] {message}")

    def error(self, message: str, indent: int = 0) -> None:
        """Log an error message with [ERROR] prefix.

        Args:
            message: The message to log
            indent: Number of spaces to indent
        """
        prefix = " " * indent
        self._logger.error(f"{prefix}[ERROR] {message}")

    def warning(self, message: str, indent: int = 0) -> None:
        """Log a warning message with [WARNING] prefix.

        Args:
            message: The message to log
            indent: Number of spaces to indent
        """
        prefix = " " * indent
        self._logger.warning(f"{prefix}[WARNING] {message}")

    def info(self, message: str, indent: int = 0) -> None:
        """Log an info message (respects verbose setting).

        Args:
            message: The message to log
            indent: Number of spaces to indent
        """
        if self.verbose:
            prefix = " " * indent
            self._logger.info(f"{prefix}{message}")

    def progress(self, current: int, total: int, message: str = "",
                 indent: int = 0) -> None:
        """Log a progress message.

        Args:
            current: Current item number (1-indexed)
            total: Total number of items
            message: Optional message to append
            indent: Number of spaces to indent
        """
        prefix = " " * indent
        progress_str = f"[{current}/{total}]"
        if message:
            self._logger.info(f"{prefix}{progress_str} {message}")
        else:
            self._logger.info(f"{prefix}{progress_str}")

    def header(self, title: str, width: int = 60) -> None:
        """Print a header section.

        Args:
            title: The header title
            width: Width of the separator line
        """
        self._logger.info("=" * width)
        self._logger.info(title)
        self._logger.info("=" * width)

    def separator(self, width: int = 60, char: str = "-") -> None:
        """Print a separator line.

        Args:
            width: Width of the separator
            char: Character to use for the separator
        """
        self._logger.info(char * width)

    def set_verbose(self, verbose: bool) -> None:
        """Set verbose mode.

        Args:
            verbose: If False, info messages are suppressed
        """
        self.verbose = verbose


# Global default logger instance
_default_logger: Optional[StatusLogger] = None


def get_logger(verbose: bool = True) -> StatusLogger:
    """Get the default status logger instance.

    Args:
        verbose: If False, info messages are suppressed

    Returns:
        StatusLogger instance
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = StatusLogger(verbose=verbose)
    else:
        _default_logger.set_verbose(verbose)
    return _default_logger
