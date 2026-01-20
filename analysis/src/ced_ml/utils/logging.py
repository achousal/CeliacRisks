"""
Consistent logging setup for CeD-ML pipeline.

Replaces print() statements with proper logging throughout the codebase.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "ced_ml",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
    use_live_log: bool = False,
) -> logging.Logger:
    """
    Setup a logger with console and optional file output.

    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_file: Optional path to log file
        format_string: Custom format string (default: timestamp + level + message)
        use_live_log: If True, log to .live file and rename on completion

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Default format
    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # If use_live_log, write to .live file first
        if use_live_log:
            live_log_file = log_file.with_suffix(".live")
            file_handler = logging.FileHandler(live_log_file, mode="a")
            # Store the final log path for later renaming
            file_handler._final_log_path = log_file
            file_handler._live_log_path = live_log_file
        else:
            file_handler = logging.FileHandler(log_file, mode="a")

        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "ced_ml") -> logging.Logger:
    """Get existing logger or create a basic one."""
    logger = logging.getLogger(name)

    # If no handlers, setup a basic console logger
    if not logger.handlers:
        logger = setup_logger(name)

    return logger


def finalize_live_log(logger: logging.Logger) -> None:
    """
    Finalize .live log files by renaming them to their final names.

    This should be called at the end of a script to mark logs as completed.
    Looks for file handlers with _live_log_path attribute and renames them.

    Args:
        logger: Logger instance to finalize
    """
    import shutil

    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            # Check if this is a live log handler
            if hasattr(handler, "_live_log_path") and hasattr(handler, "_final_log_path"):
                handler.close()
                live_path = Path(handler._live_log_path)
                final_path = Path(handler._final_log_path)

                if live_path.exists():
                    # Move .live to final name
                    shutil.move(str(live_path), str(final_path))
                    # Remove this handler since file is closed
                    logger.removeHandler(handler)


def cleanup_live_logs(log_dir: Path, pattern: str = "*.live") -> None:
    """
    Find and finalize all .live log files in a directory.

    Useful for cleanup after job completion or crashes.

    Args:
        log_dir: Directory containing .live log files
        pattern: Glob pattern for live logs (default: *.live)
    """
    import shutil

    log_dir = Path(log_dir)
    for live_log in log_dir.glob(pattern):
        final_log = live_log.with_suffix(".log")
        if live_log.exists():
            shutil.move(str(live_log), str(final_log))


class LoggerContext:
    """Context manager for temporary logger configuration."""

    def __init__(
        self,
        logger: logging.Logger,
        level: Optional[int] = None,
        log_file: Optional[Path] = None,
    ):
        self.logger = logger
        self.new_level = level
        self.new_log_file = log_file

        # Store original state
        self.original_level = logger.level
        self.original_handlers = logger.handlers.copy()

    def __enter__(self):
        if self.new_level is not None:
            self.logger.setLevel(self.new_level)

        if self.new_log_file is not None:
            formatter = logging.Formatter(
                "[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler = logging.FileHandler(self.new_log_file, mode="a")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original state
        self.logger.setLevel(self.original_level)
        self.logger.handlers.clear()
        for handler in self.original_handlers:
            self.logger.addHandler(handler)


def log_section(logger: logging.Logger, title: str, width: int = 80, char: str = "="):
    """Log a section header."""
    logger.info(char * width)
    logger.info(title)
    logger.info(char * width)


def log_dict(logger: logging.Logger, data: dict, indent: int = 0, level: int = logging.INFO):
    """Log a dictionary in a readable format."""
    for key, value in data.items():
        if isinstance(value, dict):
            logger.log(level, "  " * indent + f"{key}:")
            log_dict(logger, value, indent + 1, level)
        else:
            logger.log(level, "  " * indent + f"{key}: {value}")
