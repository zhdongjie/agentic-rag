import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from core.utils.path_utils import get_abs_path


def get_logger(
        name: str = "Agentic-RAG",
        log_dir: str | Path = get_abs_path("logs"),
        console_level: int = logging.INFO,
        to_console: bool = True,
        file_level: int = logging.DEBUG,
        to_file: bool = True,
        when: str = "midnight",
        backup_count: int = 7,
) -> logging.Logger:
    """
    Create and configure a reusable logger instance.

    This function initializes a logger with optional console and file handlers.
    It ensures that handlers are added only once per logger instance to avoid
    duplicate log outputs.

    Features:
        - Console logging with configurable level
        - File logging with timed rotation
        - Customizable log format
        - Automatic log directory creation

    Args:
        name (str): Name of the logger. Used to identify the logger instance
            and as part of the log file name.
        log_dir (str | Path): Directory where log files will be stored.
        console_level (int): Logging level for console output
            (e.g., logging.INFO, logging.DEBUG).
        to_console (bool): Whether to enable console logging.
        file_level (int): Logging level for file output.
        to_file (bool): Whether to enable file logging.
        when (str): Time interval for log file rotation.
            Common values:
                - 'midnight': rotate daily
                - 'S', 'M', 'H', 'D': seconds, minutes, hours, days
        backup_count (int): Number of backup log files to keep.

    Returns:
        logging.Logger: Configured logger instance.

    Behavior:
        - If the logger already has handlers, the existing instance is returned
          without adding new handlers (prevents duplicate logs).
        - Log files are automatically rotated based on the specified interval.
        - Log directory is created if it does not exist.

    Example:
        >>> logger = get_logger(name="my_app")
        >>> logger.info("Application started")

    Notes:
        - The logger level is set to DEBUG to allow all messages through;
          filtering is controlled by handler levels.
        - File logs are encoded in UTF-8.
    """
    instance = logging.getLogger(name)

    if instance.handlers and instance.hasHandlers():
        return instance

    instance.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )

    if to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(console_level)
        instance.addHandler(console_handler)

    if to_file:
        log_root = Path(log_dir)
        log_root.mkdir(parents=True, exist_ok=True)

        log_path = log_root / f"{name}.log"

        file_handler = TimedRotatingFileHandler(
            filename=str(log_path),
            when=when,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(file_level)
        instance.addHandler(file_handler)

    return instance


logger = get_logger()
