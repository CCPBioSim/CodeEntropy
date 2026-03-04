"""Logging configuration utilities for CodeEntropy.

This module configures consistent logging across the project with:

- Rich console output (with tracebacks) for human-readable terminal logs
- File handlers for main logs, error-only logs, command logs, and MDAnalysis logs
- A singleton Rich Console instance with recording enabled, so terminal output
  can be exported to disk at the end of a run

The design keeps responsibilities separated:
- ErrorFilter: filter logic only
- LoggingConfig: handler creation, logger wiring, and exporting recorded output
"""

from __future__ import annotations

import logging
import os

from rich.console import Console
from rich.logging import RichHandler


class ErrorFilter(logging.Filter):
    """Allow only ERROR and CRITICAL log records.

    This filter is intended for the error file handler so that the file contains
    only high-severity records and does not include DEBUG/INFO/WARNING output.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Return True if the record should be logged.

        Args:
            record: The log record being evaluated.

        Returns:
            True if record.levelno >= logging.ERROR, otherwise False.
        """
        return record.levelno >= logging.ERROR


class LoggingConfig:
    """Configure project logging with Rich console output and file handlers.

    This class wires a set of handlers onto the root logger and a few named
    loggers. It also provides a singleton Rich Console instance with recording
    enabled so that all console output can be exported to a text file later.

    Attributes:
        log_dir: Directory where log files are written.
        level: Base logging level for the root logger and file handlers.
        console: Shared Rich Console instance used by RichHandler.
        handlers: Mapping of handler name to handler instance.
    """

    _console: Console | None = None

    @classmethod
    def get_console(cls) -> Console:
        """Get or create the singleton Rich Console with recording enabled.

        Returns:
            A Rich Console instance that prints to terminal and records output.
        """
        if cls._console is None:
            cls._console = Console(record=True)
        return cls._console

    def __init__(self, folder: str, level: int = logging.INFO) -> None:
        """Initialize logging configuration.

        Args:
            folder: Base folder where the 'logs' directory will be created.
            level: Logging level for the root logger and most file handlers.
        """
        self.log_dir = os.path.join(folder, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        self.level = level
        self.console = self.get_console()
        self.handlers: dict[str, logging.Handler] = {}

        self._setup_handlers()

    def _setup_handlers(self) -> None:
        """Create handlers and assign formatters/levels/filters."""
        paths = {
            "main": os.path.join(self.log_dir, "program.log"),
            "error": os.path.join(self.log_dir, "program.err"),
            "command": os.path.join(self.log_dir, "program.com"),
            "mdanalysis": os.path.join(self.log_dir, "mdanalysis.log"),
        }

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )

        rich_handler = RichHandler(
            console=self.console,
            markup=True,
            rich_tracebacks=True,
            show_time=True,
            show_level=True,
            show_path=False,
        )
        rich_handler.setLevel(logging.INFO)

        main_handler = logging.FileHandler(paths["main"])
        main_handler.setLevel(self.level)
        main_handler.setFormatter(formatter)

        error_handler = logging.FileHandler(paths["error"])
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        error_handler.addFilter(ErrorFilter())

        command_handler = logging.FileHandler(paths["command"])
        command_handler.setLevel(logging.INFO)
        command_handler.setFormatter(formatter)

        mdanalysis_handler = logging.FileHandler(paths["mdanalysis"])
        mdanalysis_handler.setLevel(self.level)
        mdanalysis_handler.setFormatter(formatter)

        self.handlers = {
            "rich": rich_handler,
            "main": main_handler,
            "error": error_handler,
            "command": command_handler,
            "mdanalysis": mdanalysis_handler,
        }

    def configure(self) -> logging.Logger:
        """Attach configured handlers to the appropriate loggers.

        This method:
        - Attaches rich/main/error handlers to the root logger
        - Attaches command handler to the 'commands' logger (non-propagating)
        - Attaches MDAnalysis handler to the 'MDAnalysis' logger (non-propagating)

        Returns:
            A logger for the current module.
        """
        root = logging.getLogger()
        root.setLevel(self.level)

        self._add_handler_once(root, self.handlers["rich"])
        self._add_handler_once(root, self.handlers["main"])
        self._add_handler_once(root, self.handlers["error"])

        commands_logger = logging.getLogger("commands")
        commands_logger.setLevel(logging.INFO)
        commands_logger.propagate = False
        self._add_handler_once(commands_logger, self.handlers["command"])

        mda_logger = logging.getLogger("MDAnalysis")
        mda_logger.setLevel(self.level)
        mda_logger.propagate = False
        self._add_handler_once(mda_logger, self.handlers["mdanalysis"])

        return logging.getLogger(__name__)

    @staticmethod
    def _add_handler_once(logger_obj: logging.Logger, handler: logging.Handler) -> None:
        """Attach a handler to a logger only if it isn't already attached.

        Args:
            logger_obj: Logger to modify.
            handler: Handler to attach.
        """
        if handler not in logger_obj.handlers:
            logger_obj.addHandler(handler)

    def set_level(self, log_level: int) -> None:
        """Update logging levels for root and named loggers.

        Notes:
            - FileHandlers are set to the new log_level.
            - RichHandler is kept at INFO (or higher) for cleaner console output.

        Args:
            log_level: New logging level (e.g., logging.DEBUG).
        """
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)

        self._set_logger_handlers_level(root_logger, log_level)

        for logger_name in ("commands", "MDAnalysis"):
            named_logger = logging.getLogger(logger_name)
            named_logger.setLevel(log_level)
            self._set_logger_handlers_level(named_logger, log_level)

    @staticmethod
    def _set_logger_handlers_level(logger_obj: logging.Logger, log_level: int) -> None:
        """Apply level rules to all handlers on a logger.

        Args:
            logger_obj: Logger whose handlers should be updated.
            log_level: Target logging level for file handlers.
        """
        for handler in logger_obj.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.setLevel(log_level)
            else:
                handler.setLevel(logging.INFO)

    def export_console(self, filename: str = "program_output.txt") -> None:
        """Save recorded console output to a file.

        Args:
            filename: Output filename inside the log directory.
        """
        output_path = os.path.join(self.log_dir, filename)
        os.makedirs(self.log_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.console.export_text())
