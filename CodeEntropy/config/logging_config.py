import logging
import logging.config
import os

from rich.console import Console
from rich.logging import RichHandler


class LoggingConfig:
    _console = None

    def __init__(self, folder, log_level=logging.INFO):
        log_directory = os.path.join(folder, "logs")
        os.makedirs(log_directory, exist_ok=True)

        self.log_level = log_level
        self.log_directory = log_directory
        self.program_out_path = os.path.join(log_directory, "program.out")

        self.LOGGING = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "detailed": {
                    "format": "%(asctime)s "
                    "- %(levelname)s "
                    "- %(filename)s:%(lineno)d "
                    "- %(message)s",
                },
                "simple": {
                    "format": "%(message)s",
                },
            },
            "handlers": {
                "program_out_file": {
                    "class": "logging.FileHandler",
                    "filename": self.program_out_path,
                    "formatter": "simple",
                    "level": logging.INFO,
                },
                "logfile": {
                    "class": "logging.FileHandler",
                    "filename": os.path.join(log_directory, "program.log"),
                    "formatter": "detailed",
                    "level": log_level,
                },
                "errorfile": {
                    "class": "logging.FileHandler",
                    "filename": os.path.join(log_directory, "program.err"),
                    "formatter": "simple",
                    "level": logging.ERROR,
                },
                "commandfile": {
                    "class": "logging.FileHandler",
                    "filename": os.path.join(log_directory, "program.com"),
                    "formatter": "simple",
                    "level": logging.INFO,
                },
                "mdanalysis_log": {
                    "class": "logging.FileHandler",
                    "filename": os.path.join(log_directory, "mdanalysis.log"),
                    "formatter": "detailed",
                    "level": log_level,
                },
            },
            "loggers": {
                "": {
                    "handlers": ["program_out_file", "logfile", "errorfile"],
                    "level": log_level,
                },
                "MDAnalysis": {
                    "handlers": ["mdanalysis_log"],
                    "level": log_level,
                    "propagate": False,
                },
                "commands": {
                    "handlers": ["commandfile"],
                    "level": logging.INFO,
                    "propagate": False,
                },
            },
        }

    def setup_logging(self):
        # Configure file-based logging
        logging.config.dictConfig(self.LOGGING)

        rich_handler = RichHandler(
            console=LoggingConfig._console,
            markup=True,
            rich_tracebacks=True,
            show_time=True,
            show_level=True,
            show_path=False,
        )
        rich_handler.setLevel(logging.INFO)

        # Attach RichHandler to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(rich_handler)

        return logging.getLogger(__name__)

    @classmethod
    def get_console(cls):
        if cls._console is None:
            cls._console = Console()
        return cls._console

    def update_logging_level(self, log_level):
        # Update the root logger level
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        for handler in root_logger.handlers:
            handler.setLevel(
                log_level if isinstance(handler, logging.FileHandler) else logging.INFO
            )

        # Update all other loggers and their handlers
        for logger_name in self.LOGGING["loggers"]:
            logger = logging.getLogger(logger_name)
            logger.setLevel(log_level)
            for handler in logger.handlers:
                handler.setLevel(
                    log_level
                    if isinstance(handler, logging.FileHandler)
                    else logging.INFO
                )
