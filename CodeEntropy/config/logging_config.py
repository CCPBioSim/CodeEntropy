import logging
import os

from rich.console import Console
from rich.logging import RichHandler


class ErrorFilter(logging.Filter):
    """
    Logging filter that only allows records with level ERROR or higher.

    This ensures that the attached handler only processes error and critical logs,
    filtering out all lower level messages such as DEBUG and INFO.
    """

    def filter(self, record):
        return record.levelno >= logging.ERROR


class LoggingConfig:
    """
    Configures logging with Rich console output and multiple file handlers.
    Provides a single Rich Console instance that records all output for later export.

    Attributes:
        _console (Console): Shared Rich Console instance with output recording enabled.
        log_dir (str): Directory path to store log files.
        level (int): Logging level (e.g., logging.INFO).
        console (Console): The Rich Console instance used for output and logging.
        handlers (dict): Dictionary of logging handlers for console and files.
    """

    _console = None  # Shared Console with recording enabled

    @classmethod
    def get_console(cls):
        """
        Get or create a singleton Rich Console instance with recording enabled.

        Returns:
            Console: Rich Console instance that prints to terminal and records output.
        """
        if cls._console is None:
            # Create console that records output for later export
            cls._console = Console(record=True)
        return cls._console

    def __init__(self, folder, level=logging.INFO):
        """
        Initialize the logging configuration.

        Args:
            folder (str): Base folder where 'logs' directory will be created.
            level (int): Logging level (default: logging.INFO).
        """
        self.log_dir = os.path.join(folder, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.level = level

        # Use the single recorded console instance
        self.console = self.get_console()

        self._setup_handlers()

    def _setup_handlers(self):
        paths = {
            "main": os.path.join(self.log_dir, "program.log"),
            "error": os.path.join(self.log_dir, "program.err"),
            "command": os.path.join(self.log_dir, "program.com"),
            "mdanalysis": os.path.join(self.log_dir, "mdanalysis.log"),
        }

        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )

        self.handlers = {
            "rich": RichHandler(
                console=self.console,
                markup=True,
                rich_tracebacks=True,
                show_time=True,
                show_level=True,
                show_path=False,
            ),
            "main": logging.FileHandler(paths["main"]),
            "error": logging.FileHandler(paths["error"]),
            "command": logging.FileHandler(paths["command"]),
            "mdanalysis": logging.FileHandler(paths["mdanalysis"]),
        }

        self.handlers["rich"].setLevel(logging.INFO)
        self.handlers["main"].setLevel(self.level)
        self.handlers["error"].setLevel(logging.ERROR)
        self.handlers["command"].setLevel(logging.INFO)
        self.handlers["mdanalysis"].setLevel(self.level)

        for name, handler in self.handlers.items():
            if name != "rich":
                handler.setFormatter(formatter)

        # Add filter to error handler to ensure only ERROR and above are logged
        self.handlers["error"].addFilter(ErrorFilter())

    def setup_logging(self):
        """
        Configure the root logger and specific loggers with the prepared handlers.

        Returns:
            logging.Logger: Logger instance for the current module (__name__).
        """
        root = logging.getLogger()
        root.setLevel(self.level)
        root.addHandler(self.handlers["rich"])
        root.addHandler(self.handlers["main"])
        root.addHandler(self.handlers["error"])

        logging.getLogger("commands").addHandler(self.handlers["command"])
        logging.getLogger("commands").setLevel(logging.INFO)
        logging.getLogger("commands").propagate = False

        logging.getLogger("MDAnalysis").addHandler(self.handlers["mdanalysis"])
        logging.getLogger("MDAnalysis").setLevel(self.level)
        logging.getLogger("MDAnalysis").propagate = False

        return logging.getLogger(__name__)

    def update_logging_level(self, log_level):
        """
        Update the logging level for the root logger and specific sub-loggers.

        Args:
            log_level (int): New logging level (e.g., logging.DEBUG, logging.WARNING).
        """
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        for handler in root_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                handler.setLevel(log_level)
            else:
                # Keep RichHandler at INFO or higher for nicer console output
                handler.setLevel(logging.INFO)

        for logger_name in ["commands", "MDAnalysis"]:
            logger = logging.getLogger(logger_name)
            logger.setLevel(log_level)
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.setLevel(log_level)
                else:
                    handler.setLevel(logging.INFO)

    def save_console_log(self, filename="program_output.txt"):
        """
        Save all recorded console output to a text file.

        Args:
            filename (str): Name of the file to write console output to.
                            Defaults to 'program_output.txt' in the logs directory.
        """
        output_path = os.path.join(self.log_dir, filename)
        os.makedirs(self.log_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(self.console.export_text())
