import logging
import logging.config
import os


class LoggingConfig:
    def __init__(self, folder, log_level=logging.INFO):
        log_directory = os.path.join(folder, "logs")
        os.makedirs(log_directory, exist_ok=True)

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
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "simple",
                    "level": logging.INFO,
                },
                "stdout": {
                    "class": "logging.FileHandler",
                    "filename": os.path.join(log_directory, "program.out"),
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
                    "handlers": ["console", "stdout", "logfile", "errorfile"],
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
        logging.config.dictConfig(self.LOGGING)
        logging.getLogger("MDAnalysis")
        logging.getLogger("commands")
        return logging.getLogger(__name__)

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
