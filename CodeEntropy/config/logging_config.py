import logging
import logging.config
import os


class LoggingConfig:
    def __init__(self, folder, default_level=logging.INFO):
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
                    "level": "INFO",
                },
                "stdout": {
                    "class": "logging.FileHandler",
                    "filename": os.path.join(log_directory, "program.out"),
                    "formatter": "simple",
                    "level": "INFO",
                },
                "logfile": {
                    "class": "logging.FileHandler",
                    "filename": os.path.join(log_directory, "program.log"),
                    "formatter": "detailed",
                    "level": "DEBUG",
                },
                "errorfile": {
                    "class": "logging.FileHandler",
                    "filename": os.path.join(log_directory, "program.err"),
                    "formatter": "detailed",
                    "level": "ERROR",
                },
                "commandfile": {
                    "class": "logging.FileHandler",
                    "filename": os.path.join(log_directory, "program.com"),
                    "formatter": "simple",
                    "level": "INFO",
                },
                "mdanalysis_log": {
                    "class": "logging.FileHandler",
                    "filename": os.path.join(log_directory, "mdanalysis.log"),
                    "formatter": "detailed",
                    "level": "DEBUG",
                },
            },
            "loggers": {
                "": {
                    "handlers": ["console", "stdout", "logfile", "errorfile"],
                    "level": default_level,
                },
                "MDAnalysis": {
                    "handlers": ["mdanalysis_log"],
                    "level": "DEBUG",
                    "propagate": False,
                },
                "commands": {
                    "handlers": ["commandfile"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }

    def setup_logging(self):
        logging.config.dictConfig(self.LOGGING)
        logging.getLogger("MDAnalysis")
        logging.getLogger("commands")
        return logging.getLogger(__name__)
