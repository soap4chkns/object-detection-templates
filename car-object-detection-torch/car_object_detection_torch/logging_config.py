import logging
import logging.config
import sys


def setup_logging(
    log_level: str = "INFO",
) -> None:
    # Configure handlers
    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,
            "formatter": "simple",
            "level": log_level,
        }
    }

    # Logging configuration dictionary
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "simple": {
                "format": "%(levelname)s - %(message)s",
            },
        },
        "handlers": handlers,
        "root": {
            "level": log_level,
            "handlers": list(handlers.keys()),
        },
        "loggers": {
            # Configure specific loggers if needed
            "src": {
                "level": log_level,
                "handlers": list(handlers.keys()),
                "propagate": False,
            },
            # Suppress noisy third-party loggers
            "PIL": {
                "level": "WARNING",
            },
            "matplotlib": {
                "level": "WARNING",
            },
        },
    }

    # Apply the configuration
    logging.config.dictConfig(config)

    # Log the setup
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {log_level}")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
