import logging
import logging.config
import sys

from llmq.core.config import get_config


def setup_logging(component: str = "llmq", structured: bool = False) -> logging.Logger:
    """
    Set up logging for the application.

    Args:
        component: Component name for the logger
        structured: Whether to use structured JSON logging (for workers)

    Returns:
        Configured logger instance
    """
    config = get_config()

    if structured:
        # JSON structured logging for workers
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "format": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "component": "%(name)s", "message": "%(message)s"}',
                    "datefmt": "%Y-%m-%dT%H:%M:%SZ",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "stream": sys.stdout,
                    "formatter": "json",
                }
            },
            "loggers": {
                component: {
                    "level": config.log_level,
                    "handlers": ["console"],
                    "propagate": False,
                }
            },
        }
    else:
        # Human-readable logging for CLI
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "stream": sys.stderr,
                    "formatter": "standard",
                }
            },
            "loggers": {
                component: {
                    "level": config.log_level,
                    "handlers": ["console"],
                    "propagate": False,
                }
            },
        }

    logging.config.dictConfig(logging_config)
    return logging.getLogger(component)
