import logging
import os


_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DEFAULT_LEVEL = "INFO"


def setup_logging() -> None:
    if logging.getLogger().handlers:
        return
    level_name = os.getenv("LOG_LEVEL", _DEFAULT_LEVEL).upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format=_FORMAT)


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
