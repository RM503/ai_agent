import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_LOG_PATH = ROOT_DIR / "logs" / "app.log"
DEFAULT_LOG_PATH.mkdir(parents=True, exist_ok=True)

def configure_logging(
    level: str = "INFO",
    log_to_file: bool = False,
    file_path: str = DEFAULT_LOG_PATH
) -> None:
    level = getattr(logging, os.getenv("LOG_LEVEL", level).upper(), logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger("agent")
    logger.setLevel(level)
    logger.propagate = False
    logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_to_file:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file_handler = RotatingFileHandler(file_path, maxBytes=5_000_000, backupCount=3)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"agent.{name}")