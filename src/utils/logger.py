import logging
import os
import sys
from datetime import datetime

from src.core.config import Constants

class LoggingService:
    @staticmethod
    def setup_logger(name: str) -> logging.Logger:
        """Configura e retorna um logger com formato e handlers padrão."""

        log_dir = Constants.LOGS_DIR
        log_dir.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        if logger.handlers:
            return logger

        formatter = logging.Formatter(
            "%(levelname)s - %(name)s - %(message)s"
        )

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        log_filename = f"app_{timestamp}.log"

        # File Handler
        file_handler = logging.FileHandler(log_dir / log_filename, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger