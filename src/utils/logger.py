import logging

from rich.logging import RichHandler


def create_logger(name="main_logger"):
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = RichHandler()
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    return logger
