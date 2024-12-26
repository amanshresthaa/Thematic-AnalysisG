# processing/logger.py
import logging

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger configured with the given name.
    You can configure different handlers/formatters here as needed.
    """
    logger = logging.getLogger(name)

    # If desired, configure or extend the logger format/level here
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger
