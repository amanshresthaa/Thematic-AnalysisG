# chunker/utils/logger.py
import logging
import logging.config
import os
import yaml

def setup_logging(default_level=logging.INFO):
    """Setup basic logging configuration"""
    logging.basicConfig(
        level=default_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)