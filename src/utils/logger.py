# src/utils/logger.py:

import logging
import logging.config
import os
import yaml
from typing import Optional
from functools import wraps
import time

def setup_logging(
    default_path: str = 'config/logging_config.yaml',
    default_level: int = logging.INFO,
    env_key: str = 'LOG_CFG'
) -> None:
    """
    Setup logging configuration with enhanced error handling and directory creation.
    
    Args:
        default_path: Path to the logging configuration file
        default_level: Default logging level if config file is not found
        env_key: Environment variable that can be used to override the config path
    """
    try:
        path = os.getenv(env_key, default_path)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # Create log directories for all handlers
            handlers = config.get('handlers', {})
            for handler in handlers.values():
                if 'filename' in handler:
                    log_dir = os.path.dirname(handler['filename'])
                    if log_dir:
                        os.makedirs(log_dir, exist_ok=True)
            
            logging.config.dictConfig(config)
            logging.info(f"Logging configuration loaded from {path}")
        else:
            logging.basicConfig(level=default_level)
            logging.warning(f"Logging config file not found at {path}. Using basic config.")
    except Exception as e:
        logging.basicConfig(level=default_level)
        logging.error(f"Error in logging configuration: {str(e)}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name and adds extra handlers if needed.
    
    Args:
        name: Name for the logger
        
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Add a null handler if no handlers exist
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    
    return logger

def log_execution_time(logger: Optional[logging.Logger] = None):
    """
    Decorator to log function execution time.
    
    Args:
        logger: Logger instance to use. If None, creates a new logger.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(func.__module__)
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.debug(
                    f"Function '{func.__name__}' executed in {execution_time:.2f} seconds"
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"Function '{func.__name__}' failed after {execution_time:.2f} seconds. "
                    f"Error: {str(e)}"
                )
                raise
        return wrapper
    return decorator
