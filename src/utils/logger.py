# File: src/utils/logger.py

import logging
import logging.config
import os
import yaml

def setup_logging(default_path='config/logging_config.yaml', default_level=logging.INFO):
    """
    Setup logging configuration from a YAML file.
    Ensures that the log directory exists before configuring handlers.
    """
    if os.path.exists(default_path):
        with open(default_path, 'r') as f:
            config = yaml.safe_load(f.read())

        # Extract all file handler paths to ensure directories exist
        handlers = config.get('handlers', {})
        for handler_name, handler in handlers.items():
            if 'filename' in handler:
                log_file = handler['filename']
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    try:
                        os.makedirs(log_dir, exist_ok=True)
                        print(f"Created log directory: {log_dir}")
                    except Exception as e:
                        print(f"Failed to create log directory '{log_dir}': {e}")

        # Apply the logging configuration
        logging.config.dictConfig(config)
    else:
        # If the logging configuration file is missing, use basic configuration
        logging.basicConfig(level=default_level)
        logging.warning(f"Logging configuration file not found at '{default_path}'. Using basic configuration.")

# Initialize logging when this module is imported
setup_logging()
