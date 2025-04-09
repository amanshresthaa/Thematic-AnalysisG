# Add this to src/utils/history_utils.py

import os
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class HistoryManager:
    def __init__(self, base_dir='interaction_histories'):
        self.base_dir = base_dir
        self._ensure_directory()

    def _ensure_directory(self):
        """Ensures the history directory exists."""
        try:
            os.makedirs(self.base_dir, exist_ok=True)
            logger.info(f"Ensured history directory exists at: {self.base_dir}")
        except Exception as e:
            logger.error(f"Error creating history directory: {e}")

    def save_history(self, history, prefix='quotation'):
        """Saves a history entry with timestamp."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_interaction_{timestamp}.json"
            filepath = os.path.join(self.base_dir, filename)

            history_data = {
                'timestamp': timestamp,
                'history': history
            }

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2)
            
            logger.info(f"Saved interaction history to: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving history: {e}")
            return None

    def get_latest_history(self, prefix='quotation'):
        """Gets the most recent history file."""
        try:
            files = [f for f in os.listdir(self.base_dir) if f.startswith(prefix)]
            if not files:
                return None
            latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(self.base_dir, x)))
            with open(os.path.join(self.base_dir, latest_file), 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error getting latest history: {e}")
            return None