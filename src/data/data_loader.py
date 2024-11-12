import json
import logging
from typing import List, Dict, Any

from src.utils.logger import setup_logging
logger = logging.getLogger(__name__)

class JSONLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Dict[str, Any]]:
        data = []
        try:
            if self.file_path.endswith('.jsonl'):
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    for line_number, line in enumerate(f, start=1):
                        if line.strip():
                            try:
                                obj = json.loads(line)
                                data.append(obj)
                            except json.JSONDecodeError as e:
                                logger.error(f"JSON parsing error in file '{self.file_path}' at line {line_number}: {e}")
                logger.info(f"Loaded JSONL file '{self.file_path}' with {len(data)} entries successfully.")
            else:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Loaded JSON file '{self.file_path}' successfully with {len(data)} entries.")
            return data
        except FileNotFoundError:
            logger.error(f"Error loading JSON file '{self.file_path}': File does not exist.")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error loading JSON file '{self.file_path}': {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error loading JSON file '{self.file_path}': {e}")
            return []

def load_codebase_chunks(file_path: str) -> List[Dict[str, Any]]:
    logger.debug(f"Loading codebase chunks from '{file_path}'.")
    loader = JSONLoader(file_path)
    return loader.load()

def load_queries(file_path: str) -> List[Dict[str, Any]]:
    logger.debug(f"Loading queries from '{file_path}'.")
    loader = JSONLoader(file_path)
    return loader.load()
