import os
import shutil
import logging
from elasticsearch import Elasticsearch
from typing import List, Optional
import json

def setup_logging() -> logging.Logger:
    """Configure and return a logger instance."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def clear_elasticsearch_cache(logger: logging.Logger, es_host: str = "http://localhost:9200") -> None:
    """Clear Elasticsearch indices."""
    try:
        es = Elasticsearch(es_host)
        if es.ping():
            indices_to_delete = ["contextual_bm25_index"]
            for index in indices_to_delete:
                if es.indices.exists(index=index):
                    es.indices.delete(index=index)
                    logger.info(f"Deleted Elasticsearch index: {index}")
            logger.info("Elasticsearch cache cleared successfully")
    except Exception as e:
        logger.error(f"Error clearing Elasticsearch cache: {e}")

def remove_file_or_directory(path: str, logger: logging.Logger) -> None:
    """Remove a file or directory with proper error handling."""
    try:
        if os.path.isfile(path):
            os.remove(path)
            logger.info(f"Removed file: {path}")
        elif os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
            logger.info(f"Removed directory: {path}")
    except Exception as e:
        logger.error(f"Error removing {path}: {e}")

def clear_pickle_files(data_dir: str, logger: logging.Logger) -> None:
    """Remove all pickle files in the specified directory and its subdirectories."""
    try:
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith((".pkl", ".pickle")):
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    logger.info(f"Removed pickle file: {file_path}")
    except Exception as e:
        logger.error(f"Error clearing pickle files: {e}")

def clear_temporary_files(data_dir: str, logger: logging.Logger) -> None:
    """Remove temporary files like .pyc, .pyo, and __pycache__ directories."""
    try:
        for root, dirs, files in os.walk(data_dir):
            # Remove __pycache__ directories
            if "__pycache__" in dirs:
                cache_dir = os.path.join(root, "__pycache__")
                shutil.rmtree(cache_dir, ignore_errors=True)
                logger.info(f"Removed __pycache__ directory: {cache_dir}")
            
            # Remove .pyc and .pyo files
            for file in files:
                if file.endswith((".pyc", ".pyo")):
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    logger.info(f"Removed compiled Python file: {file_path}")
    except Exception as e:
        logger.error(f"Error clearing temporary files: {e}")

def create_directories(directories: List[str], logger: logging.Logger) -> None:
    """Create necessary directories."""
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {e}")

def clear_model_cache(cache_dir: str, logger: logging.Logger) -> None:
    """Clear any cached model files or artifacts."""
    try:
        model_paths = [
            os.path.join(cache_dir, "models"),
            os.path.join(cache_dir, "optimized_models"),
            os.path.join(cache_dir, "checkpoints")
        ]
        for path in model_paths:
            if os.path.exists(path):
                shutil.rmtree(path, ignore_errors=True)
                logger.info(f"Cleared model cache: {path}")
    except Exception as e:
        logger.error(f"Error clearing model cache: {e}")

def clear_output_files(output_dir: str, logger: logging.Logger) -> None:
    """Clear generated output files."""
    try:
        # Clear JSON output files
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.endswith((".json", ".jsonl")):
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    logger.info(f"Removed output file: {file_path}")
    except Exception as e:
        logger.error(f"Error clearing output files: {e}")

def clear_all_cache(base_dir: Optional[str] = None) -> None:
    """Main function to clear all cache and temporary files."""
    logger = setup_logging()
    
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define paths relative to base directory
    data_dir = os.path.join(base_dir, "data")
    cache_dir = os.path.join(base_dir, "cache")
    logs_dir = os.path.join(base_dir, "logs")
    output_dir = os.path.join(base_dir, "output")
    
    # 1. Clear Elasticsearch cache
    clear_elasticsearch_cache(logger)
    
    # 2. Clear data directories
    data_paths = [
        os.path.join(data_dir, "contextual_db"),
        os.path.join(data_dir, "optimized_program.json"),
        logs_dir,
        cache_dir,
        os.path.join(data_dir, "contextual_db", "faiss_index.bin"),
        output_dir
    ]
    
    for path in data_paths:
        remove_file_or_directory(path, logger)
    
    # 3. Clear pickle files
    clear_pickle_files(data_dir, logger)
    
    # 4. Clear temporary Python files
    clear_temporary_files(base_dir, logger)
    
    # 5. Clear model cache
    clear_model_cache(cache_dir, logger)
    
    # 6. Clear output files
    clear_output_files(output_dir, logger)
    
    # 7. Recreate necessary directories
    required_dirs = [
        data_dir,
        os.path.join(data_dir, "contextual_db"),
        logs_dir,
        cache_dir,
        output_dir
    ]
    create_directories(required_dirs, logger)
    
    logger.info("Cache clearing completed successfully")

if __name__ == "__main__":
    clear_all_cache()