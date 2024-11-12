# clear_cache.py

import os
import shutil
import logging
from elasticsearch import Elasticsearch

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def clear_all_cache():
    logger = setup_logging()
    
    # 1. Clear Elasticsearch index
    try:
        es = Elasticsearch("http://localhost:9200")
        if es.ping():
            es.indices.delete(index="contextual_bm25_index", ignore=[400, 404])
            logger.info("Elasticsearch index cleared successfully")
    except Exception as e:
        logger.error(f"Error clearing Elasticsearch index: {e}")

    # 2. Clear data directory
    data_paths = [
        "./data/contextual_db",  # Vector DB data
        "./data/optimized_program.json",  # Optimized program cache
        "./logs",  # Log files
        "./cache"  # Any other cache directory
    ]

    for path in data_paths:
        try:
            if os.path.isfile(path):
                os.remove(path)
                logger.info(f"Removed file: {path}")
            elif os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                logger.info(f"Removed directory: {path}")
        except Exception as e:
            logger.error(f"Error removing {path}: {e}")

    # 3. Clear FAISS index
    faiss_index_path = "./data/contextual_db/faiss_index.bin"
    try:
        if os.path.exists(faiss_index_path):
            os.remove(faiss_index_path)
            logger.info("FAISS index cleared successfully")
    except Exception as e:
        logger.error(f"Error clearing FAISS index: {e}")

    # 4. Clear any pickle files
    try:
        for root, dirs, files in os.walk("./data"):
            for file in files:
                if file.endswith(".pkl"):
                    os.remove(os.path.join(root, file))
                    logger.info(f"Removed pickle file: {os.path.join(root, file)}")
    except Exception as e:
        logger.error(f"Error clearing pickle files: {e}")

    # 5. Recreate necessary directories
    required_dirs = [
        "./data",
        "./data/contextual_db",
        "./logs"
    ]

    for directory in required_dirs:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {e}")

    logger.info("Cache clearing completed")

if __name__ == "__main__":
    clear_all_cache()