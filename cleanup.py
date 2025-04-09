#!/usr/bin/env python3
"""
Thematic-AnalysisG Cache Cleanup Utility

This script cleans up all cache files created by the Thematic-AnalysisG application,
including vector database files, FAISS indices, Elasticsearch data,
Python cache directories, and log files.

Usage:
    python cleanup.py [--all] [--logs] [--vector-db] [--faiss] [--elasticsearch] [--pycache] [--dry-run]

Options:
    --all             Remove all cache files (default if no option specified)
    --logs            Remove only log files
    --vector-db       Remove only vector database files
    --faiss           Remove only FAISS index files
    --elasticsearch   Remove only Elasticsearch index files
    --pycache         Remove only Python cache directories
    --dry-run         Show what would be deleted without actually deleting
"""

import os
import sys
import shutil
import glob
import argparse
from pathlib import Path
import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("cleanup")

# Root directory of the project
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_directories(base_dir, pattern):
    """Find directories matching the pattern."""
    return [d for d in Path(base_dir).glob(pattern) if d.is_dir()]


def find_files(base_dir, pattern):
    """Find files matching the pattern."""
    return [f for f in Path(base_dir).glob(pattern) if f.is_file()]


def remove_path(path, dry_run=False):
    """Remove a file or directory."""
    try:
        if dry_run:
            logger.info(f"[DRY RUN] Would remove: {path}")
            return True
        
        if os.path.isdir(path):
            shutil.rmtree(path)
            logger.info(f"Removed directory: {path}")
        else:
            os.remove(path)
            logger.info(f"Removed file: {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to remove {path}: {str(e)}")
        return False


def clean_vector_db_files(dry_run=False):
    """Clean up vector database files."""
    data_dir = os.path.join(ROOT_DIR, "data")
    if not os.path.exists(data_dir):
        logger.info("No data directory found.")
        return 0
    
    count = 0
    # Find all vector database files
    vector_db_files = find_files(data_dir, "**/contextual_vector_db.pkl")
    
    for file_path in vector_db_files:
        if remove_path(file_path, dry_run):
            count += 1
    
    logger.info(f"Vector DB files cleaned: {count}")
    return count


def clean_faiss_indices(dry_run=False):
    """Clean up FAISS index files."""
    data_dir = os.path.join(ROOT_DIR, "data")
    if not os.path.exists(data_dir):
        logger.info("No data directory found.")
        return 0
    
    count = 0
    # Find all FAISS index files
    faiss_files = find_files(data_dir, "**/faiss_index.bin")
    
    for file_path in faiss_files:
        if remove_path(file_path, dry_run):
            count += 1
    
    logger.info(f"FAISS index files cleaned: {count}")
    return count


def clean_elasticsearch_indices(dry_run=False):
    """Clean up Elasticsearch index files."""
    data_dir = os.path.join(ROOT_DIR, "data")
    if not os.path.exists(data_dir):
        logger.info("No data directory found.")
        return 0
    
    count = 0
    # Match all directories in the data folder that are likely Elasticsearch indices
    index_dirs = find_directories(data_dir, "contextual_bm25_index_*")
    
    for dir_path in index_dirs:
        if remove_path(dir_path, dry_run):
            count += 1
    
    logger.info(f"Elasticsearch index directories cleaned: {count}")
    return count


def clean_python_cache(dry_run=False):
    """Clean up Python cache directories."""
    count = 0
    # Find all __pycache__ directories
    pycache_dirs = find_directories(ROOT_DIR, "**/__pycache__")
    
    for dir_path in pycache_dirs:
        if remove_path(dir_path, dry_run):
            count += 1
    
    # Also remove .pyc and .pyo files
    pyc_files = find_files(ROOT_DIR, "**/*.pyc")
    pyo_files = find_files(ROOT_DIR, "**/*.pyo")
    
    for file_path in pyc_files + pyo_files:
        if remove_path(file_path, dry_run):
            count += 1
    
    logger.info(f"Python cache files and directories cleaned: {count}")
    return count


def clean_logs(dry_run=False):
    """Clean up log files."""
    logs_dir = os.path.join(ROOT_DIR, "logs")
    if not os.path.exists(logs_dir):
        logger.info("No logs directory found.")
        return 0
    
    count = 0
    # Find all log files
    log_files = find_files(logs_dir, "*.log")
    log_files += find_files(logs_dir, "*.log.*")  # For rotated logs
    
    for file_path in log_files:
        if remove_path(file_path, dry_run):
            count += 1
    
    logger.info(f"Log files cleaned: {count}")
    return count


def clean_empty_data_dirs(dry_run=False):
    """Clean up empty data directories after removing files."""
    data_dir = os.path.join(ROOT_DIR, "data")
    if not os.path.exists(data_dir):
        return 0
    
    count = 0
    # Find all subdirectories in the data directory
    for root, dirs, files in os.walk(data_dir, topdown=False):
        # Skip the root data directory itself
        if root == data_dir:
            continue
        
        # Check if directory is empty (no files and no subdirectories)
        if not dirs and not files:
            if remove_path(root, dry_run):
                count += 1
    
    logger.info(f"Empty data directories cleaned: {count}")
    return count


def clean_dspy_cache(dry_run=False):
    """Clean up DSPy cache files if they exist."""
    count = 0
    
    # DSPy might cache in different locations, common ones are:
    dspy_cache_dirs = [
        os.path.join(ROOT_DIR, ".dspy_cache"),
        os.path.join(ROOT_DIR, "src", ".dspy_cache"),
        os.path.join(os.path.expanduser("~"), ".dspy_cache")
    ]
    
    for cache_dir in dspy_cache_dirs:
        if os.path.exists(cache_dir):
            if remove_path(cache_dir, dry_run):
                count += 1
    
    logger.info(f"DSPy cache directories cleaned: {count}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Thematic-AnalysisG Cache Cleanup Utility")
    parser.add_argument("--all", action="store_true", help="Remove all cache files")
    parser.add_argument("--logs", action="store_true", help="Remove only log files")
    parser.add_argument("--vector-db", action="store_true", help="Remove only vector database files")
    parser.add_argument("--faiss", action="store_true", help="Remove only FAISS index files")
    parser.add_argument("--elasticsearch", action="store_true", help="Remove only Elasticsearch index files")
    parser.add_argument("--pycache", action="store_true", help="Remove only Python cache directories")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without actually deleting")
    
    args = parser.parse_args()
    
    # If no specific option is provided, clean everything
    clean_all = args.all or not (args.logs or args.vector_db or args.faiss or args.elasticsearch or args.pycache)
    
    total_cleaned = 0
    
    logger.info("Starting cache cleanup...")
    if args.dry_run:
        logger.info("DRY RUN MODE: No files will be actually deleted")
    
    # Clean specific components based on arguments
    if clean_all or args.vector_db:
        total_cleaned += clean_vector_db_files(args.dry_run)
    
    if clean_all or args.faiss:
        total_cleaned += clean_faiss_indices(args.dry_run)
    
    if clean_all or args.elasticsearch:
        total_cleaned += clean_elasticsearch_indices(args.dry_run)
    
    if clean_all or args.pycache:
        total_cleaned += clean_python_cache(args.dry_run)
    
    if clean_all or args.logs:
        total_cleaned += clean_logs(args.dry_run)
    
    if clean_all:
        total_cleaned += clean_dspy_cache(args.dry_run)
        total_cleaned += clean_empty_data_dirs(args.dry_run)
    
    if args.dry_run:
        logger.info(f"DRY RUN COMPLETE: Would have removed {total_cleaned} items")
    else:
        logger.info(f"Cache cleanup complete. Removed {total_cleaned} items.")


if __name__ == "__main__":
    main()
