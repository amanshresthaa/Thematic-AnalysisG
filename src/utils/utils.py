# src/utils/utils.py

import logging
from functools import wraps
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

def handle_assertion(func, *args, **kwargs):
    """
    Handles exceptions in assertion functions.
    Logs start, pass, or fail.
    """
    func_name = func.__name__
    start_time = time.time()
    logger.debug(f"Starting assertion check: '{func_name}'")
    try:
        func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        logger.debug(f"Assertion in '{func_name}' passed successfully in {elapsed_time:.4f}s.")
    except AssertionError as ae:
        elapsed_time = time.time() - start_time
        logger.error(f"Assertion in '{func_name}' failed in {elapsed_time:.4f}s. Error: {ae}")
        raise  # Re-raise the exception
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error during assertion in '{func_name}' in {elapsed_time:.4f}s: {e}", exc_info=True)
        raise

def check_answer_length(answer: str, max_length: int = 500) -> bool:
    """
    Checks if the answer length is within the specified limit.

    Args:
        answer (str): The generated answer to evaluate.
        max_length (int, optional): The maximum allowed length for the answer. Defaults to 500.

    Returns:
        bool: True if the answer length is within the limit, False otherwise.
    """
    return len(answer) <= max_length

def compute_similarity(query1: str, query2: str) -> float:
    """
    Computes cosine similarity between two queries using TF-IDF vectorization.

    Args:
        query1 (str): The first query string.
        query2 (str): The second query string.

    Returns:
        float: Cosine similarity score between query1 and query2.
    """
    try:
        vectorizer = TfidfVectorizer().fit_transform([query1, query2])
        vectors = vectorizer.toarray()
        similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
        return similarity
    except Exception as e:
        logger.error(f"Error computing similarity: {e}", exc_info=True)
        return 0.0
