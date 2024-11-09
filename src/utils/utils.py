# File: /Users/amankumarshrestha/Downloads/Thematic-AnalysisE/src/utils/utils.py

import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

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
