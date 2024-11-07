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

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(query1: str, query2: str) -> float:
    vectorizer = TfidfVectorizer().fit_transform([query1, query2])
    vectors = vectorizer.toarray()
    similarity = cosine_similarity([vectors[0]], [vectors[1]])[0][0]
    return similarity
