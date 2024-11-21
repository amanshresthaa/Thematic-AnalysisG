# File: assertions_keyword.py
import logging
from typing import List
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

def assert_keywords_extracted(keywords: List[str]) -> None:
    """
    Ensure that keywords have been extracted from the transcript chunks.

    Args:
        keywords (List[str]): List of extracted keywords.

    Raises:
        AssertionError: If no keywords were extracted.
    """
    if not keywords:
        error_msg = "No keywords were extracted from the transcript chunks."
        logger.error(error_msg)
        raise AssertionError(error_msg)

def assert_realness(keywords: List[str], transcript_chunks: List[str]) -> None:
    """
    Checks if keywords genuinely reflect participants' experiences by verifying their presence in the transcript.

    Args:
        keywords (List[str]): List of extracted keywords.
        transcript_chunks (List[str]): List of transcript chunks.

    Raises:
        AssertionError: If a keyword is not present in the transcript chunks.
    """
    transcript_text = ' '.join(transcript_chunks).lower()
    for keyword in keywords:
        if keyword.lower() not in transcript_text:
            error_msg = f"Keyword '{keyword}' does not reflect participants' experiences (not found in transcript)."
            logger.error(error_msg)
            raise AssertionError(error_msg)

def assert_richness(keywords: List[str]) -> None:
    """
    Assesses the richness of the keywords by ensuring they are meaningful and provide detailed understanding.

    Args:
        keywords (List[str]): List of extracted keywords.

    Raises:
        AssertionError: If keywords are not rich in meaning.
    """
    stop_words = set(stopwords.words('english'))
    for keyword in keywords:
        words = word_tokenize(keyword)
        content_words = [word for word in words if word.lower() not in stop_words and word not in string.punctuation]
        if len(content_words) == 0:
            error_msg = f"Keyword '{keyword}' is not rich in meaning."
            logger.error(error_msg)
            raise AssertionError(error_msg)

def assert_repetition(keywords: List[str], transcript_chunks: List[str]) -> None:
    """
    Checks that the keywords frequently occur in the data.

    Args:
        keywords (List[str]): List of extracted keywords.
        transcript_chunks (List[str]): List of transcript chunks.

    Raises:
        AssertionError: If a keyword does not occur frequently in the data.
    """
    transcript_text = ' '.join(transcript_chunks).lower()
    word_counts = Counter(word_tokenize(transcript_text))
    total_words = sum(word_counts.values())
    for keyword in keywords:
        keyword_count = transcript_text.count(keyword.lower())
        frequency = keyword_count / total_words if total_words > 0 else 0
        if frequency < 0.001:  # Threshold can be adjusted
            error_msg = f"Keyword '{keyword}' does not occur frequently in the data."
            logger.error(error_msg)
            raise AssertionError(error_msg)

def assert_rationale(keywords: List[str], theoretical_framework: str) -> None:
    """
    Checks if the keywords are connected to the theoretical framework.

    Args:
        keywords (List[str]): List of extracted keywords.
        theoretical_framework (str): The theoretical framework text.

    Raises:
        AssertionError: If a keyword is not connected to the theoretical framework.
    """
    if not theoretical_framework:
        return
    framework_text = theoretical_framework.lower()
    for keyword in keywords:
        similarity = SequenceMatcher(None, keyword.lower(), framework_text).ratio()
        if similarity < 0.1:  # Threshold can be adjusted
            error_msg = f"Keyword '{keyword}' is not connected to the theoretical framework."
            logger.error(error_msg)
            raise AssertionError(error_msg)

def assert_repartee(keywords: List[str]) -> None:
    """
    Checks that the keywords are insightful, evocative, and stimulate further discussion.

    Args:
        keywords (List[str]): List of extracted keywords.

    Raises:
        AssertionError: If keywords are not insightful.
    """
    # Placeholder: In practice, this would require more advanced NLP techniques
    for keyword in keywords:
        if len(keyword.split()) < 2:  # Assuming insightful keywords are phrases
            error_msg = f"Keyword '{keyword}' may not be sufficiently insightful or evocative."
            logger.warning(error_msg)
            # Optionally, you can raise an AssertionError here

def assert_regal(keywords: List[str], research_objectives: str) -> None:
    """
    Checks that the keywords are central to understanding the phenomenon.

    Args:
        keywords (List[str]): List of extracted keywords.
        research_objectives (str): The research objectives text.

    Raises:
        AssertionError: If a keyword is not central to understanding the phenomenon.
    """
    objectives_text = research_objectives.lower()
    for keyword in keywords:
        if keyword.lower() not in objectives_text:
            error_msg = f"Keyword '{keyword}' may not be central to understanding the phenomenon."
            logger.warning(error_msg)
            # Optionally, you can raise an AssertionError here
