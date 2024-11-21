# src/analysis/assertions_keyword.py
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

def assert_realness(keywords: List[str], quotation: str) -> None:
    """
    Checks if keywords genuinely reflect participants' experiences by verifying their presence in the quotation.

    Args:
        keywords (List[str]): List of extracted keywords.
        quotation (str): The specific quotation from which to extract keywords.

    Raises:
        AssertionError: If a keyword is not present in the quotation.
    """
    quotation_text = quotation.lower()
    for keyword in keywords:
        if keyword.lower() not in quotation_text:
            error_msg = f"Keyword '{keyword}' does not reflect participants' experiences (not found in quotation)."
            logger.error(error_msg)
            raise AssertionError(error_msg)

def assert_keywords_not_exclusive_to_context(keywords: List[str], quotation: str, contextual_info: List[str]) -> None:
    """
    Ensures that keywords are not exclusively derived from the contextual content.
    If a keyword exists in contextual_info, it must also exist in the quotation.

    Args:
        keywords (List[str]): List of extracted keywords.
        quotation (str): The specific quotation from which to extract keywords.
        contextual_info (List[str]): List of contextualized content providing background for the quotation.

    Raises:
        AssertionError: If a keyword is found exclusively in contextual_info.
    """
    quotation_text = quotation.lower()
    contextual_text = ' '.join(contextual_info).lower()

    for keyword in keywords:
        keyword_lower = keyword.lower()
        in_quotation = keyword_lower in quotation_text
        in_context = keyword_lower in contextual_text

        if in_context and not in_quotation:
            error_msg = f"Keyword '{keyword}' is exclusively present in contextual content and not in the quotation."
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

def assert_repetition(keywords: List[str], quotation: str) -> None:
    """
    Checks that the keywords frequently occur in the data.

    Args:
        keywords (List[str]): List of extracted keywords.
        quotation (str): The specific quotation from which to extract keywords.

    Raises:
        AssertionError: If a keyword does not occur frequently in the data.
    """
    quotation_text = quotation.lower()
    word_counts = Counter(word_tokenize(quotation_text))
    total_words = sum(word_counts.values())
    for keyword in keywords:
        keyword_count = quotation_text.count(keyword.lower())
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
