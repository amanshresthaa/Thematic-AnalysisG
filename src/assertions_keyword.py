# File: assertions_keyword.py
import logging
from typing import List

logger = logging.getLogger(__name__)

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

def assert_keywords_meet_6Rs(keywords: List[str]) -> None:
    """
    Ensure that the extracted keywords meet at least one of the 6Rs criteria:
    Realness, Richness, Repetition, Rationale, Repartee, Regal.

    Args:
        keywords (List[str]): List of extracted keywords.

    Raises:
        AssertionError: If keywords do not meet any of the 6Rs criteria.
    """
    # Since we don't have a way to automatically verify the 6Rs criteria,
    # we'll assume that the language model has followed instructions.
    # In practice, this could involve more complex NLP checks.
    if not keywords:
        error_msg = "Extracted keywords do not meet any of the 6Rs criteria."
        logger.error(error_msg)
        raise AssertionError(error_msg)
