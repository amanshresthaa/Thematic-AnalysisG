
#src/assertions_quotation.py
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def assert_pattern_representation(quotations: List[Dict[str, Any]], patterns: List[str]) -> None:
    """
    Ensure quotations represent robust patterns in the data.
    According to the paper: "Quotations should symbolize robust patterns within the data"
    and "Select quotes that demonstrate robust patterns in the data."

    Args:
        quotations (List[Dict[str, Any]]): List of quotations with metadata
        patterns (List[str]): Identified patterns in the data

    Raises:
        AssertionError: If quotations don't demonstrate robust patterns
    """
    if not patterns:
        raise AssertionError("No patterns provided for analysis")

    pattern_support = {pattern: [] for pattern in patterns}

    for quote in quotations:
        pattern_representation = quote.get("context", {}).get("pattern_representation", "")
        for pattern in patterns:
            if pattern.lower() in pattern_representation.lower():
                pattern_support[pattern].append(quote["quotation"])

    # Each pattern should be supported by multiple quotations for robustness
    for pattern, supporting_quotes in pattern_support.items():
        if len(supporting_quotes) < 2:
            error_msg = f"Pattern '{pattern}' is not robustly supported by multiple quotations"
            logger.error(error_msg)
            raise AssertionError(error_msg)

def assert_research_objective_alignment(quotations: List[Dict[str, Any]], research_objectives: str) -> None:
    """
    Ensure quotations align with research objectives.
    According to the paper: "The evaluation objectives provide a focus or domain of relevance
    for conducting the analysis"

    Args:
        quotations (List[Dict[str, Any]]): List of quotations with metadata
        research_objectives (str): Research objectives guiding the analysis

    Raises:
        AssertionError: If quotations don't align with objectives
    """
    for quote in quotations:
        relevance = quote.get('analysis_value', {}).get('relevance', '')
        if not relevance or not any(obj.lower() in relevance.lower() for obj in research_objectives.split('.')):
            error_msg = f"Quotation does not align with research objectives: '{quote.get('quotation', '')[:50]}...'"
            logger.error(error_msg)
            raise AssertionError(error_msg)

def assert_selective_transcription(quotations: List[Dict[str, Any]], transcript: str) -> None:
    """
    Ensure quotations are selectively chosen for relevance.
    According to the paper: "A more useful transcript is a more selective one" and
    "selecting parts relevant to the evaluation objectives"

    Args:
        quotations (List[Dict[str, Any]]): List of quotations with metadata
        transcript (str): Original transcript text

    Raises:
        AssertionError: If quotation selection isn't properly selective
    """
    total_words = len(transcript.split())
    quoted_words = sum(len(quote.get('quotation', '').split()) for quote in quotations)

    # Check if quotations are too verbose (should be selective)
    if quoted_words > total_words * 0.3:  # Maximum 30% of original text
        error_msg = "Quotation selection is not selective enough"
        logger.error(error_msg)
        raise AssertionError(error_msg)

def assert_creswell_categorization(quotations: List[Dict[str, Any]]) -> None:
    """
    Verify proper use of Creswell's quotation categories.
    According to the paper: "Creswell (2012) classified quotations into three types:
    discrete, embedded, and longer quotations"

    Args:
        quotations (List[Dict[str, Any]]): List of quotations with metadata

    Raises:
        AssertionError: If quotations don't follow Creswell's guidelines
    """
    categories = {'longer': 0, 'discrete': 0, 'embedded': 0}

    for quote in quotations:
        category = quote.get("creswell_category", "").lower()
        if category not in categories:
            error_msg = f"Invalid Creswell category '{category}'"
            logger.error(error_msg)
            raise AssertionError(error_msg)

        quote_length = len(quote.get("quotation", "").split())

        # Length guidelines based on Creswell's classifications
        if category == "longer" and quote_length < 40:
            error_msg = f"'Longer' quotation is too short ({quote_length} words)"
            logger.error(error_msg)
            raise AssertionError(error_msg)
        elif category == "discrete" and quote_length > 30:
            error_msg = f"'Discrete' quotation is too long ({quote_length} words)"
            logger.error(error_msg)
            raise AssertionError(error_msg)
        elif category == "embedded" and quote_length > 10:
            error_msg = f"'Embedded' quotation is too long ({quote_length} words)"
            logger.error(error_msg)
            raise AssertionError(error_msg)

        categories[category] += 1

def assert_reader_engagement(quotations: List[Dict[str, Any]]) -> None:
    """
    Ensure quotations enhance reader engagement.
    According to the paper: "Quotations can enhance the readers' engagement with the text"
    and should not be chosen merely to "incite controversy"

    Args:
        quotations (List[Dict[str, Any]]): List of quotations with metadata

    Raises:
        AssertionError: If quotations don't promote proper engagement
    """
    for quote in quotations:
        # Check for essential engagement elements
        has_context = bool(quote.get("context", {}).get("situation"))
        has_interpretation = bool(quote.get("analysis_value", {}).get("relevance"))
        has_pattern = bool(quote.get("context", {}).get("pattern_representation"))

        if not all([has_context, has_interpretation, has_pattern]):
            error_msg = "Quotation lacks essential engagement elements (context, interpretation, or pattern connection)"
            logger.error(error_msg)
            raise AssertionError(error_msg)

        # Check against controversial selection without substance
        quote_text = quote.get("quotation", "")
        if "!" in quote_text and not has_interpretation:
            error_msg = "Quotation appears selected for controversy without substantive contribution"
            logger.error(error_msg)
            raise AssertionError(error_msg)
