import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def assert_relevant_quotations(quotations: List[Dict[str, Any]], research_objectives: str) -> None:
    """
    Ensure that each quotation is relevant to the research objectives.

    Args:
        quotations (List[Dict[str, Any]]): List of quotations with associated metadata.
        research_objectives (str): The research objectives guiding the analysis.

    Raises:
        AssertionError: If a quotation does not align with the research objectives.
    """
    for quote in quotations:
        if research_objectives.lower() not in quote.get('analysis_value', {}).get('relevance', '').lower():
            error_msg = f"Quotation '{quote.get('quotation', '')[:50]}...' does not align with research objectives."
            logger.error(error_msg)
            raise AssertionError(error_msg)

def assert_confidentiality(quotations: List[Dict[str, Any]], sensitive_keywords: List[str]) -> None:
    """
    Ensure that no quotations contain sensitive or identifiable information.

    Args:
        quotations (List[Dict[str, Any]]): List of quotations with associated metadata.
        sensitive_keywords (List[str]): List of keywords that should not appear in quotations.

    Raises:
        AssertionError: If a quotation contains sensitive information.
    """
    for quote in quotations:
        quote_text = quote.get("quotation", "").lower()
        for keyword in sensitive_keywords:
            if keyword.lower() in quote_text:
                error_msg = f"Quotation '{quote.get('quotation', '')[:50]}...' contains sensitive keyword '{keyword}'."
                logger.error(error_msg)
                raise AssertionError(error_msg)

def assert_diversity_of_quotations(quotations: List[Dict[str, Any]], min_participants: int = 3) -> None:
    """
    Ensure that quotations represent a diverse set of participants.

    Args:
        quotations (List[Dict[str, Any]]): List of quotations with associated metadata.
        min_participants (int): Minimum number of different participants required.

    Raises:
        AssertionError: If quotations do not represent the required diversity.
    """
    participant_ids = {q.get("participant_id") for q in quotations if q.get("participant_id")}
    if len(participant_ids) < min_participants:
        error_msg = f"Only {len(participant_ids)} unique participants are represented in quotations; minimum required is {min_participants}."
        logger.error(error_msg)
        raise AssertionError(error_msg)

def assert_contextual_adequacy(quotations: List[Dict[str, Any]], transcript_chunks: List[str]) -> None:
    """
    Ensure that each quotation has adequate context.

    Args:
        quotations (List[Dict[str, Any]]): List of quotations with associated metadata.
        transcript_chunks (List[str]): List of transcript chunks.

    Raises:
        AssertionError: If a quotation lacks necessary context or is not found in transcript.
    """
    for quote in quotations:
        quote_text = quote.get("quotation", "")
        # Check if the quote exists in any of the transcript chunks
        in_transcript = any(quote_text in chunk for chunk in transcript_chunks)
        if not in_transcript:
            error_msg = f"Quotation '{quote_text[:50]}...' not found in any transcript chunk."
            logger.error(error_msg)
            raise AssertionError(error_msg)
        context = quote.get("context", {}).get("situation", "").strip()
        if not context:
            error_msg = f"Quotation '{quote_text[:50]}...' lacks contextual information."
            logger.error(error_msg)
            raise AssertionError(error_msg)

def assert_philosophical_alignment(quotations: List[Dict[str, Any]], theoretical_framework: Dict[str, str]) -> None:
    """
    Ensure that quotations align with the researcher's philosophical stance.

    Args:
        quotations (List[Dict[str, Any]]): List of quotations with associated metadata.
        theoretical_framework (Dict[str, str]): The theoretical and philosophical framework guiding the analysis.

    Raises:
        AssertionError: If any quotation does not align with the specified orientation.
    """
    philosophical_approach = theoretical_framework.get("philosophical_approach", "").lower()
    for quote in quotations:
        theoretical_alignment = quote.get("analysis_value", {}).get("theoretical_alignment", "").lower()
        if philosophical_approach not in theoretical_alignment:
            error_msg = f"Quotation '{quote.get('quotation', '')[:50]}...' does not align with the philosophical approach '{philosophical_approach}'."
            logger.error(error_msg)
            raise AssertionError(error_msg)

def assert_patterns_identified(patterns_identified: List[str]) -> None:
    """
    Ensure that patterns have been identified and are valid.

    Args:
        patterns_identified (List[str]): List of identified patterns.

    Raises:
        AssertionError: If no patterns are identified or if any pattern is invalid.
    """
    if not patterns_identified:
        error_msg = "No patterns were identified in the analysis."
        logger.error(error_msg)
        raise AssertionError(error_msg)
    for pattern in patterns_identified:
        if not isinstance(pattern, str) or not pattern.strip():
            error_msg = f"Invalid pattern identified: '{pattern}'."
            logger.error(error_msg)
            raise AssertionError(error_msg)

def assert_theoretical_interpretation(theoretical_interpretation: str) -> None:
    """
    Ensure that the theoretical interpretation is adequate.

    Args:
        theoretical_interpretation (str): The theoretical interpretation text.

    Raises:
        AssertionError: If the interpretation is empty or insufficient.
    """
    if not theoretical_interpretation.strip():
        error_msg = "Theoretical interpretation is empty."
        logger.error(error_msg)
        raise AssertionError(error_msg)
    if len(theoretical_interpretation.strip().split()) < 20:
        error_msg = "Theoretical interpretation is too brief."
        logger.error(error_msg)
        raise AssertionError(error_msg)

def assert_research_alignment(research_alignment: str) -> None:
    """
    Ensure that the research alignment explanation is adequate.

    Args:
        research_alignment (str): The research alignment text.

    Raises:
        AssertionError: If the alignment explanation is empty or insufficient.
    """
    if not research_alignment.strip():
        error_msg = "Research alignment explanation is empty."
        logger.error(error_msg)
        raise AssertionError(error_msg)
    if len(research_alignment.strip().split()) < 20:
        error_msg = "Research alignment explanation is too brief."
        logger.error(error_msg)
        raise AssertionError(error_msg)
