# src/assertions.py
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def assert_relevant_quotations(quotations: List[Dict[str, Any]], themes: List[str]) -> None:
    """
    Ensure that each quotation is relevant to at least one identified theme.
    
    Args:
        quotations (List[Dict[str, Any]]): List of quotations with associated metadata.
        themes (List[str]): List of identified themes in the analysis.
    
    Raises:
        AssertionError: If a quotation does not align with any theme.
    """
    for quote in quotations:
        if not any(theme.lower() in [t.lower() for t in quote.get('themes', [])] for theme in themes):
            error_msg = f"Quotation '{quote.get('quote', '')[:50]}...' does not align with any identified theme."
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
                error_msg = f"Quotation '{quote.get('quote', '')[:50]}...' contains sensitive keyword '{keyword}'."
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
    participant_ids = {q.get("participant_id") for q in quotations}
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
        context = quote.get("context", "").strip()
        # Check if the quote exists in any of the transcript chunks
        in_transcript = any(quote_text in chunk for chunk in transcript_chunks)
        if not in_transcript:
            error_msg = f"Quotation '{quote_text[:50]}...' not found in any transcript chunk."
            logger.error(error_msg)
            raise AssertionError(error_msg)
        if not context:
            error_msg = f"Quotation '{quote_text[:50]}...' lacks contextual information."
            logger.error(error_msg)
            raise AssertionError(error_msg)

def assert_philosophical_alignment(quotations: List[Dict[str, Any]], theoretical_framework: str) -> None:
    """
    Ensure that quotations align with the researcher's philosophical stance.
    
    Args:
        quotations (List[Dict[str, Any]]): List of quotations with associated metadata.
        theoretical_framework (str): The theoretical and philosophical framework guiding the analysis.
    
    Raises:
        AssertionError: If any quotation does not align with the specified orientation.
    """
    framework_lower = theoretical_framework.lower()
    for quote in quotations:
        alignment_score = float(quote.get("alignment_score", 0.0))
        analysis_notes = quote.get("analysis_notes", "").lower()
        if "constructivist" in framework_lower:
            if alignment_score < 0.7 or "subjective" not in analysis_notes:
                error_msg = f"Quotation '{quote.get('quote', '')[:50]}...' does not align with constructivist orientation."
                logger.error(error_msg)
                raise AssertionError(error_msg)
        elif "critical realism" in framework_lower:
            if alignment_score < 0.7 or "objective" not in analysis_notes:
                error_msg = f"Quotation '{quote.get('quote', '')[:50]}...' does not align with critical realism orientation."
                logger.error(error_msg)
                raise AssertionError(error_msg)
        else:
            if alignment_score < 0.7:
                error_msg = f"Quotation '{quote.get('quote', '')[:50]}...' has insufficient alignment score."
                logger.error(error_msg)
                raise AssertionError(error_msg)
