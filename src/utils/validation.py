"""
Validation functions for thematic analysis pipeline.
"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def validate_relevance(quotations: List[Dict[str, Any]], context: str) -> bool:
    """
    Validates that each quotation is relevant to the given context.
    """
    try:
        for quote in quotations:
            if quote["quotation"] not in context:
                logger.debug(f"Quotation '{quote['QUOTE']}' not found in context.")
                return False
        return True
    except Exception as e:
        logger.error(f"Error in validate_relevance: {e}", exc_info=True)
        return False

def validate_quality(quotations: List[Dict[str, Any]]) -> bool:
    """
    Validates the quality and representation of each quotation.
    """
    try:
        for quote in quotations:
            if len(quote["quotation"].strip()) < 10:  # Example quality check
                logger.debug(f"Quotation '{quote['QUOTE']}' is too short to be considered high quality.")
                return False
        return True
    except Exception as e:
        logger.error(f"Error in validate_quality: {e}", exc_info=True)
        return False

def validate_context_clarity(quotations: List[str], context: str) -> bool:
    """
    Validates that each quotation is clear and has sufficient context.
    """
    try:
        for quote in quotations:
            if quote.lower() not in context.lower():
                logger.debug(f"Quotation '{quote}' lacks sufficient context.")
                return False
        return True
    except Exception as e:
        logger.error(f"Error in validate_context_clarity: {e}", exc_info=True)
        return False
