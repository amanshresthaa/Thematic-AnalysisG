# File: /Users/amankumarshrestha/Downloads/Thematic-AnalysisE/src/utils/validation_functions.py

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def validate_relevance(quotations: List[str], research_objectives: str) -> bool:
    """
    Validates that each quotation is relevant to the research objectives.
    """
    try:
        for quote in quotations:
            # Simple keyword matching; can be enhanced with NLP techniques
            if not any(obj.lower() in quote.lower() for obj in research_objectives.split()):
                logger.debug(f"Quotation '{quote}' is not relevant to the research objectives.")
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
            # Additional quality checks can be added here
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
