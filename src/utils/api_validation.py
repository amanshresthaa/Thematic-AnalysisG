"""
API Key Validation Utilities

This module provides functions to validate API keys for various services
and report issues before they cause runtime errors.
"""

import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def validate_api_keys() -> Dict[str, Any]:
    """
    Validate all API keys and report any issues.
    
    Returns:
        Dict with 'valid' (bool) and 'issues' (list of string messages)
    """
    issues = []
    all_valid = True
    
    # Check Cohere API key
    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key:
        issues.append("COHERE_API_KEY is not set. Cohere reranking will not be available.")
        all_valid = False
    else:
        try:
            import cohere
            client = cohere.Client(api_key=cohere_key)
            # Optional lightweight test call could be added here
        except Exception as e:
            issues.append(f"COHERE_API_KEY is invalid or expired: {str(e)}")
            all_valid = False
    
    # Check OpenAI API key (if used for embeddings)
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        issues.append("OPENAI_API_KEY is not set. Embedding generation may fail.")
        all_valid = False
    
    # Report issues
    if issues:
        logger.warning("API Key validation issues detected:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        logger.warning("The application will attempt to use alternative methods where possible.")
    else:
        logger.info("All API keys validated successfully.")
    
    return {
        "valid": all_valid,
        "issues": issues
    }
