# src/assertions_keyword.py

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def assert_keywords_not_exclusive_to_quotation(
    keywords: List[Dict[str, Any]], 
    quotation: str, 
    contextualized_contents: List[str]
) -> None:
    """
    Assert that keywords capture concepts beyond verbatim quotes.
    
    According to the guideline: "Keywords should represent broader concepts, not just exact phrases from quotations."

    Args:
        keywords (List[Dict[str, Any]]): List of extracted keywords with their analysis.
        quotation (str): The original quotation.
        contextualized_contents (List[str]): Additional contextual content for analysis.
    
    Raises:
        AssertionError: If any keyword is too closely tied to verbatim quotes.
    """
    for keyword in keywords:
        keyword_text = keyword.get("keyword", "").strip().lower()
        if not keyword_text:
            error_msg = "Keyword is empty or missing."
            logger.error(error_msg)
            raise AssertionError(error_msg)
        
        if keyword_text in quotation.lower():
            # Check if the keyword is supported by contextualized contents
            if not any(keyword_text in context.lower() for context in contextualized_contents):
                error_msg = f"Keyword '{keyword_text}' appears to be too closely tied to the quotation."
                logger.error(error_msg)
                raise AssertionError(error_msg)
            else:
                logger.debug(f"Keyword '{keyword_text}' is adequately supported by contextual content.")

def assert_keyword_specificity(
    keywords: List[Dict[str, Any]], 
    theoretical_framework: Dict[str, str]
) -> None:
    """
    Assert that keywords are sufficiently specific and meaningful.
    
    According to the guideline: "Keywords should be specific enough to convey meaningful concepts within the theoretical framework."

    Args:
        keywords (List[Dict[str, Any]]): List of extracted keywords with their analysis.
        theoretical_framework (Dict[str, str]): Theoretical framework for context.
    
    Raises:
        AssertionError: If any keyword is too broad or vague.
    """
    for keyword in keywords:
        keyword_text = keyword.get("keyword", "").strip()
        if not keyword_text:
            error_msg = "Keyword is empty or missing."
            logger.error(error_msg)
            raise AssertionError(error_msg)
        
        # Example check: keyword length (can be customized based on requirements)
        if len(keyword_text.split()) < 2:
            error_msg = f"Keyword '{keyword_text}' is too broad or vague."
            logger.error(error_msg)
            raise AssertionError(error_msg)
        
        logger.debug(f"Keyword '{keyword_text}' meets specificity requirements.")

def assert_keyword_relevance(
    keywords: List[Dict[str, Any]], 
    research_objectives: str,
    contextualized_contents: List[str]
) -> None:
    """
    Assert that keywords are relevant to the research objectives.
    
    According to the guideline: "Keywords should align closely with the research objectives to ensure focused analysis."

    Args:
        keywords (List[Dict[str, Any]]): List of extracted keywords with their analysis.
        research_objectives (str): The research objectives.
        contextualized_contents (List[str]): Additional contextual content for analysis.
    
    Raises:
        AssertionError: If any keyword is not sufficiently relevant to the research objectives.
    """
    objectives = [obj.strip().lower() for obj in research_objectives.split('.') if obj.strip()]
    if not objectives:
        error_msg = "No research objectives provided for relevance check."
        logger.error(error_msg)
        raise AssertionError(error_msg)
    
    for keyword in keywords:
        keyword_text = keyword.get("keyword", "").strip().lower()
        if not keyword_text:
            error_msg = "Keyword is empty or missing."
            logger.error(error_msg)
            raise AssertionError(error_msg)
        
        # Check if keyword aligns with any research objective
        if not any(obj in keyword_text for obj in objectives):
            # Additionally, check if keyword appears in contextualized contents
            if not any(keyword_text in context.lower() for context in contextualized_contents):
                error_msg = f"Keyword '{keyword_text}' is not sufficiently relevant to the research objectives."
                logger.error(error_msg)
                raise AssertionError(error_msg)
        
        logger.debug(f"Keyword '{keyword_text}' is relevant to the research objectives.")

def assert_keyword_distinctiveness(
    keywords: List[Dict[str, Any]]
) -> None:
    """
    Assert that keywords are sufficiently distinct from one another.
    
    According to the guideline: "Keywords should be unique and not redundant to ensure comprehensive coverage of concepts."

    Args:
        keywords (List[Dict[str, Any]]): List of extracted keywords with their analysis.
    
    Raises:
        AssertionError: If any keywords are too similar or duplicated.
    """
    seen_keywords = set()
    for keyword in keywords:
        keyword_text = keyword.get("keyword", "").strip().lower()
        if not keyword_text:
            error_msg = "Keyword is empty or missing."
            logger.error(error_msg)
            raise AssertionError(error_msg)
        
        if keyword_text in seen_keywords:
            error_msg = f"Keyword '{keyword_text}' is duplicated and not distinct."
            logger.error(error_msg)
            raise AssertionError(error_msg)
        
        seen_keywords.add(keyword_text)
        logger.debug(f"Keyword '{keyword_text}' is distinct.")

