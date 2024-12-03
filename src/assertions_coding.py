# src/assertions_coding.py

import logging
from typing import List, Dict, Any, Set

logger = logging.getLogger(__name__)

def assert_robustness(codes: List[Dict[str, Any]]) -> None:
    """
    Ensure coding accurately and comprehensively captures the essence of the data.

    Raises:
        AssertionError: If any code does not comprehensively represent the data.
    """
    for code in codes:
        data_extracts = code.get("data_extracts", [])
        if not data_extracts:
            error_msg = f"Code '{code.get('code_name', '')}' has no data extracts."
            logger.error(error_msg)
            raise AssertionError(error_msg)
        
        # Example metric: keyword diversity (assuming each extract has 'keywords')
        keyword_set: Set[str] = set()
        for extract in data_extracts:
            keywords = extract.get("keywords", [])
            keyword_set.update([kw.lower() for kw in keywords])
        
        if len(keyword_set) < 2:
            error_msg = f"Code '{code.get('code_name', '')}' lacks keyword diversity."
            logger.error(error_msg)
            raise AssertionError(error_msg)
        
        logger.debug(f"Code '{code.get('code_name', '')}' passes robustness check.")

def assert_reflectiveness(codes: List[Dict[str, Any]], theoretical_framework: Dict[str, str]) -> None:
    """
    Ensure codes reflect the relationship between data and the theoretical framework.

    Raises:
        AssertionError: If any code does not align with the theoretical framework.
    """
    framework_concepts = set(k.lower() for k in theoretical_framework.keys())
    
    for code in codes:
        data_extracts = code.get("data_extracts", [])
        aligned = False
        for extract in data_extracts:
            keywords = extract.get("keywords", [])
            if any(kw.lower() in framework_concepts for kw in keywords):
                aligned = True
                break
        if not aligned:
            error_msg = f"Code '{code.get('code_name', '')}' does not align with the theoretical framework."
            logger.error(error_msg)
            raise AssertionError(error_msg)
        
        logger.debug(f"Code '{code.get('code_name', '')}' passes reflectiveness check.")

def assert_resplendence(codes: List[Dict[str, Any]]) -> None:
    """
    Ensure codes offer rich and comprehensive explanations of the context.

    Raises:
        AssertionError: If any code lacks comprehensive contextual explanations.
    """
    for code in codes:
        analysis_notes = code.get("analysis_notes", "")
        if not analysis_notes or len(analysis_notes.split()) < 50:
            error_msg = f"Code '{code.get('code_name', '')}' lacks comprehensive analysis notes."
            logger.error(error_msg)
            raise AssertionError(error_msg)
        
        logger.debug(f"Code '{code.get('code_name', '')}' passes resplendence check.")

def assert_relevance(codes: List[Dict[str, Any]], research_objectives: str) -> None:
    """
    Ensure codes accurately represent the data and align with research objectives.

    Raises:
        AssertionError: If any code does not align with research objectives.
    """
    objectives = [obj.strip().lower() for obj in research_objectives.split('.') if obj.strip()]
    if not objectives:
        error_msg = "No research objectives provided for relevance check."
        logger.error(error_msg)
        raise AssertionError(error_msg)
    
    for code in codes:
        code_name = code.get("code_name", "")
        aligns = False
        data_extracts = code.get("data_extracts", [])
        for extract in data_extracts:
            relevance = extract.get("relevance", "")
            if any(obj in relevance.lower() for obj in objectives):
                aligns = True
                break
        if not aligns:
            error_msg = f"Code '{code_name}' does not align with any research objectives."
            logger.error(error_msg)
            raise AssertionError(error_msg)
        
        logger.debug(f"Code '{code_name}' passes relevance check.")

def assert_radicality(codes: List[Dict[str, Any]]) -> None:
    """
    Ensure codes provide unique insights and may challenge dominant narratives.

    Raises:
        AssertionError: If any code lacks uniqueness or does not offer fresh perspectives.
    """
    seen_codes = set()
    for code in codes:
        code_name = code.get("code_name", "").lower()
        if code_name in seen_codes:
            error_msg = f"Code '{code.get('code_name', '')}' is not unique."
            logger.error(error_msg)
            raise AssertionError(error_msg)
        seen_codes.add(code_name)
        
        data_extracts = code.get("data_extracts", [])
        unique_insights = set()
        for extract in data_extracts:
            insight = extract.get("insight", "").lower()
            if insight:
                unique_insights.add(insight)
        
        if len(unique_insights) < 1:
            error_msg = f"Code '{code.get('code_name', '')}' does not offer unique insights."
            logger.error(error_msg)
            raise AssertionError(error_msg)
        
        logger.debug(f"Code '{code.get('code_name', '')}' passes radicality check.")

def assert_righteousness(codes: List[Dict[str, Any]]) -> None:
    """
    Ensure codes fit logically within the coding framework and avoid overlaps.

    Raises:
        AssertionError: If any code overlaps conceptually with others or lacks logical consistency.
    """
    code_definitions = {code.get("code_name", "").lower(): code for code in codes}
    
    for code_name, code in code_definitions.items():
        related_codes = code.get("related_codes", [])
        for related_code in related_codes:
            related_code_lower = related_code.lower()
            if related_code_lower not in code_definitions:
                error_msg = f"Related code '{related_code}' for code '{code_name}' does not exist."
                logger.error(error_msg)
                raise AssertionError(error_msg)
            
            # Example overlap check: shared keywords
            code_keywords = set(kw.lower() for extract in code.get("data_extracts", []) for kw in extract.get("keywords", []))
            related_keywords = set(kw.lower() for extract in code_definitions[related_code_lower].get("data_extracts", []) for kw in extract.get("keywords", []))
            overlap = code_keywords.intersection(related_keywords)
            if len(overlap) > 5:  # Threshold for overlap
                error_msg = f"Code '{code_name}' overlaps significantly with related code '{related_code}'."
                logger.error(error_msg)
                raise AssertionError(error_msg)
        
        logger.debug(f"Code '{code_name}' passes righteousness check.")

def assert_code_representation(codes: List[Dict[str, Any]]) -> None:
    """
    Ensure each code accurately encapsulates underlying concepts and meanings.

    Raises:
        AssertionError: If any code does not consistently reflect its definition.
    """
    for code in codes:
        code_name = code.get("code_name", "")
        definition = code.get("definition", "").lower()
        data_extracts = code.get("data_extracts", [])
        
        for extract in data_extracts:
            quotation = extract.get("quotation", "").lower()
            if definition not in quotation:
                error_msg = f"Data extract in code '{code_name}' does not reflect the code's definition."
                logger.error(error_msg)
                raise AssertionError(error_msg)
        
        logger.debug(f"Code '{code_name}' passes code representation check.")

def assert_code_specificity(codes: List[Dict[str, Any]]) -> None:
    """
    Ensure codes are specific enough to convey clear meanings within the theoretical framework.

    Raises:
        AssertionError: If any code is too vague or broad.
    """
    for code in codes:
        code_name = code.get("code_name", "")
        if len(code_name.split()) < 2:
            error_msg = f"Code name '{code_name}' is too broad or vague."
            logger.error(error_msg)
            raise AssertionError(error_msg)
        
        data_extracts = code.get("data_extracts", [])
        for extract in data_extracts:
            keywords = extract.get("keywords", [])
            for kw in keywords:
                if len(kw.split()) < 2:
                    error_msg = f"Keyword '{kw}' in code '{code_name}' is too broad or vague."
                    logger.error(error_msg)
                    raise AssertionError(error_msg)
        
        logger.debug(f"Code '{code_name}' passes specificity check.")

def assert_code_relevance(codes: List[Dict[str, Any]], research_objectives: str) -> None:
    """
    Ensure each code aligns with research objectives and contributes to answering research questions.

    Raises:
        AssertionError: If any code does not contribute to the research objectives.
    """
    objectives = [obj.strip().lower() for obj in research_objectives.split('.') if obj.strip()]
    if not objectives:
        error_msg = "No research objectives provided for code relevance check."
        logger.error(error_msg)
        raise AssertionError(error_msg)
    
    for code in codes:
        code_name = code.get("code_name", "")
        aligns = False
        data_extracts = code.get("data_extracts", [])
        for extract in data_extracts:
            relevance = extract.get("relevance", "")
            if any(obj in relevance.lower() for obj in objectives):
                aligns = True
                break
        if not aligns:
            error_msg = f"Code '{code_name}' does not contribute to any research objectives."
            logger.error(error_msg)
            raise AssertionError(error_msg)
        
        logger.debug(f"Code '{code_name}' passes code relevance check.")

def assert_code_distinctiveness(codes: List[Dict[str, Any]]) -> None:
    """
    Ensure codes are distinct and do not overlap or duplicate each other.

    Raises:
        AssertionError: If any codes are too similar or duplicated.
    """
    seen_definitions = {}
    for code in codes:
        code_name = code.get("code_name", "").lower()
        definition = code.get("definition", "").lower()
        if definition in seen_definitions:
            error_msg = f"Code '{code_name}' has a duplicate definition with code '{seen_definitions[definition]}'."
            logger.error(error_msg)
            raise AssertionError(error_msg)
        seen_definitions[definition] = code_name
        logger.debug(f"Code '{code_name}' is distinct.")

def run_all_coding_assertions(
    codes: List[Dict[str, Any]],
    research_objectives: str,
    theoretical_framework: Dict[str, str]
) -> None:
    """
    Run all coding assertions to validate the coding framework.

    Args:
        codes (List[Dict[str, Any]]): List of codes with metadata.
        research_objectives (str): Research objectives guiding the analysis.
        theoretical_framework (Dict[str, str]): Theoretical framework for context.

    Raises:
        AssertionError: If any of the assertions fail.
    """
    assert_robustness(codes)
    assert_reflectiveness(codes, theoretical_framework)
    assert_resplendence(codes)
    assert_relevance(codes, research_objectives)
    assert_radicality(codes)
    assert_righteousness(codes)
    assert_code_representation(codes)
    assert_code_specificity(codes)
    assert_code_relevance(codes, research_objectives)
    assert_code_distinctiveness(codes)
    logger.info("All coding assertions passed successfully.")
