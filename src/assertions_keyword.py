import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def assert_keyword_realness(keywords: List[Dict[str, Any]]) -> None:
    """
    Assert that keywords reflect genuine experiences and perceptions of participants.
    
    Args:
        keywords: List of extracted keywords with their analysis
    
    Raises:
        AssertionError: If keywords don't demonstrate sufficient realness
    """
    for keyword in keywords:
        analysis = keyword.get("analysis_value", {})
        realness = analysis.get("realness", "")
        
        assert realness.strip(), f"Keyword '{keyword['keyword']}' lacks realness analysis"
        assert len(realness) >= 50, f"Realness analysis for '{keyword['keyword']}' is too brief"

def assert_keyword_richness(keywords: List[Dict[str, Any]]) -> None:
    """
    Assert that keywords provide rich, detailed understanding of the phenomenon.
    
    Args:
        keywords: List of extracted keywords with their analysis
    
    Raises:
        AssertionError: If keywords don't demonstrate sufficient richness
    """
    for keyword in keywords:
        analysis = keyword.get("analysis_value", {})
        richness = analysis.get("richness", "")
        
        assert richness.strip(), f"Keyword '{keyword['keyword']}' lacks richness analysis"
        assert len(richness) >= 50, f"Richness analysis for '{keyword['keyword']}' is too brief"

def assert_keyword_repetition(keywords: List[Dict[str, Any]]) -> None:
    """
    Assert that keywords show significant patterns of repetition in the data.
    
    Args:
        keywords: List of extracted keywords with their analysis
    
    Raises:
        AssertionError: If keywords don't demonstrate meaningful repetition
    """
    for keyword in keywords:
        analysis = keyword.get("analysis_value", {})
        repetition = analysis.get("repetition", "")
        
        assert repetition.strip(), f"Keyword '{keyword['keyword']}' lacks repetition analysis"
        assert "frequency" in repetition.lower(), f"Repetition analysis for '{keyword['keyword']}' doesn't discuss frequency"

def assert_keyword_rationale(keywords: List[Dict[str, Any]], theoretical_framework: Dict[str, str]) -> None:
    """
    Assert that keywords connect to the theoretical framework.
    
    Args:
        keywords: List of extracted keywords with their analysis
        theoretical_framework: The research's theoretical foundation
    
    Raises:
        AssertionError: If keywords don't demonstrate theoretical connection
    """
    theory = theoretical_framework.get("theory", "").lower()
    
    for keyword in keywords:
        analysis = keyword.get("analysis_value", {})
        rationale = analysis.get("rationale", "").lower()
        
        assert rationale.strip(), f"Keyword '{keyword['keyword']}' lacks theoretical rationale"
        assert any(term in rationale for term in theory.split()), f"Rationale for '{keyword['keyword']}' doesn't connect to theoretical framework"

def assert_keyword_repartee(keywords: List[Dict[str, Any]]) -> None:
    """
    Assert that keywords are insightful and stimulate further discussion.
    
    Args:
        keywords: List of extracted keywords with their analysis
    
    Raises:
        AssertionError: If keywords don't demonstrate insightful qualities
    """
    for keyword in keywords:
        analysis = keyword.get("analysis_value", {})
        repartee = analysis.get("repartee", "")
        
        assert repartee.strip(), f"Keyword '{keyword['keyword']}' lacks repartee analysis"
        assert "insight" in repartee.lower() or "discussion" in repartee.lower(), \
            f"Repartee analysis for '{keyword['keyword']}' doesn't demonstrate insight or discussion potential"

def assert_keyword_regal(keywords: List[Dict[str, Any]]) -> None:
    """
    Assert that keywords are central to understanding the phenomenon.
    
    Args:
        keywords: List of extracted keywords with their analysis
    
    Raises:
        AssertionError: If keywords don't demonstrate centrality to phenomenon
    """
    for keyword in keywords:
        analysis = keyword.get("analysis_value", {})
        regal = analysis.get("regal", "")
        
        assert regal.strip(), f"Keyword '{keyword['keyword']}' lacks regal analysis"
        assert "central" in regal.lower() or "core" in regal.lower() or "essential" in regal.lower(), \
            f"Regal analysis for '{keyword['keyword']}' doesn't demonstrate centrality to phenomenon"