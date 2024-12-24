import logging
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import dspy
from functools import lru_cache

logger = logging.getLogger(__name__)

@dataclass
class AssertionThresholds:
    """Configuration thresholds for keyword assertions."""
    min_analysis_words: int = 10
    min_keywords_per_r: int = 2
    min_theoretical_alignment: float = 0.7
    min_repetition_count: int = 2
    min_framework_aspects: int = 2
    min_total_keywords: int = 3

@dataclass
class AssertionConfig:
    """Complete configuration for keyword assertions."""
    enabled_assertions: List[str] = None
    required_passes: int = 3
    thresholds: AssertionThresholds = None
    strict_mode: bool = False
    detailed_logging: bool = True
    
    def __post_init__(self):
        if self.enabled_assertions is None:
            self.enabled_assertions = ["all"]
        if self.thresholds is None:
            self.thresholds = AssertionThresholds()

class KeywordAssertions:
    """Enhanced DSPy assertions for granular keyword validation in thematic analysis."""
    
    def __init__(self, config: Optional[AssertionConfig] = None):
        """Initialize with optional configuration."""
        self.config = config or AssertionConfig()
        logger.info("Initialized KeywordAssertions with config: %s", self.config)
    
    @lru_cache(maxsize=128)
    def _tokenize_text(self, text: str) -> Set[str]:
        """Tokenize text into words, with caching for performance."""
        return set(text.lower().strip().split())
    
    def _validate_realness_keyword(self, keyword: Dict[str, Any]) -> bool:
        """Validate a single keyword's realness analysis."""
        kw_text = keyword.get("keyword", "")
        realness_value = keyword.get("analysis_value", {}).get("realness", "").strip()
        
        if not realness_value:
            logger.error(f"Keyword '{kw_text}' lacks Realness analysis")
            return False
        
        word_count = len(realness_value.split())
        if word_count < self.config.thresholds.min_analysis_words:
            logger.warning(
                f"Realness analysis for '{kw_text}' has only {word_count} words "
                f"(min: {self.config.thresholds.min_analysis_words})"
            )
            return False
        
        return True
    
    def assert_realness_analysis(self, keywords: List[Dict[str, Any]], quotation: str) -> bool:
        """Validates the Realness dimension analysis for each keyword."""
        logger.debug("Starting Realness Analysis assertion")
        realness_keywords = [k for k in keywords if "Realness" in k.get("6Rs_framework", [])]
        
        if len(realness_keywords) < self.config.thresholds.min_keywords_per_r:
            logger.warning(
                f"Found only {len(realness_keywords)} Realness keywords, "
                f"minimum required: {self.config.thresholds.min_keywords_per_r}"
            )
            return False
        
        for keyword in realness_keywords:
            if not self._validate_realness_keyword(keyword):
                return False
        
        logger.debug("Completed Realness Analysis assertion")
        return True
    
    def _calculate_theoretical_alignment(self, analysis_text: str, theoretical_framework: Dict[str, str]) -> float:
        """Calculate theoretical alignment score."""
        theory = theoretical_framework.get("theory", "").lower()
        philosophical_approach = theoretical_framework.get("philosophical_approach", "").lower()
        
        theory_words = self._tokenize_text(theory + " " + philosophical_approach)
        analysis_words = self._tokenize_text(analysis_text)
        
        if not theory_words:
            return 0.0
            
        return len(theory_words.intersection(analysis_words)) / len(theory_words)
    
    def assert_richness_analysis(self, keywords: List[Dict[str, Any]], 
                               theoretical_framework: Dict[str, str]) -> bool:
        """Validates the Richness dimension analysis with theoretical alignment."""
        logger.debug("Starting Richness Analysis assertion")
        richness_keywords = [k for k in keywords if "Richness" in k.get("6Rs_framework", [])]
        
        if len(richness_keywords) < self.config.thresholds.min_keywords_per_r:
            logger.warning(
                f"Found only {len(richness_keywords)} Richness keywords, "
                f"minimum required: {self.config.thresholds.min_keywords_per_r}"
            )
            return False
        
        for keyword in richness_keywords:
            kw_text = keyword.get("keyword", "")
            richness_value = keyword.get("analysis_value", {}).get("richness", "").strip()
            
            if not richness_value:
                logger.error(f"Keyword '{kw_text}' lacks Richness analysis")
                return False
            
            alignment_score = self._calculate_theoretical_alignment(richness_value, theoretical_framework)
            
            if alignment_score < self.config.thresholds.min_theoretical_alignment:
                logger.warning(
                    f"Low theoretical alignment (score: {alignment_score:.2f}) for '{kw_text}'"
                )
                return False
        
        logger.debug("Completed Richness Analysis assertion")
        return True
    
    def _count_keyword_occurrences(self, keyword: str, contents: List[str]) -> int:
        """Count occurrences of a keyword in contextualized contents."""
        return sum(keyword.lower() in content.lower() for content in contents)
    
    def assert_repetition_analysis(self, keywords: List[Dict[str, Any]], 
                                 contextualized_contents: List[str]) -> bool:
        """Validates the Repetition dimension with contextual evidence."""
        logger.debug("Starting Repetition Analysis assertion")
        
        repetition_keywords = [k for k in keywords if "Repetition" in k.get("6Rs_framework", [])]
        
        if len(repetition_keywords) < self.config.thresholds.min_keywords_per_r:
            logger.warning(
                f"Found only {len(repetition_keywords)} Repetition keywords, "
                f"minimum required: {self.config.thresholds.min_keywords_per_r}"
            )
            return False
        
        for keyword in repetition_keywords:
            kw_text = keyword.get("keyword", "").lower()
            repetition_value = keyword.get("analysis_value", {}).get("repetition", "").strip()
            
            if not repetition_value:
                logger.error(f"Keyword '{kw_text}' lacks Repetition analysis")
                return False
            
            occurrences = self._count_keyword_occurrences(kw_text, contextualized_contents)
            
            if occurrences < self.config.thresholds.min_repetition_count:
                logger.warning(
                    f"Keyword '{kw_text}' appears only {occurrences} times "
                    f"(min: {self.config.thresholds.min_repetition_count})"
                )
                return False
        
        logger.debug("Completed Repetition Analysis assertion")
        return True
    
    def _check_framework_coverage(self, keywords: List[Dict[str, Any]]) -> bool:
        """Check if keywords have sufficient framework aspect coverage."""
        aspects_coverage = [len(kw.get("6Rs_framework", [])) for kw in keywords]
        return all(ac >= self.config.thresholds.min_framework_aspects for ac in aspects_coverage)

def validate_keywords_dspy(
    keywords: List[Dict[str, Any]], 
    quotation: str,
    contextualized_contents: List[str], 
    research_objectives: str,
    theoretical_framework: Dict[str, str],
    config: Optional[AssertionConfig] = None
) -> Dict[str, Any]:
    """
    Comprehensive keyword validation with configurable assertions and detailed reporting.
    
    Args:
        keywords: List of keyword dictionaries to validate
        quotation: Original quotation text
        contextualized_contents: List of contextual content strings
        research_objectives: Research objectives text
        theoretical_framework: Dictionary containing theoretical framework details
        config: Optional assertion configuration
    
    Returns:
        Dict containing validation results and any suggestions/warnings
    """
    logger.info("Starting keyword validation with DSPy assertions")
    config = config or AssertionConfig()
    assertions = KeywordAssertions(config)
    
    validation_report = {
        "passed": True,
        "assertions_run": [],
        "failed_assertions": [],
        "suggestions": [],
        "warnings": []
    }
    
    try:
        passes = 0
        assertion_methods = {
            "realness": lambda: assertions.assert_realness_analysis(keywords, quotation),
            "richness": lambda: assertions.assert_richness_analysis(keywords, theoretical_framework),
            "repetition": lambda: assertions.assert_repetition_analysis(keywords, contextualized_contents)
        }
        
        # Run enabled assertions
        enabled = config.enabled_assertions
        for assertion_name, assertion_method in assertion_methods.items():
            if "all" in enabled or assertion_name in enabled:
                try:
                    logger.debug(f"Running {assertion_name} assertion")
                    if assertion_method():
                        validation_report["assertions_run"].append(assertion_name)
                        passes += 1
                    else:
                        validation_report["failed_assertions"].append({
                            "type": assertion_name,
                            "error": f"{assertion_name} assertion failed"
                        })
                        validation_report["passed"] = False
                except Exception as e:
                    logger.error(f"{assertion_name} assertion failed with error: {str(e)}")
                    validation_report["failed_assertions"].append({
                        "type": assertion_name,
                        "error": str(e)
                    })
                    validation_report["passed"] = False
        
        # Framework coverage check
        if not assertions._check_framework_coverage(keywords):
            message = (
                f"Some keywords cover fewer than "
                f"{config.thresholds.min_framework_aspects} framework aspects"
            )
            validation_report["warnings"].append(message)
            dspy.Suggest(False, message)
        
        # Total keywords check
        if len(keywords) < config.thresholds.min_total_keywords:
            message = (
                f"Total keywords ({len(keywords)}) below minimum threshold "
                f"({config.thresholds.min_total_keywords})"
            )
            validation_report["warnings"].append(message)
            dspy.Suggest(False, message)
        
        # Required passes check
        if passes < config.required_passes:
            message = f"Insufficient assertion passes: {passes}/{config.required_passes}"
            validation_report["warnings"].append(message)
            if config.strict_mode:
                validation_report["passed"] = False
                validation_report["failed_assertions"].append({
                    "type": "required_passes",
                    "error": message
                })
        
        if config.detailed_logging:
            logger.debug("Validation report: %s", validation_report)
        
        return validation_report
    
    except Exception as e:
        error_msg = f"Unexpected error during keyword validation: {str(e)}"
        logger.error(error_msg, exc_info=True)
        validation_report["passed"] = False
        validation_report["failed_assertions"].append({
            "type": "unknown",
            "error": error_msg
        })
        return validation_report