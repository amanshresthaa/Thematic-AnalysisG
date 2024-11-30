import logging
from typing import Dict, Any, List
import dspy

from src.analysis.extract_keyword import KeywordExtractionSignature
from src.assertions import (
    assert_keyword_realness,
    assert_keyword_richness,
    assert_keyword_repetition,
    assert_keyword_rationale,
    assert_keyword_repartee,
    assert_keyword_regal
)

logger = logging.getLogger(__name__)

class KeywordExtractionModule(dspy.Module):
    """
    DSPy module for extracting and analyzing keywords using the 6Rs framework.
    """
    def __init__(self):
        super().__init__()
        self.chain = dspy.TypedChainOfThought(KeywordExtractionSignature)
    
    def forward(self, research_objectives: str, transcript_chunk: str,
                contextualized_contents: List[str], theoretical_framework: Dict[str, str]) -> Dict[str, Any]:
        """
        Execute keyword extraction and analysis with the 6Rs framework.
        """
        try:
            logger.debug("Starting keyword extraction with 6Rs framework")
            
            # Get initial response
            response = self.chain(
                research_objectives=research_objectives,
                transcript_chunk=transcript_chunk,
                contextualized_contents=contextualized_contents,
                theoretical_framework=theoretical_framework
            )
            
            # Extract components for validation
            keywords = response.get("keywords", [])
            
            # Apply assertions
            assert_keyword_realness(keywords)
            assert_keyword_richness(keywords)
            assert_keyword_repetition(keywords)
            assert_keyword_rationale(keywords, theoretical_framework)
            assert_keyword_repartee(keywords)
            assert_keyword_regal(keywords)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in KeywordExtractionModule.forward: {e}")
            return {}