# src/analysis/select_keyword_module.py
import logging
from typing import Dict, Any, List
import dspy

from src.analysis.select_keyword import KeywordExtractionSignature

logger = logging.getLogger(__name__)

class SelectKeywordModule(dspy.Module):
    """
    DSPy module to extract keywords from individual quotations using the 6Rs framework.
    """
    def __init__(self):
        super().__init__()
        self.chain = dspy.TypedChainOfThought(KeywordExtractionSignature)

    def forward(self, research_objectives: str, quotation: str, contextual_info: List[str], theoretical_framework: str = None) -> Dict[str, Any]:
        try:
            logger.debug(f"Running keyword extraction for quotation: {quotation[:100]}...")
            logger.debug(f"Research objectives: {research_objectives[:100]}...")
            logger.debug(f"Context info count: {len(contextual_info)}")
            
            response = self.chain(
                research_objectives=research_objectives,
                quotation=quotation,
                contextual_info=contextual_info,
                theoretical_framework=theoretical_framework
            )
            
            keywords = response.get("keywords", [])
            if not keywords:
                logger.warning("No keywords returned from chain")
            else:
                logger.info(f"Successfully extracted {len(keywords)} keywords")
                
            return {
                "keywords": keywords
            }
        except Exception as e:
            logger.error(f"Error in SelectKeywordModule.forward: {e}", exc_info=True)
            return {
                "keywords": [],
                "error": f"Error occurred during keyword extraction: {str(e)}"
            }