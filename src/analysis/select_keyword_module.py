# File: select_keyword_module.py
import logging
from typing import Dict, Any, List
import dspy

from src.analysis.select_keyword import KeywordExtractionSignature

logger = logging.getLogger(__name__)

class SelectKeywordModule(dspy.Module):
    """
    DSPy module to extract keywords from transcript chunks using the 6Rs framework.
    """
    def __init__(self):
        super().__init__()
        self.chain = dspy.TypedChainOfThought(KeywordExtractionSignature)

    def forward(self, research_objectives: str, transcript_chunks: List[str], theoretical_framework: str = None) -> Dict[str, Any]:
        try:
            logger.debug("Running SelectKeywordModule for keyword extraction.")
            response = self.chain(
                research_objectives=research_objectives,
                transcript_chunks=transcript_chunks,
                theoretical_framework=theoretical_framework
            )
            keywords = response.get("keywords", [])
            logger.info(f"Extracted {len(keywords)} keywords.")
            return {
                "keywords": keywords
            }
        except Exception as e:
            logger.error(f"Error in SelectKeywordModule.forward: {e}", exc_info=True)
            return {
                "keywords": [],
                "error": f"Error occurred during keyword extraction: {str(e)}"
            }
