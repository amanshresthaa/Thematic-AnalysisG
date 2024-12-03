
# src/analysis/coding_module.py

import logging
from typing import Dict, Any, List
import dspy

from src.analysis.coding import CodingAnalysisSignature

logger = logging.getLogger(__name__)

class CodingAnalysisModule(dspy.Module):
    """
    DSPy module for developing and analyzing codes using the 6Rs framework,
    building upon extracted keywords.
    """
    def __init__(self):
        super().__init__()
        self.chain = dspy.TypedChainOfThought(CodingAnalysisSignature)

    def forward(self, research_objectives: str, quotation: str,
                keywords: List[Dict[str, Any]], contextualized_contents: List[str],
                theoretical_framework: Dict[str, str]) -> Dict[str, Any]:
        """
        Execute coding analysis with the 6Rs framework, building upon extracted keywords.

        Args:
            research_objectives (str): Research goals and questions
            quotation (str): Selected quotation for analysis
            keywords (List[Dict[str, Any]]): Previously extracted keywords
            contextualized_contents (List[str]): Additional context
            theoretical_framework (Dict[str, str]): Theoretical foundation

        Returns:
            Dict[str, Any]: Complete coding analysis results including codes and analysis
        """
        try:
            logger.debug("Running CodingAnalysisModule with integrated code assertions.")
            response = self.chain(
                research_objectives=research_objectives,
                quotation=quotation,
                keywords=keywords,
                contextualized_contents=contextualized_contents,
                theoretical_framework=theoretical_framework
            )
            return response
        except Exception as e:
            logger.error(f"Error in CodingAnalysisModule.forward: {e}", exc_info=True)
            return {}


