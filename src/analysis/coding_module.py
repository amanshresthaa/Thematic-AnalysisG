# analysis/coding_module.py
import logging
from typing import Dict, Any, List
import dspy
from analysis.coding import CodingAnalysisSignature

logger = logging.getLogger('my_logger')

class CodingAnalysisModule(dspy.Module):
    """
    DSPy module for developing and analyzing codes using the 6Rs framework.
    """
    def __init__(self):
        super().__init__()
        self.chain = dspy.TypedChainOfThought(CodingAnalysisSignature)

    def forward(self, research_objectives: str, quotation: str,
                keywords: List[str], contextualized_contents: List[str],
                theoretical_framework: Dict[str, str]) -> Dict[str, Any]:
        logger.info("Starting coding analysis.")
        logger.debug(f"Input parameters: Research Objectives='{research_objectives[:100]}', "
                     f"Quotation='{quotation[:100]}', Keywords={keywords}, "
                     f"Contextualized Contents={contextualized_contents[:2]}, "
                     f"Theoretical Framework={theoretical_framework}")

        try:
            response = self.chain(
                research_objectives=research_objectives,
                quotation=quotation,
                keywords=keywords,
                contextualized_contents=contextualized_contents,
                theoretical_framework=theoretical_framework
            )
            logger.info("Successfully completed coding analysis.")
            logger.debug(f"Response: {response}")

            if not response.get("codes"):
                logger.warning("No codes were generated. Possible issue with inputs or prompt formulation.")
            
            return response
        except Exception as e:
            logger.error(f"Error during coding analysis: {e}", exc_info=True)
            return {}
