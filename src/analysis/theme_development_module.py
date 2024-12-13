#theme_development_module.py
import logging
from typing import Dict, Any, List
import dspy
from .theme_development import ThemedevelopmentAnalysisSignature

logger = logging.getLogger(__name__)

class ThemedevelopmentAnalysisModule(dspy.Module):
    """
    DSPy module for developing and refining themes from previously derived codes.
    Now does not require quotation or keywords.
    """

    def __init__(self):
        super().__init__()
        self.chain = dspy.TypedChainOfThought(ThemedevelopmentAnalysisSignature)

    def forward(self, 
                research_objectives: str, 
                codes: List[Dict[str, Any]], 
                theoretical_framework: Dict[str, str]) -> Dict[str, Any]:
        logger.info("Starting theme development analysis.")
        try:
            response = self.chain(
                research_objectives=research_objectives,
                codes=codes,
                theoretical_framework=theoretical_framework
            )

            logger.info("Successfully completed theme development analysis.")
            if not response.get("themes"):
                logger.warning("No themes were generated. Possible issue with inputs or prompt formulation.")
            
            return response
        except Exception as e:
            logger.error(f"Error during theme development analysis: {e}", exc_info=True)
            return {}
