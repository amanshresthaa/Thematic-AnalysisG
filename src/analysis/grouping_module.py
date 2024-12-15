# grouping_module.py

import logging
from typing import Dict, Any, List
import dspy
from analysis.grouping import GroupingAnalysisSignature

logger = logging.getLogger(__name__)

class GroupingAnalysisModule(dspy.Module):
    """
    DSPy module for grouping similar codes into potential themes,
    building upon code definitions, research objectives, and 
    a theoretical framework.
    """

    def __init__(self):
        super().__init__()
        self.chain = dspy.TypedChainOfThought(GroupingAnalysisSignature)

    def forward(
        self,
        research_objectives: str,
        theoretical_framework: Dict[str, str],
        codes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Executes the grouping analysis using the defined signature 
        and the language model.
        """
        logger.info("Starting grouping analysis.")
        try:
            response = self.chain(
                research_objectives=research_objectives,
                theoretical_framework=theoretical_framework,
                codes=codes
            )

            if not response.get("groupings"):
                logger.warning("No groupings were generated. Possible issue with inputs or prompt formulation.")

            logger.info("Successfully completed grouping analysis.")
            return response
        except Exception as e:
            logger.error(f"Error during grouping analysis: {e}", exc_info=True)
            return {}
