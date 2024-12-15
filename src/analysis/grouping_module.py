import logging
from typing import Dict, Any, List
import dspy
from grouping import GroupingAnalysisSignature, group_codes

logger = logging.getLogger(__name__)

class GroupingAnalysisModule(dspy.Module):
    """
    DSPy module for grouping codes into sets without introducing theme labels.
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
        and grouping logic, producing only sets of codes.
        """
        logger.info("Starting grouping analysis.")

        try:
            # Execute the grouping logic
            groupings = group_codes(
                research_objectives=research_objectives,
                theoretical_framework=theoretical_framework,
                codes=codes
            )

            if not groupings:
                logger.warning("No groupings were generated.")
                return {
                    "groupings": [],
                    "message": "No groupings were generated. Please review the input codes."
                }

            logger.info(f"Grouping analysis completed with {len(groupings)} groups formed.")
            return {"groupings": groupings}

        except Exception as e:
            logger.error(f"Error during grouping analysis: {e}", exc_info=True)
            return {
                "error": f"Grouping analysis failed due to an unexpected error: {str(e)}"
            }
