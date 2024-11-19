import logging
from typing import Dict, Any, List
import dspy

from .select_quotation import SelectQuotationSignature

logger = logging.getLogger(__name__)

class SelectQuotationModule(dspy.Module):
    """
    DSPy module to select quotations based on research objectives and transcript chunks.
    """
    def __init__(self):
        super().__init__()
        self.chain = dspy.TypedChainOfThought(SelectQuotationSignature)

    def forward(self, research_objectives: str, transcript_chunks: List[str]) -> Dict[str, Any]:
        try:
            logger.debug("Running SelectQuotationModule.")
            
            # Validate inputs
            if not research_objectives or not transcript_chunks:
                logger.warning("Missing required inputs for quotation selection.")
                return {
                    "quotations": [],
                    "purpose": ""
                }

            # Process through chain
            response = self.chain(
                research_objectives=research_objectives,
                transcript_chunks=transcript_chunks
            )

            # Extract and validate results
            quotations = response.get("quotations", [])
            purpose = response.get("purpose", "")

            # Log results
            logger.info(f"Selected {len(quotations)} quotations.")
            if quotations:
                logger.debug("First quotation sample: " + str(quotations[0]))

            return {
                "quotations": quotations,
                "purpose": purpose
            }

        except Exception as e:
            logger.error(f"Error in SelectQuotationModule.forward: {e}", exc_info=True)
            return {
                "quotations": [],
                "purpose": ""
            }