# File: /Users/amankumarshrestha/Downloads/Thematic-AnalysisE/src/select_quotation_module.py

import logging
from typing import Dict, Any, List
import dspy
from select_quotation import SelectQuotationSignature

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
            response = self.chain(research_objectives=research_objectives, transcript_chunks=transcript_chunks)
            quotations = response.get("quotations", [])
            types_and_functions = response.get("types_and_functions", [])
            purpose = response.get("purpose", "")
            logger.info(f"Selected {len(quotations)} quotations.")
            return {
                "quotations": quotations,
                "types_and_functions": types_and_functions,
                "purpose": purpose
            }
        except Exception as e:
            logger.error(f"Error in SelectQuotationModule.forward: {e}", exc_info=True)
            return {
                "quotations": [],
                "types_and_functions": [],
                "purpose": ""
            }
