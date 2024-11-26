# analysis/select_quotation_module.py
import logging
from typing import Dict, Any, List
import dspy

from src.analysis.select_quotation import EnhancedQuotationModule

logger = logging.getLogger(__name__)

class SelectQuotationModule(dspy.Module):
    """
    DSPy module to select and analyze quotations based on research objectives,
    transcript chunks, and theoretical framework.
    """
    def __init__(self):
        super().__init__()
        self.enhanced_module = EnhancedQuotationModule()

    def forward(self, research_objectives: str, transcript_chunk: str, 
                contextualized_contents: List[str], theoretical_framework: Dict[str, str]) -> Dict[str, Any]:
        try:
            logger.debug("Running SelectQuotationModule with integrated theoretical analysis.")
            response = self.enhanced_module.forward(
                research_objectives=research_objectives,
                transcript_chunk=transcript_chunk,
                contextualized_contents=contextualized_contents,
                theoretical_framework=theoretical_framework
            )
            # The response already includes 'transcript_info', 'quotations', 'analysis', and 'answer'
            return response
        except Exception as e:
            logger.error(f"Error in SelectQuotationModule.forward: {e}", exc_info=True)
            return {}
