# File: src/analysis/select_quotation_module_alt.py
import logging
from typing import Dict, Any, List
import dspy

from src.analysis.select_quotation_alt import EnhancedQuotationModuleAlt

logger = logging.getLogger(__name__)

class SelectQuotationModuleAlt(dspy.Module):
    """
    DSPy module to select quotations based on research objectives, transcript chunks, and theoretical framework.
    Utilizes EnhancedQuotationModuleAlt for robust assertion-based selection with an alternative prompt.
    """
    def __init__(self):
        super().__init__()
        self.enhanced_module = EnhancedQuotationModuleAlt()

    def forward(self, research_objectives: str, transcript_chunks: List[str], theoretical_framework: str) -> Dict[str, Any]:
        try:
            logger.debug("Running SelectQuotationModuleAlt with theoretical framework.")
            response = self.enhanced_module.forward(
                research_objectives=research_objectives,
                transcript_chunks=transcript_chunks,
                theoretical_framework=theoretical_framework
            )
            quotations = response.get("quotations", [])
            analysis = response.get("analysis", "")
            logger.info(f"Selected {len(quotations)} quotations aligned with theoretical framework (Alt).")
            return {
                "quotations": quotations,
                "analysis": analysis
            }
        except Exception as e:
            logger.error(f"Error in SelectQuotationModuleAlt.forward: {e}", exc_info=True)
            return {
                "quotations": [],
                "analysis": ""
            }
