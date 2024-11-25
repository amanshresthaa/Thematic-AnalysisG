# src/analysis/select_quotation_module.py
import logging
from typing import Dict, Any, List
import dspy

from src.analysis.select_quotation import EnhancedQuotationModule
from src.assertions import (
    assert_relevant_quotations,
    assert_confidentiality,
    assert_diversity_of_quotations,
    assert_contextual_adequacy,
    assert_philosophical_alignment
)

logger = logging.getLogger(__name__)

class SelectQuotationModule(dspy.Module):
    """
    DSPy module to select and classify quotations based on research objectives,
    transcript chunks, and theoretical framework.
    """
    def __init__(self):
        super().__init__()
        self.enhanced_module = EnhancedQuotationModule()

    def forward(self, research_objectives: str, transcript_chunks: List[str], theoretical_framework: Dict[str, str]) -> Dict[str, Any]:
        try:
            logger.debug("Running SelectQuotationModule with theoretical framework.")
            response = self.enhanced_module.forward(
                research_objectives=research_objectives,
                transcript_chunks=transcript_chunks,
                theoretical_framework=theoretical_framework
            )
            quotations = response.get("quotations", [])
            analysis = response.get("analysis", "")
            logger.info(f"Selected {len(quotations)} quotations aligned with theoretical framework.")
            return {
                "quotations": quotations,
                "analysis": analysis
            }
        except Exception as e:
            logger.error(f"Error in SelectQuotationModule.forward: {e}", exc_info=True)
            return {
                "quotations": [],
                "analysis": ""
            }