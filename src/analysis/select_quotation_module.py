import logging
from typing import Dict, Any, List
import dspy

from src.analysis.select_quotation import EnhancedQuotationModule
from src.assertions import (
    assert_pattern_representation,
    assert_research_objective_alignment,
    assert_selective_transcription,
    assert_creswell_categorization,
    assert_reader_engagement
)

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
            
            quotations = response.get("quotations", [])
            analysis = response.get("analysis", {})
            patterns = analysis.get("patterns_identified", [])

            assert_pattern_representation(quotations, patterns)
            assert_research_objective_alignment(quotations, research_objectives)
            assert_selective_transcription(quotations, transcript_chunk)
            assert_creswell_categorization(quotations)
            assert_reader_engagement(quotations)

            return response
        except Exception as e:
            logger.error(f"Error in SelectQuotationModule.forward: {e}", exc_info=True)
            return {}