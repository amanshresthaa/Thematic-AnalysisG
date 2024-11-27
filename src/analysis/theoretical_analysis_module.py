# src/analysis/theoretical_analysis_module.py
import logging
from typing import Dict, Any, List
import dspy

from src.analysis.theoretical_analysis import TheoreticalAnalysisSignature
from src.assertions import (
    assert_patterns_identified,
    assert_theoretical_interpretation,
    assert_research_alignment
)

logger = logging.getLogger(__name__)

class TheoreticalAnalysisModule(dspy.Module):
    """
    DSPy module to perform theoretical analysis on selected quotations.
    """
    def __init__(self):
        super().__init__()
        self.chain = dspy.TypedChainOfThought(TheoreticalAnalysisSignature)

    def forward(self, quotations: List[Dict[str, Any]], theoretical_framework: Dict[str, str], research_objectives: str) -> Dict[str, Any]:
        try:
            logger.debug("Running TheoreticalAnalysisModule.")
            response = self.chain(
                quotations=quotations,
                theoretical_framework=theoretical_framework,
                research_objectives=research_objectives
            )
            patterns_identified = response.get("patterns_identified", [])
            theoretical_interpretation = response.get("theoretical_interpretation", "")
            research_alignment = response.get("research_alignment", "")
            practical_implications = response.get("practical_implications", "")
            return {
                "patterns_identified": patterns_identified,
                "theoretical_interpretation": theoretical_interpretation,
                "research_alignment": research_alignment,
                "practical_implications": practical_implications
            }
        except Exception as e:
            logger.error(f"Error in TheoreticalAnalysisModule.forward: {e}", exc_info=True)
            return {}
