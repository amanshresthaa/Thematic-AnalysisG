# src/analysis/theoretical_analysis.py
import logging
from typing import List, Dict, Any
import dspy
import json

from src.assertions import (
    assert_patterns_identified,
    assert_theoretical_interpretation,
    assert_research_alignment
)

logger = logging.getLogger(__name__)

class TheoreticalAnalysisSignature(dspy.Signature):
    quotations: List[Dict[str, Any]] = dspy.InputField(
        desc="List of selected quotations with their classifications and contexts"
    )
    theoretical_framework: Dict[str, str] = dspy.InputField(
        desc="The theoretical and philosophical framework guiding the analysis"
    )
    research_objectives: str = dspy.InputField(
        desc="The research objectives statement"
    )
    patterns_identified: List[str] = dspy.OutputField(
        desc="List of patterns identified from the quotations"
    )
    theoretical_interpretation: str = dspy.OutputField(
        desc="Analysis of the patterns through the theoretical lens"
    )
    research_alignment: str = dspy.OutputField(
        desc="Explanation of how the analysis aligns with the research objectives"
    )
    practical_implications: str = dspy.OutputField(
        desc="Implications of the findings for practice or further research"
    )

    def create_prompt(self, quotations: List[Dict[str, Any]], theoretical_framework: Dict[str, str], research_objectives: str) -> str:
        prompt = (
            f"You are performing a theoretical analysis based on the provided quotations.\n\n"
            f"Research Objectives:\n{research_objectives}\n\n"
            f"Theoretical Framework:\n{theoretical_framework}\n\n"
            f"Quotations:\n{json.dumps(quotations, indent=2)}\n\n"
            f"Task: Identify patterns, provide theoretical interpretation, and explain how the analysis aligns with the research objectives.\n"
            f"Also, discuss the practical implications of the findings.\n"
            f"Return the results in valid JSON format with the following structure:\n"
            f"{{\n"
            f"  \"patterns_identified\": [\"pattern1\", \"pattern2\", ...],\n"
            f"  \"theoretical_interpretation\": \"Your analysis here\",\n"
            f"  \"research_alignment\": \"Explanation of alignment\",\n"
            f"  \"practical_implications\": \"Implications of findings\"\n"
            f"}}"
        )
        return prompt

    def forward(self, quotations: List[Dict[str, Any]], theoretical_framework: Dict[str, str], research_objectives: str) -> Dict[str, Any]:
        try:
            logger.debug("Starting theoretical analysis.")
            prompt = self.create_prompt(quotations, theoretical_framework, research_objectives)
            response = self.language_model.generate(
                prompt=prompt,
                max_tokens=1500,
                temperature=1.0
            ).strip()

            # Parse the response
            response_json = json.loads(response)
            patterns_identified = response_json.get("patterns_identified", [])
            theoretical_interpretation = response_json.get("theoretical_interpretation", "")
            research_alignment = response_json.get("research_alignment", "")
            practical_implications = response_json.get("practical_implications", "")

            # Apply assertions
            assert_patterns_identified(patterns_identified)
            assert_theoretical_interpretation(theoretical_interpretation)
            assert_research_alignment(research_alignment)

            return {
                "patterns_identified": patterns_identified,
                "theoretical_interpretation": theoretical_interpretation,
                "research_alignment": research_alignment,
                "practical_implications": practical_implications
            }

        except AssertionError as af:
            logger.warning(f"Assertion failed during theoretical analysis: {af}")
            return self.handle_failed_assertion(af, quotations, theoretical_framework, research_objectives)
        except Exception as e:
            logger.error(f"Error in TheoreticalAnalysisSignature.forward: {e}", exc_info=True)
            return {}

    def handle_failed_assertion(self, assertion_failure: AssertionError,
                                quotations: List[Dict[str, Any]],
                                theoretical_framework: Dict[str, str],
                                research_objectives: str) -> Dict[str, Any]:
        """Handles failed assertions by attempting to generate improved analysis."""
        try:
            focused_prompt = (
                f"The previous attempt failed because: {assertion_failure}\n\n"
                f"Research Objectives:\n{research_objectives}\n\n"
                f"Theoretical Framework:\n{theoretical_framework}\n\n"
                f"Please ensure that the analysis addresses the following:\n"
                f"1. Identifies clear patterns from the quotations.\n"
                f"2. Provides a theoretical interpretation aligning with the theoretical framework.\n"
                f"3. Explains how the analysis aligns with the research objectives.\n"
                f"4. Discusses practical implications of the findings.\n\n"
                f"Quotations:\n{json.dumps(quotations, indent=2)}\n\n"
                f"Return the results in the specified JSON format."
            )

            response = self.language_model.generate(
                prompt=focused_prompt,
                max_tokens=1500,
                temperature=0.5
            ).strip()

            # Parse the response
            response_json = json.loads(response)
            patterns_identified = response_json.get("patterns_identified", [])
            theoretical_interpretation = response_json.get("theoretical_interpretation", "")
            research_alignment = response_json.get("research_alignment", "")
            practical_implications = response_json.get("practical_implications", "")

            # Re-apply assertions
            assert_patterns_identified(patterns_identified)
            assert_theoretical_interpretation(theoretical_interpretation)
            assert_research_alignment(research_alignment)

            return {
                "patterns_identified": patterns_identified,
                "theoretical_interpretation": theoretical_interpretation,
                "research_alignment": research_alignment,
                "practical_implications": practical_implications
            }

        except AssertionError as af_inner:
            logger.error(f"Refined analysis still failed assertions: {af_inner}")
            return {}
        except Exception as e:
            logger.error(f"Error in handle_failed_assertion: {e}", exc_info=True)
            return {}
