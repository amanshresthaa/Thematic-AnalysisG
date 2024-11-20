# File: src/analysis/select_quotation_alt.py
import logging
from typing import List, Dict, Any
import dspy
import json

from src.assertions_alt import (
    assert_relevant_quotations,
    assert_confidentiality,
    assert_diversity_of_quotations,
    assert_contextual_adequacy,
    assert_philosophical_alignment
)

logger = logging.getLogger(__name__)

class EnhancedQuotationSignatureAlt(dspy.Signature):
    """
    Enhanced signature for selecting relevant quotations based on an alternative thematic analysis approach.
    """
    research_objectives: str = dspy.InputField(
        desc="The research objectives that provide focus for conducting the analysis"
    )
    transcript_chunks: List[str] = dspy.InputField(
        desc="Chunks of transcript from which quotations are to be selected"
    )
    theoretical_framework: str = dspy.InputField(
        desc="The theoretical and philosophical framework guiding the analysis"
    )
    quotations: List[Dict[str, Any]] = dspy.OutputField(
        desc="List of selected quotations with their types and analytical context"
    )
    analysis: str = dspy.OutputField(
        desc="Analysis of how the selected quotations support the research objectives"
    )

    def parse_quotations(self, response: str) -> List[Dict[str, Any]]:
        """Parses quotations from the LM response."""
        try:
            # Assuming the LM returns a JSON-formatted string for easy parsing
            response_json = json.loads(response)
            quotation_list = response_json.get("quotations", [])
            return quotation_list
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing quotations: {e}")
            return []

    def extract_analysis(self, response: str) -> str:
        """Extracts analysis section from the LM response."""
        try:
            # Assuming the LM returns a JSON-formatted string with an 'analysis' field
            response_json = json.loads(response)
            analysis = response_json.get("analysis", "")
            return analysis
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed: {e}")
            return ""
        except Exception as e:
            logger.error(f"Error extracting analysis: {e}")
            return ""

    def create_prompt(self, research_objectives: str, transcript_chunks: List[str], theoretical_framework: str) -> str:
        """Creates the alternative prompt for the language model."""
        chunks_formatted = "\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(transcript_chunks)])
        prompt = (
            f"Alternative Thematic Analysis Prompt:\n\n"
            f"Research Objectives:\n{research_objectives}\n\n"
            f"Theoretical Framework:\n{theoretical_framework}\n\n"
            f"Transcript Chunks:\n{chunks_formatted}\n\n"
            f"Task: [Your alternative prompt instructions here]\n\n"
            f"Return the analysis in valid JSON format."
        )
        return prompt

    def forward(self, research_objectives: str, transcript_chunks: List[str], theoretical_framework: str) -> Dict[str, Any]:
        try:
            logger.debug("Starting enhanced quotation selection process with theoretical framework (Alt).")
            
            # Generate the prompt
            prompt = self.create_prompt(research_objectives, transcript_chunks, theoretical_framework)
            
            # Generate response
            response = self.language_model.generate(
                prompt=prompt,
                max_tokens=1500,
                temperature=0.7
            ).strip()
            
            # Parse quotations and analysis
            quotation_list = self.parse_quotations(response)
            analysis = self.extract_analysis(response)
            
            # Apply assertions
            assert_relevant_quotations(quotation_list, self.research_objectives)
            assert_confidentiality(quotation_list, sensitive_keywords=['confidential', 'secret'])
            assert_diversity_of_quotations(quotation_list, min_participants=3)
            assert_contextual_adequacy(quotation_list, self.transcript_chunks)
            assert_philosophical_alignment(quotation_list, self.theoretical_framework)
            
            logger.info(f"Successfully selected {len(quotation_list)} quotations aligned with theoretical framework (Alt).")
            return {
                "quotations": quotation_list,
                "analysis": analysis
            }

        except AssertionError as af:
            logger.warning(f"Assertion failed during quotation selection (Alt): {af}")
            return self.handle_failed_assertion(af, research_objectives, transcript_chunks, theoretical_framework)
        except Exception as e:
            logger.error(f"Error in EnhancedQuotationSignatureAlt.forward: {e}", exc_info=True)
            return {
                "quotations": [],
                "analysis": f"Error occurred during quotation selection (Alt): {str(e)}"
            }

    def handle_failed_assertion(self, assertion_failure: AssertionError,
                               research_objectives: str, transcript_chunks: List[str],
                               theoretical_framework: str) -> Dict[str, Any]:
        """Handles failed assertions by attempting to generate improved quotations."""
        try:
            focused_prompt = (
                f"The previous attempt failed because: {assertion_failure}\n\n"
                f"Research Objectives:\n{research_objectives}\n\n"
                f"Theoretical Framework:\n{theoretical_framework}\n\n"
                "Please select quotations that specifically address this issue by ensuring:\n"
                "1. Proper quotation type (Discrete/Embedded/Longer)\n"
                "2. Clear function and purpose\n"
                "3. Sufficient context and ethical considerations\n"
                "4. Strong theoretical alignment\n"
                "5. Diverse participant representation\n\n"
                "Transcript Chunks:\n" +
                "\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(transcript_chunks)])
            )

            response = self.language_model.generate(
                prompt=focused_prompt,
                max_tokens=1500,
                temperature=0.5
            ).strip()

            quotation_list = self.parse_quotations(response)
            analysis = self.extract_analysis(response)

            # Re-apply assertions on the new quotations
            assert_relevant_quotations(quotation_list, research_objectives)
            assert_confidentiality(quotation_list, sensitive_keywords=['confidential', 'secret'])
            assert_diversity_of_quotations(quotation_list, min_participants=3)
            assert_contextual_adequacy(quotation_list, transcript_chunks)
            assert_philosophical_alignment(quotation_list, theoretical_framework)

            return {
                "quotations": quotation_list,
                "analysis": f"Note: Quotations were refined to address: {assertion_failure}\n\n{analysis}"
            }

        except AssertionError as af_inner:
            logger.error(f"Refined quotations still failed assertions (Alt): {af_inner}")
            return {
                "quotations": [],
                "analysis": f"Failed to select appropriate quotations after refinement (Alt): {str(af_inner)}"
            }
        except Exception as e:
            logger.error(f"Error in handle_failed_assertion (Alt): {e}", exc_info=True)
            return {
                "quotations": [],
                "analysis": f"Failed to select appropriate quotations (Alt): {str(assertion_failure)}"
            }

class EnhancedQuotationModuleAlt(dspy.Module):
    """
    DSPy module implementing the enhanced quotation selection functionality with an alternative prompt.
    """
    def __init__(self):
        super().__init__()
        self.chain = dspy.TypedChainOfThought(EnhancedQuotationSignatureAlt)

    def forward(self, research_objectives: str, transcript_chunks: List[str], theoretical_framework: str) -> Dict[str, Any]:
        try:
            logger.debug("Running EnhancedQuotationModuleAlt with theoretical framework.")
            response = self.chain(
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
            logger.error(f"Error in EnhancedQuotationModuleAlt.forward: {e}", exc_info=True)
            return {
                "quotations": [],
                "analysis": ""
            }
