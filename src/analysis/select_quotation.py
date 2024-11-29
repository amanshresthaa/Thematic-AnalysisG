import logging
import re
import json
from typing import List, Dict, Any

import dspy

from src.assertions import (
    assert_pattern_representation,
    assert_research_objective_alignment,
    assert_selective_transcription,
    assert_creswell_categorization,
    assert_reader_engagement
)

logger = logging.getLogger(__name__)

class EnhancedQuotationSignature(dspy.Signature):
    """
    A comprehensive signature for conducting thematic analysis of interview transcripts
    following Braun and Clarke's (2006) methodology. This signature supports systematic
    qualitative analysis with robust pattern recognition and theoretical integration.
    """
    
    # Input Fields
    research_objectives: str = dspy.InputField(
        desc="Research goals and questions guiding the thematic analysis"
    )
    
    transcript_chunk: str = dspy.InputField(
        desc="Primary transcript segment for analysis"
    )
    
    contextualized_contents: List[str] = dspy.InputField(
        desc="Additional contextual content to support interpretation"
    )
    
    theoretical_framework: Dict[str, str] = dspy.InputField(
        desc="""Theoretical foundation including:
        - theory: Primary theoretical approach
        - philosophical_approach: Underlying philosophical foundation
        - rationale: Justification for chosen approach"""
    )
    
    # Output Fields
    transcript_info: Dict[str, Any] = dspy.OutputField(
        desc="""Comprehensive context information including:
        - transcript_chunk: Selected content
        - research_objectives: Analysis goals
        - theoretical_framework: Complete framework details"""
    )
    
    retrieved_chunks: List[str] = dspy.OutputField(
        desc="Collection of relevant transcript segments retrieved for analysis"
    )
    
    retrieved_chunks_count: int = dspy.OutputField(
        desc="Number of transcript chunks retrieved and analyzed"
    )
    
    used_chunk_ids: List[str] = dspy.OutputField(
        desc="Identifiers of transcript chunks utilized in the analysis"
    )
    
    quotations: List[Dict[str, Any]] = dspy.OutputField(
        desc="""Selected quotations with detailed analysis:
        - quotation: Exact quote text
        - creswell_category: Classification (longer/discrete/embedded)
        - classification: Content type
        - context: 
            * preceding_question
            * situation
            * pattern_representation
        - analysis_value:
            * relevance to objectives
            * pattern support
            * theoretical alignment"""
    )
    
    analysis: Dict[str, Any] = dspy.OutputField(
        desc="""Comprehensive thematic analysis including:
        - philosophical_underpinning: Analysis approach
        - patterns_identified: Key patterns discovered
        - theoretical_interpretation: Framework application
        - methodological_reflection:
            * pattern_robustness
            * theoretical_alignment
            * researcher_reflexivity
        - practical_implications: Applied insights"""
    )
    
    answer: Dict[str, Any] = dspy.OutputField(
        desc="""Analysis synthesis and contributions:
        - summary: Key findings
        - theoretical_contribution: Theory advancement
        - methodological_contribution:
            * approach
            * pattern_validity
            * theoretical_integration"""
    )

    def create_prompt(self, research_objectives: str, transcript_chunk: str, 
                     contextualized_contents: List[str], theoretical_framework: Dict[str, str]) -> str:
        """Creates the prompt for the language model."""
        
        # Format contextualized contents
        chunks_formatted = "\n\n".join([
            f"Content {i+1}:\n{content}" 
            for i, content in enumerate([transcript_chunk] + contextualized_contents)
        ])

        theory = theoretical_framework.get("theory", "")
        philosophical_approach = theoretical_framework.get("philosophical_approach", "")
        rationale = theoretical_framework.get("rationale", "")

        prompt = (
            f"You are an experienced qualitative researcher conducting a thematic analysis of "
            f"interview transcripts using Braun and Clarke's (2006) approach. Your task is to "
            f"analyze the provided transcript chunks while adhering to key principles from their "
            f"thematic analysis methodology.\n\n"

            f"First, review the transcript chunks and contextualized_content:\n\n"
            f"{chunks_formatted}\n\n"
            
            f"Research Objectives:\n"
            f"<research_objectives>\n"
            f"{research_objectives}\n"
            f"</research_objectives>\n\n"

            f"Theoretical Framework:\n"
            f"<theoretical_framework>\n"
            f"Theory: {theory}\n"
            f"Philosophical Approach: {philosophical_approach}\n"
            f"Rationale: {rationale}\n"
            f"</theoretical_framework>\n\n"
                    
            f"Your analysis should follow these steps:\n\n"

            f"1. **Quotation Selection**:\n"
            f"   - Select quotes that demonstrate robust patterns in the data.\n"
            f"   - Classify quotes using Creswell's categories:\n"
            f"     a) Longer quotations: For complex understandings\n"
            f"     b) Discrete quotations: For diverse perspectives\n"
            f"     c) Embedded quotations: Brief phrases showing text shifts\n"
            f"   - Ensure quotes enhance reader engagement and highlight unique findings.\n"
            f"   - Provide adequate context for accurate comprehension.\n\n"

            f"2. **Pattern Recognition**:\n"
            f"   - Identify patterns emerging from data rather than predetermined categories.\n"
            f"   - Support patterns with multiple quotations.\n"
            f"   - Maintain theoretical alignment while remaining open to emerging themes.\n"
            f"   - Document methodological decisions transparently.\n\n"

            f"3. **Theoretical Integration**:\n"
            f"   - Demonstrate clear philosophical underpinning.\n"
            f"   - Show how findings connect to the theoretical framework.\n"
            f"   - Practice researcher reflexivity throughout analysis.\n"
            f"   - Balance selectivity with comprehensiveness.\n\n"

            f"For each step of your analysis, wrap your analysis process in <analysis_process> tags to explain your thought process and reasoning before providing the final output. It's OK for this section to be quite long.\n\n"

            f"Your final output should follow this JSON structure:\n\n"

            "{\n"
            "  \"transcript_info\": {\n"
            "    \"transcript_chunk\": \"\",                    // Selected transcript content\n"
            "    \"research_objectives\": \"\",                 // Research goals guiding analysis\n"
            "    \"theoretical_framework\": {\n"
            "      \"theory\": \"\",                            // Primary theoretical approach\n"
            "      \"philosophical_approach\": \"\",            // Philosophical foundation\n"
            "      \"rationale\": \"\"                          // Justification for approach\n"
            "    }\n"
            "  },\n"
            "  \"retrieved_chunks\": [],                        // List of retrieved chunks (if needed)\n"
            "  \"retrieved_chunks_count\": 0,                   // Count of retrieved chunks\n"
            "  \"contextualized_contents\": [],                 // List of contextualized contents\n"
            "  \"used_chunk_ids\": [],                          // List of used chunk IDs\n"
            "  \"quotations\": [\n"
            "    {\n"
            "      \"quotation\": \"\",                         // Exact quote text\n"
            "      \"creswell_category\": \"\",                 // longer/discrete/embedded\n"
            "      \"classification\": \"\",                    // Content type\n"
            "      \"context\": {\n"
            "        \"preceding_question\": \"\",              // Prior question\n"
            "        \"situation\": \"\",                       // Context description\n"
            "        \"pattern_representation\": \"\"           // Pattern linkage\n"
            "      },\n"
            "      \"analysis_value\": {\n"
            "        \"relevance\": \"\",                       // Research objective alignment\n"
            "        \"pattern_support\": \"\",                 // Pattern evidence\n"
            "        \"theoretical_alignment\": \"\"            // Framework connection\n"
            "      }\n"
            "    }\n"
            "  ],\n"
            "  \"analysis\": {\n"
            "    \"philosophical_underpinning\": \"\",          // Analysis approach\n"
            "    \"patterns_identified\": [\"\"],               // Key patterns found\n"
            "    \"theoretical_interpretation\": \"\",          // Framework application\n"
            "    \"methodological_reflection\": {\n"
            "      \"pattern_robustness\": \"\",                // Pattern evidence\n"
            "      \"theoretical_alignment\": \"\",             // Framework fit\n"
            "      \"researcher_reflexivity\": \"\"             // Interpretation awareness\n"
            "    },\n"
            "    \"practical_implications\": \"\"               // Applied insights\n"
            "  },\n"
            "  \"answer\": {\n"
            "    \"summary\": \"\",                            // Key findings\n"
            "    \"theoretical_contribution\": \"\",            // Theory advancement\n"
            "    \"methodological_contribution\": {\n"
            "      \"approach\": \"\",                         // Method used\n"
            "      \"pattern_validity\": \"\",                 // Evidence quality\n"
            "      \"theoretical_integration\": \"\"           // Theory-data synthesis\n"
            "    }\n"
            "  }\n"
            "}\n\n"

            f"**Important Instructions:**\n"
            f"- **Your final output must strictly follow the JSON structure provided below, including all fields exactly as specified, even if some fields are empty. Do not omit any fields.**\n"
            f"- **Use double quotes for all strings.**\n"
            f"- **Do not include any additional commentary or text outside of the JSON structure.**\n\n"
            
            f"Remember to wrap your analysis process in <analysis_process> tags throughout your analysis to show your chain of thought before providing the final JSON output.\n\n"
        )
        return prompt

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parses the complete response from the language model."""
        try:
            # Use regex to extract JSON content within a code block tagged as json
            json_match = re.search(r"```json\s*(\{.*\})\s*```", response, re.DOTALL)
            if not json_match:
                logger.error("No valid JSON found in response.")
                logger.debug(f"Full response received: {response}")
                return {}
            json_string = json_match.group(1)

            response_json = json.loads(json_string)
            return response_json
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed: {e}, Response: {response}")
            return {}
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {}

    def forward(self, research_objectives: str, transcript_chunk: str, 
                contextualized_contents: List[str], theoretical_framework: Dict[str, str]) -> Dict[str, Any]:
        """Executes the quotation selection and analysis process with retry mechanism."""
        for attempt in range(3):  # Retry mechanism with up to 3 attempts
            try:
                logger.debug(f"Attempt {attempt + 1} - Starting enhanced quotation selection and analysis process.")
                
                # Generate the prompt
                prompt = self.create_prompt(
                    research_objectives,
                    transcript_chunk,
                    contextualized_contents,
                    theoretical_framework
                )
                
                # Generate response from the language model
                response = self.language_model.generate(
                    prompt=prompt,
                    max_tokens=6000,
                    temperature=0.5
                ).strip()
                
                logger.debug(f"Attempt {attempt + 1} - Response received from language model.")
                
                # Parse the complete response
                parsed_response = self.parse_response(response)
                
                if not parsed_response:
                    raise ValueError("Parsed response is empty. Possibly invalid JSON.")
                
                # Extract components
                quotations = parsed_response.get("quotations", [])
                analysis = parsed_response.get("analysis", {})
                
                # Apply assertions
                assert_pattern_representation(quotations, analysis.get("patterns_identified", []))
                assert_research_objective_alignment(quotations, research_objectives)
                assert_selective_transcription(quotations, transcript_chunk)
                assert_creswell_categorization(quotations)
                assert_reader_engagement(quotations)
                
                
                logger.info(f"Attempt {attempt + 1} - Successfully completed analysis with {len(quotations)} quotations.")
                return parsed_response

            except AssertionError as af:
                logger.warning(f"Attempt {attempt + 1} - Assertion failed during analysis: {af}")
                logger.debug(f"Attempt {attempt + 1} - Response causing assertion failure: {response}")
                # Handle failed assertion by refining the prompt and retrying
                parsed_response = self.handle_failed_assertion(
                    af, research_objectives, transcript_chunk, contextualized_contents, theoretical_framework
                )
                if parsed_response:
                    return parsed_response
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} - Error in EnhancedQuotationSignature.forward: {e}", exc_info=True)
                # Continue to next attempt
                
        logger.error(f"Failed to generate valid output after multiple attempts. Last response: {response}")
        return {}

    def handle_failed_assertion(self, assertion_failure: AssertionError,
                                research_objectives: str, transcript_chunk: str,
                                contextualized_contents: List[str],
                                theoretical_framework: Dict[str, str]) -> Dict[str, Any]:
        """
        Handles failed assertions by attempting to generate improved analysis.
        This method refines the prompt based on the specific assertion failure and retries the generation.
        """
        try:
            logger.debug("Handling failed assertion by refining the prompt.")

            # Generate the initial prompt
            focused_prompt = self.create_prompt(
                research_objectives,
                transcript_chunk,
                contextualized_contents,
                theoretical_framework
            )
            
            # Append instructions to address the specific assertion failure
            focused_prompt += (
                f"\n\nThe previous attempt failed because: {assertion_failure}\n"
                f"Please ensure that your analysis addresses this specific issue while maintaining "
                f"all other requirements for thorough theoretical analysis."
            )

            # Generate a new response with the refined prompt
            response = self.language_model.generate(
                prompt=focused_prompt,
                max_tokens=2000,
                temperature=0.5
            ).strip()

            logger.debug("Response received from language model after handling assertion failure.")

            # Parse the new response
            parsed_response = self.parse_response(response)
            
            if not parsed_response:
                raise ValueError("Parsed response is empty after handling assertion failure.")
            
            # Re-apply all assertions
            quotations = parsed_response.get("quotations", [])
            analysis = parsed_response.get("analysis", {})
            
            assert_pattern_representation(quotations, analysis.get("patterns_identified", []))
            assert_research_objective_alignment(quotations, research_objectives)
            assert_selective_transcription(quotations, transcript_chunk)
            assert_creswell_categorization(quotations)
            assert_reader_engagement(quotations)
            

            logger.info("Successfully handled failed assertion and obtained valid analysis.")
            return parsed_response

        except AssertionError as af_inner:
            logger.error(f"Refined analysis still failed assertions: {af_inner}")
            return {}
        except Exception as e:
            logger.error(f"Error in handle_failed_assertion: {e}", exc_info=True)
            return {}

class EnhancedQuotationModule(dspy.Module):
    """
    DSPy module implementing the enhanced quotation selection and theoretical analysis functionality.
    """
    def __init__(self):
        super().__init__()
        self.chain = dspy.TypedChainOfThought(EnhancedQuotationSignature)

    def forward(self, research_objectives: str, transcript_chunk: str, 
                contextualized_contents: List[str], theoretical_framework: Dict[str, str]) -> Dict[str, Any]:
        try:
            logger.debug("Running EnhancedQuotationModule with integrated theoretical analysis.")
            response = self.chain(
                research_objectives=research_objectives,
                transcript_chunk=transcript_chunk,
                contextualized_contents=contextualized_contents,
                theoretical_framework=theoretical_framework
            )
            return response
        except Exception as e:
            logger.error(f"Error in EnhancedQuotationModule.forward: {e}", exc_info=True)
            return {}
