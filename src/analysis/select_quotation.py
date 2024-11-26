#analysis/select_quotation.py
import logging
from typing import List, Dict, Any
import dspy
import json
import re
from src.assertions import (
    assert_relevant_quotations,
    assert_confidentiality,
    assert_diversity_of_quotations,
    assert_contextual_adequacy,
    assert_philosophical_alignment,
    assert_patterns_identified,
    assert_theoretical_interpretation,
    assert_research_alignment
)

logger = logging.getLogger(__name__)

class EnhancedQuotationSignature(dspy.Signature):
    """
    Enhanced signature for selecting and analyzing quotations using Braun and Clarke's thematic analysis.
    """
    research_objectives: str = dspy.InputField(
        desc="The research objectives that provide focus for conducting the analysis"
    )
    transcript_chunk: str = dspy.InputField(
        desc="The main transcript chunk being analyzed"
    )
    contextualized_contents: List[str] = dspy.InputField(
        desc="List of related content providing additional context"
    )
    theoretical_framework: Dict[str, str] = dspy.InputField(
        desc="The theoretical and philosophical framework guiding the analysis"
    )
    transcript_info: Dict[str, Any] = dspy.OutputField(
        desc="Information about the transcript and analysis context"
    )
    quotations: List[Dict[str, Any]] = dspy.OutputField(
        desc="List of selected quotations with their classifications and analyses"
    )
    analysis: Dict[str, Any] = dspy.OutputField(
        desc="Comprehensive analysis including patterns and theoretical interpretation"
    )
    answer: Dict[str, Any] = dspy.OutputField(
        desc="Summary of findings and contributions"
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
    """Parses the complete response from the language model and ensures it matches the desired JSON structure."""
    try:
        # Extract JSON content within a code block tagged as json
        json_match = re.search(r"```json\s*(\{.*\})\s*```", response, re.DOTALL)
        if not json_match:
            logger.error("No valid JSON found in response.")
            return {}
        json_string = json_match.group(1)

        response_json = json.loads(json_string)

        # Ensure all required top-level fields are present
        desired_structure = {
            "transcript_info": {
                "transcript_chunk": "",
                "research_objectives": "",
                "theoretical_framework": {
                    "theory": "",
                    "philosophical_approach": "",
                    "rationale": ""
                }
            },
            "retrieved_chunks": [],
            "retrieved_chunks_count": 0,
            "contextualized_contents": [],
            "used_chunk_ids": [],
            "quotations": [],
            "analysis": {
                "philosophical_underpinning": "",
                "patterns_identified": [""],
                "theoretical_interpretation": "",
                "methodological_reflection": {
                    "pattern_robustness": "",
                    "theoretical_alignment": "",
                    "researcher_reflexivity": ""
                },
                "practical_implications": ""
            },
            "answer": {
                "summary": "",
                "theoretical_contribution": "",
                "methodological_contribution": {
                    "approach": "",
                    "pattern_validity": "",
                    "theoretical_integration": ""
                }
            }
        }

        def ensure_structure(source: Dict[str, Any], target: Dict[str, Any]):
            for key, value in target.items():
                if key not in source:
                    source[key] = value
                else:
                    if isinstance(value, dict):
                        ensure_structure(source[key], value)
                    elif isinstance(value, list):
                        if not isinstance(source[key], list):
                            source[key] = value
                    else:
                        if source[key] == "":
                            source[key] = value

        ensure_structure(response_json, desired_structure)

        # Enhance quotations with required fields
        if "quotations" in response_json:
            for quote in response_json["quotations"]:
                quote.setdefault("creswell_category", "")
                quote.setdefault("classification", "")
                quote.setdefault("context", {
                    "preceding_question": "",
                    "situation": "",
                    "pattern_representation": ""
                })
                quote.setdefault("analysis_value", {
                    "relevance": "",
                    "pattern_support": "",
                    "theoretical_alignment": ""
                })
        else:
            response_json["quotations"] = []

        return response_json

    except json.JSONDecodeError as e:
        logger.error(f"JSON decoding failed: {e}, Response: {response}")
        return {}
    except Exception as e:
        logger.error(f"Error parsing response: {e}")
        return {}



    def forward(self, research_objectives: str, transcript_chunk: str, 
                contextualized_contents: List[str], theoretical_framework: Dict[str, str]) -> Dict[str, Any]:
        try:
            logger.debug("Starting enhanced quotation selection and analysis process.")
            
            # Generate the prompt
            prompt = self.create_prompt(
                research_objectives,
                transcript_chunk,
                contextualized_contents,
                theoretical_framework
            )
            
            # Generate response
            response = self.language_model.generate(
                prompt=prompt,
                max_tokens=6000,
                temperature=0.5
            ).strip()
            
            # Parse the complete response
            parsed_response = self.parse_response(response)
            
            # Extract components
            quotations = parsed_response.get("quotations", [])
            analysis = parsed_response.get("analysis", {})
            
            # Apply assertions
            assert_relevant_quotations(quotations, research_objectives)
            assert_confidentiality(quotations, sensitive_keywords=['confidential', 'secret'])
            assert_diversity_of_quotations(quotations, min_participants=3)
            assert_contextual_adequacy(quotations, [transcript_chunk] + contextualized_contents)
            assert_philosophical_alignment(quotations, theoretical_framework)
            
            # Additional assertions for theoretical analysis
            patterns = analysis.get("patterns_identified", [])
            theoretical_interpretation = analysis.get("theoretical_interpretation", "")
            research_alignment = analysis.get("methodological_reflection", {}).get("theoretical_alignment", "")
            
            assert_patterns_identified(patterns)
            assert_theoretical_interpretation(theoretical_interpretation)
            assert_research_alignment(research_alignment)
            
            logger.info(f"Successfully completed analysis with {len(quotations)} quotations.")
            return parsed_response

        except AssertionError as af:
            logger.warning(f"Assertion failed during analysis: {af}")
            return self.handle_failed_assertion(af, research_objectives, transcript_chunk, 
                                                 contextualized_contents, theoretical_framework)
        except Exception as e:
            logger.error(f"Error in EnhancedQuotationSignature.forward: {e}", exc_info=True)
            return {}

    def handle_failed_assertion(self, assertion_failure: AssertionError,
                                research_objectives: str, transcript_chunk: str,
                                contextualized_contents: List[str],
                                theoretical_framework: Dict[str, str]) -> Dict[str, Any]:
        """Handles failed assertions by attempting to generate improved analysis."""
        try:
            focused_prompt = self.create_prompt(
                research_objectives,
                transcript_chunk,
                contextualized_contents,
                theoretical_framework
            )
            
            focused_prompt += (
                f"\n\nThe previous attempt failed because: {assertion_failure}\n"
                f"Please ensure that your analysis addresses this specific issue while maintaining "
                f"all other requirements for thorough theoretical analysis."
            )

            response = self.language_model.generate(
                prompt=focused_prompt,
                max_tokens=2000,
                temperature=0.5
            ).strip()

            parsed_response = self.parse_response(response)
            
            # Re-apply all assertions
            quotations = parsed_response.get("quotations", [])
            analysis = parsed_response.get("analysis", {})
            
            assert_relevant_quotations(quotations, research_objectives)
            assert_confidentiality(quotations, sensitive_keywords=['confidential', 'secret'])
            assert_diversity_of_quotations(quotations, min_participants=3)
            assert_contextual_adequacy(quotations, [transcript_chunk] + contextualized_contents)
            assert_philosophical_alignment(quotations, theoretical_framework)
            
            patterns = analysis.get("patterns_identified", [])
            theoretical_interpretation = analysis.get("theoretical_interpretation", "")
            research_alignment = analysis.get("methodological_reflection", {}).get("theoretical_alignment", "")
            
            assert_patterns_identified(patterns)
            assert_theoretical_interpretation(theoretical_interpretation)
            assert_research_alignment(research_alignment)

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
