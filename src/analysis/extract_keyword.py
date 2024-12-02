# src/analysis/extract_keyword.py

import logging
from typing import Dict, Any, List
import dspy
from dataclasses import dataclass
import json
from src.assertions_keyword import (
    assert_keywords_not_exclusive_to_quotation,
    assert_keyword_specificity,
    assert_keyword_distinctiveness,
    assert_keyword_relevance
)

logger = logging.getLogger(__name__)

@dataclass
class KeywordAnalysisValue:
    """Analysis values for each of the 6Rs framework dimensions."""
    realness: str
    richness: str
    repetition: str
    rationale: str
    repartee: str
    regal: str

class KeywordExtractionSignature(dspy.Signature):
    """
    A comprehensive signature for conducting thematic keyword extraction from quotations,
    following advanced qualitative methodologies. This signature supports systematic keyword analysis
    with robust pattern recognition and theoretical integration, aligned with the 6Rs framework.
    """

    # Input Fields
    research_objectives: str = dspy.InputField(
        desc="Research goals and questions guiding the keyword analysis"
    )

    quotation: str = dspy.InputField(
        desc="Selected quotation for keyword extraction"
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
    quotation_info: Dict[str, Any] = dspy.OutputField(
        desc="""Comprehensive context information including:
        - quotation: Selected quotation content
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

    keywords: List[Dict[str, Any]] = dspy.OutputField(
        desc="""Extracted keywords with detailed analysis:
        - keyword: The keyword text
        - category: Classification of the keyword
        - 6Rs framework: List of R's keyword falls under, Example: [Realness, Richness]
        - analysis_value: (Only for R's keyword falls under)
            * realness: Reflection of genuine experiences
            * richness: Depth of meaning
            * repetition: Recurrence in quotation
            * rationale: Connection to theoretical foundations
            * repartee: Contribution to discussions
            * regal: Centrality to the topic"""
    )

    analysis: Dict[str, Any] = dspy.OutputField(
        desc="""Comprehensive keyword analysis including:
        - patterns_identified: Key patterns discovered
        - theoretical_interpretation: Framework application
        - methodological_reflection:
            * pattern_robustness
            * theoretical_alignment
            * researcher_reflexivity
        - practical_implications: Applied insights"""
    )

    def create_prompt(self, research_objectives: str, quotation: str,
                     contextualized_contents: List[str], theoretical_framework: Dict[str, str]) -> str:
        """Creates the prompt for enhanced keyword extraction from quotation."""

        chunks_formatted = "\n\n".join([
            f"Content {i+1}:\n{content}"
            for i, content in enumerate([quotation] + contextualized_contents)
        ])

        theory = theoretical_framework.get("theory", "")
        philosophical_approach = theoretical_framework.get("philosophical_approach", "")
        rationale = theoretical_framework.get("rationale", "")

        prompt = (
            f"You are an experienced qualitative researcher conducting keyword analysis using "
            f"the 6Rs methodology. Your task is to identify and analyze meaningful keywords "
            f"from the following quotation while maintaining methodological rigor.\n\n"

            f"Quotation for Analysis:\n"
            f"{quotation}\n\n"

            f"Additional Context:\n"
            f"{chunks_formatted}\n\n"

            f"Research Objectives:\n"
            f"{research_objectives}\n\n"

            f"Theoretical Framework:\n"
            f"Theory: {theory}\n"
            f"Philosophical Approach: {philosophical_approach}\n"
            f"Rationale: {rationale}\n\n"

            f"Your analysis should follow the 6Rs framework:\n"
            f"1. Realness: Select words reflecting genuine experiences\n"
            f"2. Richness: Identify words with deep meaning\n"
            f"3. Repetition: Note recurring patterns\n"
            f"4. Rationale: Connect to theoretical foundations\n"
            f"5. Repartee: Consider discussion value\n"
            f"6. Regal: Focus on centrality to topic\n\n"

            f"Consider how each keyword contributes to understanding the quotation "
            f"within its broader context and theoretical framework.\n\n"

            f"Your final output should follow this JSON structure:\n\n"

            "{\n"
            "  \"quotation_info\": {\n"
            "    \"quotation\": \"\",                      // Selected quotation content\n"
            "    \"research_objectives\": \"\",             // Analysis goals\n"
            "    \"theoretical_framework\": {\n"
            "      \"theory\": \"\",                        // Primary theoretical approach\n"
            "      \"philosophical_approach\": \"\",        // Philosophical foundation\n"
            "      \"rationale\": \"\"                      // Justification for approach\n"
            "    }\n"
            "  },\n"
            "  \"retrieved_chunks\": [],                    // List of retrieved chunks (if needed)\n"
            "  \"retrieved_chunks_count\": 0,               // Count of retrieved chunks\n"
            "  \"used_chunk_ids\": [],                      // List of used chunk IDs\n"
            "  \"keywords\": [\n"
            "    {\n"
            "      \"keyword\": \"\",                       // The keyword text\n"
            "      \"category\": \"\",                      // Classification of the keyword\n"
            "      \"6Rs_framework\": [],                   // List of R's keyword falls under\n"
            "      },\n"
            "      \"analysis_value\": {\n"
            "        \"realness\": \"\",                     // Reflection of genuine experiences\n"
            "        \"richness\": \"\",                     // Depth of meaning\n"
            "        \"repetition\": \"\",                   // Recurrence in quotation\n"
            "        \"rationale\": \"\",                    // Connection to theoretical foundations\n"
            "        \"repartee\": \"\",                     // Contribution to discussions\n"
            "        \"regal\": \"\"                         // Centrality to the topic\n"
            "      }\n"
            "    }\n"
            "  ],\n"
            "  \"analysis\": {\n"
            "    \"patterns_identified\": [\"\"],             // Key patterns found\n"
            "    \"theoretical_interpretation\": \"\",        // Framework application\n"
            "    \"methodological_reflection\": {\n"
            "      \"pattern_robustness\": \"\",              // Pattern evidence\n"
            "      \"theoretical_alignment\": \"\",           // Framework fit\n"
            "      \"researcher_reflexivity\": \"\"           // Interpretation awareness\n"
            "    },\n"
            "    \"practical_implications\": \"\"             // Applied insights\n"
            "  }\n"
            "}\n\n"

            f"**Important Instructions:**\n"
            f"- **Your final output must strictly follow the JSON structure provided above, including all fields exactly as specified, even if some fields are empty. Do not omit any fields.**\n"
            f"- **Use double quotes for all strings.**\n"
            f"- **Do not include any additional commentary or text outside of the JSON structure.**\n\n"

            f"Remember to wrap your analysis process in <analysis_process> tags throughout your analysis to show your chain of thought before providing the final JSON output.\n\n"
        )
        return prompt

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Parses the complete response from the language model."""
        try:
            # Use regex to extract JSON content within a code block tagged as json
            import re
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

    def validate_keywords(self, keywords: List[Dict[str, Any]], quotation: str,
                         contextualized_contents: List[str], research_objectives: str,
                         theoretical_framework: Dict[str, str]) -> None:
        """Validates extracted keywords against the assertions."""
        try:
            assert_keywords_not_exclusive_to_quotation(keywords, quotation, contextualized_contents)
            assert_keyword_specificity(keywords, theoretical_framework)
            assert_keyword_relevance(keywords, research_objectives, contextualized_contents)
            assert_keyword_distinctiveness(keywords)
            logger.debug("All keyword assertions passed successfully.")
        except AssertionError as ae:
            logger.error(f"Keyword validation failed: {ae}")
            raise

    def forward(self, research_objectives: str, quotation: str,
                contextualized_contents: List[str], theoretical_framework: Dict[str, str]) -> Dict[str, Any]:
        """Executes keyword extraction with retry mechanism."""
        for attempt in range(3):
            try:
                logger.debug(f"Attempt {attempt + 1} - Starting keyword extraction from quotation")

                # Generate the prompt
                prompt = self.create_prompt(
                    research_objectives,
                    quotation,
                    contextualized_contents,
                    theoretical_framework
                )

                # Generate response from the language model
                response = self.language_model.generate(
                    prompt=prompt,
                    max_tokens=3000,
                    temperature=0.7
                ).strip()

                logger.debug(f"Attempt {attempt + 1} - Response received from language model.")

                # Parse the complete response
                parsed_response = self.parse_response(response)

                if not parsed_response:
                    raise ValueError("Parsed response is empty. Possibly invalid JSON.")

                # Extract components
                keywords = parsed_response.get("keywords", [])
                analysis = parsed_response.get("analysis", {})

                # Validate extracted keywords
                self.validate_keywords(
                    keywords=keywords,
                    quotation=quotation,
                    contextualized_contents=contextualized_contents,
                    research_objectives=research_objectives,
                    theoretical_framework=theoretical_framework
                )

                logger.info(f"Attempt {attempt + 1} - Successfully extracted and validated {len(keywords)} keywords.")
                return parsed_response

            except AssertionError as ae:
                logger.warning(f"Attempt {attempt + 1} - Assertion failed during keyword extraction: {ae}")
                logger.debug(f"Attempt {attempt + 1} - Response causing assertion failure: {response}")
                # Optionally, refine the prompt or handle the failure as needed
                # For simplicity, we'll retry up to 3 times
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} - Error in KeywordExtractionSignature.forward: {e}", exc_info=True)
                # Continue to next attempt

        logger.error(f"Failed to extract valid keywords after {3} attempts. Last response: {response}")
        return {}
