# extract_keyword.py

import logging
from typing import Dict, Any, List
import dspy
from dataclasses import dataclass

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
        - context:
            * surrounding_text: Text surrounding the keyword
            * usage_pattern: How the keyword is used in the quotation
            * source_location: Where in the quotation the keyword was found
        - analysis_value:
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
            f"within its broader context and theoretical framework."
        )
        return prompt

    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parses the response from the language model into a structured dictionary.
        This method needs to be implemented based on the expected response format.
        """
        # Placeholder implementation. Adjust based on actual response structure.
        try:
            parsed = eval(response)  # Caution: Using eval can be dangerous. Use a safe parser instead.
            return parsed
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return {}

    def validate_keywords(self, keywords: List[Dict[str, Any]], quotation: str) -> None:
        """Validates extracted keywords against the quotation and 6Rs framework."""
        if not keywords:
            raise AssertionError("No keywords extracted from quotation")
            
        for keyword in keywords:
            # Ensure keyword exists in quotation
            if keyword["keyword"].lower() not in quotation.lower():
                raise AssertionError(f"Keyword '{keyword['keyword']}' not found in quotation")

            # Validate 6Rs analysis
            analysis_value = keyword.get("analysis_value", {})
            for dimension in ["realness", "richness", "repetition", "rationale", "repartee", "regal"]:
                if not analysis_value.get(dimension):
                    raise AssertionError(f"Missing {dimension} analysis for keyword: {keyword['keyword']}")

    def forward(self, research_objectives: str, quotation: str,
                contextualized_contents: List[str], theoretical_framework: Dict[str, str]) -> Dict[str, Any]:
        """Executes keyword extraction with retry mechanism."""
        for attempt in range(3):
            try:
                logger.debug(f"Attempt {attempt + 1} - Starting keyword extraction from quotation")
                
                prompt = self.create_prompt(
                    research_objectives,
                    quotation,
                    contextualized_contents,
                    theoretical_framework
                )
                
                response = self.language_model.generate(
                    prompt=prompt,
                    max_tokens=3000,
                    temperature=0.7
                ).strip()
                
                parsed_response = self.parse_response(response)
                
                if not parsed_response:
                    raise ValueError("Failed to parse response")
                
                # Validate extracted keywords
                keywords = parsed_response.get("keywords", [])
                self.validate_keywords(keywords, quotation)
                
                logger.info(f"Successfully extracted {len(keywords)} keywords from quotation")
                return parsed_response

            except AssertionError as ae:
                logger.warning(f"Attempt {attempt + 1} - Validation failed: {ae}")
                continue
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} - Error: {e}", exc_info=True)
                
        logger.error("Failed to generate valid output after multiple attempts")
        return {}
