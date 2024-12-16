# src/analysis/theme_development.py

import logging
from typing import Dict, Any, List
import dspy
import json
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ThemeDimensionsEvaluation:
    integrated: str
    abstracted: str
    coherent: str
    theoretically_aligned: str

class ThemedevelopmentAnalysisSignature(dspy.Signature):
    """
    Signature for elevating coded data into broader, more abstract themes.
    """

    # Input Fields
    research_objectives: str = dspy.InputField(
        desc=(
            "A statement of the research goals and questions. This guides the theme development by "
            "ensuring that all identified themes remain relevant and aligned with what the study "
            "aims to explore."
        )
    )

    quotation: str = dspy.InputField(
        desc=(
            "The original quotation from the data that was analyzed during the coding stage. "
            "Serves as a reference point for developing higher-level themes that go beyond this "
            "specific excerpt."
        )
    )

    keywords: List[str] = dspy.InputField(
        desc=(
            "Keywords extracted from the original quotation, providing anchors for thematic development. "
            "These terms help ensure that the emerging themes remain grounded in the data."
        )
    )

    codes: List[Dict[str, Any]] = dspy.InputField(
        desc=(
            "A set of previously developed and validated codes derived from the quotation. "
            "Each code contains a label, definition, and an evaluation along various dimensions. "
            "Themes should build upon these codes, grouping and abstracting them into higher-level concepts."
        )
    )

    theoretical_framework: Dict[str, str] = dspy.InputField(
        desc=(
            "The theoretical foundation guiding the analysis. Themes should reflect and engage with "
            "the specified theory, philosophical approach, and rationale. This ensures that the generated "
            "themes are not only data-driven but also theory-informed."
        )
    )

    transcript_chunk: str = dspy.InputField(
        desc=(
            "The original transcript chunk or contextual segment associated with the quotation. "
            "Provides a broader context for understanding how the identified codes and emerging themes "
            "fit into the larger narrative or dataset."
        )
    )

    # Output Fields
    themes: List[Dict[str, Any]] = dspy.OutputField(
        desc=(
            "A list of identified themes. Each theme includes:\n"
            " - **theme**: The name or label of the emerging theme.\n"
            " - **description**: A detailed explanation of the theme’s meaning and how it relates to the data.\n"
            " - **associated_codes**: Which codes contribute to this theme, demonstrating how multiple codes are synthesized.\n"
            " - **dimensions_evaluation**: An evaluation of how well the theme integrates, abstracts, and aligns theoretically.\n"
            " - **theoretical_integration**: How this theme fits within the specified theoretical framework.\n"
        )
    )

    analysis: Dict[str, Any] = dspy.OutputField(
        desc=(
            "An analysis of the theme development process, including:\n"
            " - **methodological_reflection**: Insights on the process of moving from codes to themes.\n"
            " - **alignment_with_research_objectives**: An evaluation of how well the themes address the original research aims.\n"
            " - **future_implications**: How these themes might inform subsequent steps in the research, "
            "further analysis, or practical applications."
        )
    )

    def create_prompt(self, research_objectives: str, quotation: str, keywords: List[str],
                      codes: List[Dict[str, Any]], theoretical_framework: Dict[str, str],
                      transcript_chunk: str) -> str:
        # Format keywords
        keywords_formatted = "\n".join([f"- {kw}" for kw in keywords])

        # Format codes for display
        codes_formatted = "\n".join([
            f"- Code: {code.get('code', 'N/A')}\n  Definition: {code.get('definition', 'N/A')}\n"
            f"  6Rs_framework: {code.get('6Rs_framework', [])}\n" 
            for code in codes
        ])

        theory = theoretical_framework.get("theory", "N/A")
        philosophical_approach = theoretical_framework.get("philosophical_approach", "N/A")
        rationale = theoretical_framework.get("rationale", "N/A")

        prompt = (
            f"You are a qualitative researcher skilled in thematic analysis. Your goal is to synthesize these codes "
            f"into higher-level themes that capture deeper patterns and theoretical insights.\n\n"

            f"**Quotation:**\n{quotation}\n\n"
            f"**Keywords:**\n{keywords_formatted}\n\n"
            f"**Codes Derived from Quotation:**\n{codes_formatted}\n\n"
            f"**Transcript Context:**\n{transcript_chunk}\n\n"
            f"**Research Objectives:**\n{research_objectives}\n\n"
            f"**Theoretical Framework:**\n"
            f"- Theory: {theory}\n"
            f"- Philosophical Approach: {philosophical_approach}\n"
            f"- Rationale: {rationale}\n\n"

            f"**Instructions:**\n"
            f"- Identify overarching themes that group and abstract the codes into more general concepts.\n"
            f"- Each theme should integrate multiple codes, showing how they collectively represent a larger concept.\n"
            f"- Evaluate each theme along dimensions of integration, abstraction, coherence, and theoretical alignment.\n"
            f"- Describe how each theme fits within the theoretical framework and addresses the research objectives.\n"
            f"- Present your response as a JSON object encapsulated in ```json``` code blocks.\n\n"

            f"Your final output should contain:\n"
            f" - A 'themes' array, with each theme having 'theme', 'description', 'associated_codes', "
            f"   'dimensions_evaluation', and 'theoretical_integration'.\n"
            f" - An 'analysis' object with 'methodological_reflection', 'alignment_with_research_objectives', "
            f"   and 'future_implications'."
        )
        return prompt

    def parse_response(self, response: str) -> Dict[str, Any]:
        try:
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
            if not json_match:
                logger.error("No valid JSON found in the response.")
                logger.debug(f"Full response: {response}")
                return {}
            json_string = json_match.group(1)
            response_json = json.loads(json_string)
            return response_json
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed: {e}. Response: {response}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error during response parsing: {e}")
            return {}

    def forward(self, research_objectives: str, quotation: str, keywords: List[str], 
                codes: List[Dict[str, Any]], theoretical_framework: Dict[str, str], 
                transcript_chunk: str) -> Dict[str, Any]:
        # Attempt up to 3 times to get a valid response
        for attempt in range(3):
            try:
                logger.debug(f"Theme development attempt {attempt + 1}")
                prompt = self.create_prompt(
                    research_objectives,
                    quotation,
                    keywords,
                    codes,
                    theoretical_framework,
                    transcript_chunk
                )

                response = self.language_model.generate(
                    prompt=prompt,
                    max_tokens=8000,
                    temperature=0.5
                ).strip()

                logger.debug(f"Attempt {attempt + 1} - Response received from language model.")

                parsed_response = self.parse_response(response)

                if not parsed_response:
                    raise ValueError("Parsed response is empty or invalid.")
                if not parsed_response.get("themes"):
                    raise ValueError("No themes generated. Check inputs or prompt.")

                logger.info(f"Attempt {attempt + 1} - Successfully developed {len(parsed_response.get('themes', []))} themes.")
                return parsed_response

            except ValueError as ve:
                logger.warning(f"Attempt {attempt + 1} - {ve}")
                logger.debug(f"Response: {response}")
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} - Error in ThemedevelopmentAnalysisSignature.forward: {e}", exc_info=True)

        logger.error("Failed to develop valid themes after 3 attempts.")
        return {
            "error": "Failed to develop valid themes after multiple attempts. Please review inputs and prompt."
        }
