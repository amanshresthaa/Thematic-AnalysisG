#theme_development.py
import logging
from typing import Dict, Any, List
import dspy
import json
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
    Signature updated to not require quotation and keywords for theme development.
    Required fields now:
    - research_objectives (str)
    - codes (List[Dict[str, Any]])
    - theoretical_framework (Dict[str, str])
    """
    
    research_objectives: str = dspy.InputField(
        desc="A statement of the research aims and guiding questions."
    )

    # Quotation and Keywords removed

    codes: List[Dict[str, Any]] = dspy.InputField(
        desc="A list of codes derived from previous analysis steps."
    )

    theoretical_framework: Dict[str, str] = dspy.InputField(
        desc="The theoretical framework guiding the analysis."
    )

    themes: List[Dict[str, Any]] = dspy.OutputField(
        desc="A list of identified themes with details."
    )

    analysis: Dict[str, Any] = dspy.OutputField(
        desc="An analysis of the theme development process."
    )

    def create_prompt(self, research_objectives: str, codes: List[Dict[str, Any]],
                      theoretical_framework: Dict[str, str]) -> str:
        codes_formatted = "\n".join([
            f"- Code: {code.get('code', 'N/A')}\n  Definition: {code.get('definition', 'N/A')}\n"
            f"  6Rs_framework: {code.get('6Rs_framework', [])}\n"
            for code in codes
        ])

        theory = theoretical_framework.get("theory", "N/A")
        philosophical_approach = theoretical_framework.get("philosophical_approach", "N/A")
        rationale = theoretical_framework.get("rationale", "N/A")

        prompt = (
            f"You are a qualitative researcher skilled in thematic analysis. You have a set of codes.\n"
            f"Your goal is to synthesize these codes into higher-level themes that capture deeper patterns.\n\n"

            f"**Research Objectives:**\n{research_objectives}\n\n"
            f"**Theoretical Framework:**\n"
            f"- Theory: {theory}\n"
            f"- Philosophical Approach: {philosophical_approach}\n"
            f"- Rationale: {rationale}\n\n"

            f"**Codes Derived:**\n{codes_formatted}\n\n"

            f"**Instructions:**\n"
            f"- Identify overarching themes from the codes.\n"
            f"- Integrate multiple codes into each theme.\n"
            f"- Evaluate each theme (integration, abstraction, coherence, alignment).\n"
            f"- Discuss how each theme fits into the theoretical framework.\n"
            f"- Present the response as a JSON object in ```json``` code block.\n\n"

            f"Your final output should contain:\n"
            f" - 'themes': array of themes (name, description, associated_codes, dimensions_evaluation, theoretical_integration)\n"
            f" - 'analysis': object with methodological_reflection, alignment_with_research_objectives, future_implications."
        )
        return prompt

    def parse_response(self, response: str) -> Dict[str, Any]:
        try:
            import re
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
            if not json_match:
                logger.error("No valid JSON found in the response.")
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

    def forward(self, research_objectives: str, codes: List[Dict[str, Any]], 
                theoretical_framework: Dict[str, str]) -> Dict[str, Any]:
        for attempt in range(3):
            try:
                logger.debug(f"Attempt {attempt + 1} - Initiating theme development analysis.")
                prompt = self.create_prompt(
                    research_objectives,
                    codes,
                    theoretical_framework
                )

                response = self.language_model.generate(
                    prompt=prompt,
                    max_tokens=3000,
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
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} - Error in ThemedevelopmentAnalysisSignature.forward: {e}", exc_info=True)

        logger.error("Failed to develop valid themes after 3 attempts.")
        return {
            "error": "Failed to develop valid themes after multiple attempts. Please review inputs and prompt."
        }
