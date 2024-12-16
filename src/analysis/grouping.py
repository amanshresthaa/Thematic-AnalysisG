# src/analysis/grouping.py

import logging
from typing import Dict, Any, List
import dspy
from dataclasses import dataclass
import json
import re

logger = logging.getLogger(__name__)

@dataclass
class GroupingAnalysisSignature(dspy.Signature):
    """
    A signature for grouping similar codes into potential themes based 
    on research objectives and a theoretical framework.
    """

    # Input Fields
    research_objectives: str = dspy.InputField(
        desc="The specific goals and research questions guiding the analysis."
    )

    theoretical_framework: Dict[str, str] = dspy.InputField(
        desc="Detailed information about the theoretical foundation supporting the analysis."
    )

    codes: List[Dict[str, Any]] = dspy.InputField(
        desc="""
        A structured collection of developed codes, each accompanied by its definition. 
        Each code entry includes:
         - code: The name or label of the developed code.
         - definition: A precise and clear explanation of the code's meaning and scope.
        """
    )

    # Output Fields
    groupings: List[Dict[str, Any]] = dspy.OutputField(
        desc="""
        A structured collection of code groupings, representing potential themes. Each grouping entry includes:
         - group_label: A tentative label that reflects the shared meaning of the grouped codes.
         - codes: A list of codes belonging to this grouping.
         - rationale: A brief explanation for grouping these codes together. 
        """
    )

    def create_prompt(
        self,
        research_objectives: str,
        theoretical_framework: Dict[str, str],
        codes: List[Dict[str, Any]]
    ) -> str:
        """
        Creates a prompt to be given to the language model, instructing it to group the provided codes into potential themes.
        """

        # Extract theory details for context
        theory = theoretical_framework.get("theory", "N/A")
        philosophical_approach = theoretical_framework.get("philosophical_approach", "N/A")
        rationale = theoretical_framework.get("rationale", "N/A")

        # Format codes for display
        formatted_codes = "\n".join([
            f"- **{c['code']}**: {c['definition']}"
            for c in codes
        ])

        # Instructions for the LLM
        prompt = (
            f"You are an expert qualitative researcher specializing in thematic analysis. "
            f"You have a set of codes derived from prior coding stages. Your task is to group these codes into "
            f"higher-level themes that reflect shared meanings or patterns. The groupings should align with "
            f"the given research objectives and theoretical framework.\n\n"

            f"**Research Objectives:**\n{research_objectives}\n\n"

            f"**Theoretical Framework:**\n"
            f"- Theory: {theory}\n"
            f"- Philosophical Approach: {philosophical_approach}\n"
            f"- Rationale: {rationale}\n\n"

            f"**Codes to be Grouped:**\n{formatted_codes}\n\n"

            f"Your response should identify meaningful themes or groups that these codes can be organized into. "
            f"For each grouping, provide:\n"
            f"- **group_label**: A tentative label describing the shared meaning of the grouped codes.\n"
            f"- **codes**: A list of the code labels in that grouping.\n"
            f"- **rationale**: A brief explanation of why these codes were grouped together.\n\n"

            f"Return your final answer in JSON format inside triple backticks, e.g.:\n"
            f"```json\n{{\n  \"groupings\": [\n    {{\n      \"group_label\": \"Label\",\n      \"codes\": [\"Code A\", \"Code B\"],\n      \"rationale\": \"Explanation.\"\n    }}\n  ]\n}}\n```\n"
        )
        return prompt

    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Extracts and parses the JSON content from the language model's response.
        """
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

    def forward(
        self,
        research_objectives: str,
        theoretical_framework: Dict[str, str],
        codes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Executes the grouping analysis by generating a prompt and sending it to the language model.
        Tries multiple times in case the response is not valid.
        """
        prompt = self.create_prompt(
            research_objectives=research_objectives,
            theoretical_framework=theoretical_framework,
            codes=codes
        )

        for attempt in range(3):
            try:
                response = self.language_model.generate(
                    prompt=prompt,
                    max_tokens=2000,
                    temperature=0.5
                ).strip()

                parsed_response = self.parse_response(response)

                if not parsed_response or 'groupings' not in parsed_response:
                    raise ValueError("Parsed response is empty or missing 'groupings'.")

                groupings = parsed_response.get("groupings", [])
                if not groupings:
                    raise ValueError("No groupings were generated. Check the prompt and input data.")

                logger.info(f"Successfully generated {len(groupings)} groupings.")
                return parsed_response

            except ValueError as ve:
                logger.warning(f"Attempt {attempt + 1} - ValueError: {ve}")
                logger.debug(f"Response causing ValueError: {response}")
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} - Error in GroupingAnalysisSignature.forward: {e}", exc_info=True)

        logger.error("Failed to produce valid groupings after 3 attempts.")
        return {
            "error": "Failed to develop valid groupings after multiple attempts. Please review the input data and prompt."
        }
