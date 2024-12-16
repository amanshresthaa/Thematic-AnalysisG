#analysis/coding.py
import logging
from typing import Dict, Any, List
import dspy
from dataclasses import dataclass
import json
from src.assertions_coding import (
    assert_robustness,
    assert_reflectiveness,
    assert_resplendence,
    assert_relevance,
    assert_radicality,
    assert_righteousness,
    assert_code_representation,
    assert_code_specificity,
    assert_code_relevance,
    assert_code_distinctiveness,
    run_all_coding_assertions
)

logger = logging.getLogger(__name__)

@dataclass
class SixRsEvaluation:
    """Evaluation metrics for each dimension of the 6Rs framework."""
    robust: str
    reflective: str
    resplendent: str
    relevant: str
    radical: str
    righteous: str

class CodingAnalysisSignature(dspy.Signature):
    """
    Signature for conducting a comprehensive thematic coding analysis utilizing the 6Rs framework.
    
    This signature facilitates the systematic analysis of qualitative data by guiding the user
    through the process of thematic coding, ensuring methodological rigor and theoretical alignment.
    """

    # Input Fields
    research_objectives: str = dspy.InputField(
        desc=(
            "A detailed statement of the study's overarching goals and specific research questions "
            "that direct the coding analysis. This should clearly articulate what the research aims to "
            "accomplish and the key questions it seeks to address, providing a foundation for the "
            "subsequent coding process."
        )
    )

    quotation: str = dspy.InputField(
        desc=(
            "The specific excerpt, passage, or segment selected from the data set for coding analysis. "
            "This quotation serves as the primary source text from which themes and codes will be derived, "
            "capturing the essential content to be analyzed."
        )
    )

    keywords: List[str] = dspy.InputField(
        desc=(
            "A curated list of keywords previously identified to inform and guide the coding process. "
            "Each keyword should be a distinct string that represents a significant theme or concept "
            "relevant to the research objectives, aiding in the systematic identification of patterns "
            "within the quotation."
        )
    )

    contextualized_contents: List[str] = dspy.InputField(
        desc=(
            "Additional contextual information that provides background and deeper insight into the "
            "primary quotation. This may include related texts, situational context, historical background, "
            "or any other supplementary content that enhances the understanding and interpretation of the "
            "quotation being analyzed."
        )
    )

    theoretical_framework: Dict[str, str] = dspy.InputField(
        desc=(
            "The foundational theoretical framework that underpins the analysis, detailing the guiding "
            "theoretical approach. This dictionary should include:\n"
            " - **theory**: The primary theoretical perspective or model being applied to the analysis.\n"
            " - **philosophical_approach**: The underlying philosophical stance that informs the theoretical "
            "framework, such as positivism, interpretivism, critical theory, etc.\n"
            " - **rationale**: A comprehensive justification for selecting this particular theoretical approach, "
            "explaining how it aligns with and supports the research objectives and questions."
        )
    )

    # Output Fields
    coding_info: Dict[str, Any] = dspy.OutputField(
        desc=(
            "Comprehensive context and metadata related to the coding analysis, including:\n"
            " - **quotation**: The original passage selected for analysis.\n"
            " - **research_objectives**: The specific goals and research questions that guide the analysis.\n"
            " - **theoretical_framework**: Detailed information about the theoretical foundation supporting the analysis.\n"
            " - **keywords**: The list of keywords extracted or utilized from the quotation to inform coding."
        )
    )

    codes: List[Dict[str, Any]] = dspy.OutputField(
        desc=(
            "A structured collection of developed codes, each accompanied by an in-depth analysis. Each code entry includes:\n"
            " - **code**: The name or label of the developed code.\n"
            " - **definition**: A precise and clear explanation of the code's meaning and scope.\n"
            " - **6Rs_framework**: The specific dimensions of the 6Rs framework (robust, reflective, resplendent, relevant, radical, righteous) that the code satisfies. Only the dimensions that apply are listed (e.g., ['robust', 'relevant']).\n"
            " - **6Rs_evaluation**: Detailed justifications and evaluation metrics for the Rs listed in **6Rs_framework**, explaining how the code satisfies those dimensions. Evaluations focus only on these specific Rs, including:\n"
            "     * **robust**: Assessment of how effectively the code captures the fundamental essence of the data.\n"
            "     * **reflective**: Evaluation of the code's alignment and relationship with the theoretical framework.\n"
            "     * **resplendent**: Analysis of the code's comprehensiveness and the depth of understanding it provides.\n"
            "     * **relevant**: Determination of the code's appropriateness and contextual fit within the data.\n"
            "     * **radical**: Identification of the code's uniqueness and the novelty of the insights it offers.\n"
            "     * **righteous**: Verification of the code's logical consistency and alignment with the overarching theoretical framework."
        )
    )


    analysis: Dict[str, Any] = dspy.OutputField(
        desc=(
            "An extensive analysis of the coding process, encompassing:\n"
            " - **theoretical_integration**: An explanation of how the developed codes integrate with and apply the theoretical framework.\n"
            " - **methodological_reflection**: Critical reflections on the coding methodology, including:\n"
            "     * **code_robustness**: An evaluation of the strength, reliability, and consistency of the codes.\n"
            "     * **theoretical_alignment**: Analysis of the extent to which the codes align with the theoretical framework.\n"
            "     * **researcher_reflexivity**: Consideration of the researcher's own influence, biases, and assumptions in the coding process.\n"
            " - **practical_implications**: Insights derived from the coding analysis and their potential applications or implications for practice, policy, or further research."
        )
    )

    def create_prompt(self, research_objectives: str, quotation: str,
                     keywords: List[str], contextualized_contents: List[str],
                     theoretical_framework: Dict[str, str]) -> str:
        """Generates a detailed prompt for conducting enhanced coding analysis."""

        # Format keywords for clarity
        keywords_formatted = "\n".join([
            f"- **{kw}**"
            for kw in keywords
        ])

        # Format contextualized contents with clear labeling
        contents_formatted = "\n\n".join([
            f"**Content {i+1}:**\n{content}"
            for i, content in enumerate([quotation] + contextualized_contents)
        ])

        # Extract theoretical framework components with default empty strings
        theory = theoretical_framework.get("theory", "N/A")
        philosophical_approach = theoretical_framework.get("philosophical_approach", "N/A")
        rationale = theoretical_framework.get("rationale", "N/A")

        # Construct the prompt with clear sections and instructions
        prompt = (
            f"You are an experienced qualitative researcher specializing in thematic coding analysis "
            f"utilizing the 6Rs framework and grounded in {theory}. Your objective is to develop and critically analyze codes based on "
            f"the provided keywords and quotation, ensuring methodological rigor and theoretical alignment.\n\n"

            f"**Quotation for Analysis:**\n{quotation}\n\n"

            f"**Identified Keywords:**\n{keywords_formatted}\n\n"

            f"**Additional Contextualized Contents:**\n{contents_formatted}\n\n"

            f"**Research Objectives:**\n{research_objectives}\n\n"

            f"**Theoretical Framework:**\n"
            f"- **Theory:** {theory}\n"
            f"- **Philosophical Approach:** {philosophical_approach}\n"
            f"- **Rationale:** {rationale}\n\n"

            f"**Guidelines for Analysis:**\n"
            f"Your analysis should adhere to the 6Rs framework, addressing each dimension as follows:\n"
            f"1. **Robust:** Ensure that the code captures the true essence of the data in a theoretically sound manner.\n"
            f"2. **Reflective:** Demonstrate clear relationships between the data and the theoretical framework.\n"
            f"3. **Resplendent:** Provide a comprehensive understanding that encompasses all relevant aspects.\n"
            f"4. **Relevant:** Accurately represent the data, ensuring appropriateness and contextual fit.\n"
            f"5. **Radical:** Introduce unique and innovative insights that advance understanding.\n"
            f"6. **Righteous:** Maintain logical alignment with the overarching theoretical framework.\n\n"

            f"**Example Code Development:**\n"
            f"- **Code:** Economic Vulnerability\n"
            f"  - **Definition:** Victims originate from economically disadvantaged backgrounds, lacking financial stability.\n"
            f"  - **Keywords:** Poverty, Lack of Education\n"
            f"  - **6Rs Evaluation:** Robust, Relevant\n"
            f"  - **Theoretical Alignment:** Connects economic factors with victim vulnerability as per {theory}.\n"
            f"  - **Supporting Quotes:** [\"Victims of sex trafficking often come from vulnerable backgrounds, such as poverty...\"]\n"
            f"  - **Analytical Memos:** Economic hardship is a primary factor that increases susceptibility to trafficking offers.\n\n"

            f"**Instructions:**\n"
            f"- Each code should include its definition, associated keywords, 6Rs evaluation, theoretical alignment, supporting quotes, and analytical memos.\n"
            f"- Ensure that the codes are directly related to the theoretical framework and research objectives.\n"
            f"- Use the identified keywords as foundational themes for each code.\n"
            f"- Present the response in JSON format encapsulated within ```json``` blocks."
        )
        return prompt

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Extracts and parses the JSON content from the language model's response."""
        try:
            import re
            # Use regex to find JSON content within code blocks
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
            if not json_match:
                logger.error("No valid JSON found in the response.")
                logger.debug(f"Full response received: {response}")
                return {}
            json_string = json_match.group(1)

            # Parse the JSON string into a Python dictionary
            response_json = json.loads(json_string)
            return response_json
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed: {e}. Response: {response}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error during response parsing: {e}")
            return {}

    def validate_codes(self, codes: List[Dict[str, Any]], research_objectives: str,
                      theoretical_framework: Dict[str, str]) -> None:
        """
        Validates the developed codes against the 6Rs framework and other assertions.
        
        This method ensures that each code meets the defined quality standards and aligns
        with the research objectives and theoretical framework.

        Args:
            codes (List[Dict[str, Any]]): The list of developed codes with their metadata.
            research_objectives (str): The research goals and questions.
            theoretical_framework (Dict[str, str]): Details of the theoretical foundation.

        Raises:
            AssertionError: If any code fails to meet the validation criteria.
        """
        try:
            run_all_coding_assertions(
                codes=codes,
                research_objectives=research_objectives,
                theoretical_framework=theoretical_framework
            )
            logger.debug("All coding assertions passed successfully.")
        except AssertionError as ae:
            logger.error(f"Code validation failed: {ae}")
            # Additional logging to identify problematic codes
            for code in codes:
                try:
                    assert_code_relevance(code, research_objectives, theoretical_framework)
                    # Add other individual assertions as needed
                except AssertionError as individual_ae:
                    logger.error(f"Validation failed for code '{code.get('code', 'Unknown')}': {individual_ae}")
            raise

    def forward(self, research_objectives: str, quotation: str,
                keywords: List[str], contextualized_contents: List[str],
                theoretical_framework: Dict[str, str]) -> Dict[str, Any]:
        """Executes the coding analysis with retry mechanism."""
        for attempt in range(3):
            try:
                logger.debug(f"Attempt {attempt + 1} - Initiating coding analysis.")

                # Generate the prompt for the language model
                prompt = self.create_prompt(
                    research_objectives,
                    quotation,
                    keywords,
                    contextualized_contents,
                    theoretical_framework
                )

                # Interact with the language model to generate a response
                response = self.language_model.generate(
                    prompt=prompt,
                    max_tokens=8000,
                    temperature=0.5  # Adjusted for greater consistency
                ).strip()

                logger.debug(f"Attempt {attempt + 1} - Response received from language model.")

                # Parse the JSON response
                parsed_response = self.parse_response(response)

                if not parsed_response:
                    raise ValueError("Parsed response is empty or invalid JSON.")

                # Extract codes and analysis from the parsed response
                codes = parsed_response.get("codes", [])
                analysis = parsed_response.get("analysis", {})

                if not codes:
                    raise ValueError("No codes were generated. Please check the prompt and input data.")

                # Validate the developed codes
                self.validate_codes(
                    codes=codes,
                    research_objectives=research_objectives,
                    theoretical_framework=theoretical_framework
                )

                logger.info(f"Attempt {attempt + 1} - Successfully developed and validated {len(codes)} codes.")
                return parsed_response

            except AssertionError as ae:
                logger.warning(f"Attempt {attempt + 1} - Assertion failed during coding analysis: {ae}")
                logger.debug(f"Attempt {attempt + 1} - Response causing assertion failure: {response}")
            except ValueError as ve:
                logger.warning(f"Attempt {attempt + 1} - ValueError: {ve}")
                logger.debug(f"Attempt {attempt + 1} - Response causing ValueError: {response}")
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} - Error in CodingAnalysisSignature.forward: {e}", exc_info=True)

        logger.error(f"Failed to develop valid codes after 3 attempts. Last response: {response}")
        # Optionally, provide a summary of issues or next steps
        return {
            "error": "Failed to develop valid codes after 3 attempts. Please review the input data and prompt for possible improvements."
        }
