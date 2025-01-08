# analysis/coding.py
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
from analysis.base_analysis import BaseAnalysisSignature

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

class CodingAnalysisSignature(BaseAnalysisSignature, dspy.Signature):
    """
    Signature for comprehensive thematic coding analysis utilizing the 6Rs framework.
    Inherits common behaviors from BaseAnalysisSignature.
    """

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
            "This quotation serves as the primary source text from which themes and codes will be derived."
        )
    )

    keywords: List[str] = dspy.InputField(
        desc="A curated list of keywords to guide the coding process."
    )

    contextualized_contents: List[str] = dspy.InputField(
        desc="Additional contextual information related to the quotation."
    )

    theoretical_framework: Dict[str, str] = dspy.InputField(
        desc="The foundational theoretical framework that underpins the analysis."
    )

    coding_info: Dict[str, Any] = dspy.OutputField(
        desc="Comprehensive context and metadata related to the coding analysis."
    )

    codes: List[Dict[str, Any]] = dspy.OutputField(
        desc="A structured collection of developed codes with in-depth analysis."
    )

    analysis: Dict[str, Any] = dspy.OutputField(
        desc="An extensive analysis of the coding process."
    )

    def create_prompt(self, research_objectives: str, quotation: str,
                      keywords: List[str], contextualized_contents: List[str],
                      theoretical_framework: Dict[str, str]) -> str:
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

    def validate_codes(self, codes: List[Dict[str, Any]], research_objectives: str,
                       theoretical_framework: Dict[str, str]) -> None:
        """
        Validates developed codes specific to coding analysis.
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
            for code in codes:
                try:
                    assert_code_relevance(code, research_objectives, theoretical_framework)
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
                prompt = self.create_prompt(
                    research_objectives,
                    quotation,
                    keywords,
                    contextualized_contents,
                    theoretical_framework
                )
                response = self.language_model.generate(
                    prompt=prompt,
                    max_tokens=8000,
                    temperature=0.5
                ).strip()
                logger.debug(f"Attempt {attempt + 1} - Response received from language model.")
                parsed_response = self.parse_response(response)

                if not parsed_response:
                    raise ValueError("Parsed response is empty or invalid JSON.")

                codes = parsed_response.get("codes", [])
                analysis = parsed_response.get("analysis", {})

                if not codes:
                    raise ValueError("No codes were generated. Please check the prompt and input data.")

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

        logger.error("Failed to develop valid codes after 3 attempts.")
        return {
            "error": "Failed to develop valid codes after 3 attempts. Please review the input data and prompt for possible improvements."
        }
