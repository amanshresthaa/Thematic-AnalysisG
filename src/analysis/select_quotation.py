# analysis/select_quotation.py

import logging
from typing import List, Dict, Any
import dspy
import json

from src.utils.validation_functions import validate_relevance, validate_quality, validate_context_clarity

logger = logging.getLogger(__name__)

class SelectQuotationSignature(dspy.Signature):
    """
    Select relevant quotations from transcript chunks based on research objectives.
    """
    research_objectives: str = dspy.InputField(
        desc="The research objectives guiding the selection of quotations."
    )
    transcript_chunks: List[str] = dspy.InputField(
        desc="Chunks of transcript from which quotations are to be selected."
    )
    quotations: List[Dict[str, Any]] = dspy.OutputField(
        desc="List of selected quotations with their types and functions."
    )
    purpose: str = dspy.OutputField(
        desc="Purpose of selecting the quotations."
    )

    def forward(self, research_objectives: str, transcript_chunks: List[str]) -> Dict[str, Any]:
        try:
            logger.debug("Starting quotation selection process.")
            prompt = (
                f"You are an expert in qualitative research and thematic analysis.\n\n"
                f"**Research Objectives**:\n{research_objectives}\n\n"
                f"**Transcript Chunks**:\n" +
                "\n".join([f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(transcript_chunks)]) +
                "\n\n"
                f"**Task:** Extract **3-5** relevant quotations from the transcript chunks that align with the research objectives provided. "
                f"Provide each quotation in the following JSON format within a list:\n\n"
                f"```json\n"
                f"[\n"
                f"    {{\n"
                f"        \"quote\": \"This is the first quotation.\",\n"
                f"        \"category\": \"Descriptive\",\n"
                f"        \"sub_type\": \"Observation\",\n"
                f"        \"function\": \"Evidence\",\n"
                f"        \"thematic_code\": \"Theme1\",\n"
                f"        \"strength\": 0.85\n"
                f"    }}\n"
                f"]\n"
                f"```\n"
                f"Ensure that the response is a valid JSON array containing the quotations and the overall purpose. "
                f"If no quotations are available, respond with an empty list and an empty string."
            )

            response = self.language_model.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.7,
                top_p=0.9,
                n=1,
                stop=None
            ).strip()

            logger.info("Quotations selected successfully.")
            parsed_response = self._parse_json_response(response)
            quotations = parsed_response.get("quotations", [])
            purpose = parsed_response.get("purpose", "")

            # Assertions
            # a. Relevance to Research Objectives
            dspy.Assert(
                validate_relevance(quotations, research_objectives),
                msg="Quotations are not sufficiently relevant to the research objectives.",
                backtrack=self
            )

            # b. Quality and Representation
            dspy.Assert(
                validate_quality(quotations),
                msg="One or more quotations do not meet the quality standards.",
                backtrack=self
            )

            # f. Context and Clarity
            context = "\n".join(transcript_chunks)
            dspy.Assert(
                validate_context_clarity(quotations, context),
                msg="Quotations lack sufficient context or clarity.",
                backtrack=self
            )

            return {
                "quotations": quotations,
                "purpose": purpose
            }
        except Exception as e:
            logger.error(f"Error in SelectQuotationSignature.forward: {e}", exc_info=True)
            return {
                "quotations": [],
                "purpose": ""
            }

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        try:
            # Attempt to parse the JSON array of quotations
            start_index = response.find('[')
            end_index = response.rfind(']') + 1
            if start_index == -1 or end_index == -1:
                logger.warning("JSON array of quotations not found in response.")
                return {"quotations": [], "purpose": ""}
            quotations_json = response[start_index:end_index]
            quotations = json.loads(quotations_json)
            return {
                "quotations": quotations,
                "purpose": ""
            }
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response. Returning empty fields.")
            return {
                "quotations": [],
                "purpose": ""
            }
