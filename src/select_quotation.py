# File: /Users/amankumarshrestha/Downloads/Thematic-AnalysisE/src/select_quotation.py

import logging
from typing import List, Dict, Any
import dspy
from utils.validation_functions import validate_relevance, validate_quality, validate_context_clarity

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
    quotations: List[str] = dspy.OutputField(
        desc="List of selected quotations relevant to the research objectives."
    )
    types_and_functions: List[str] = dspy.OutputField(
        desc="Types and functions of the selected quotations."
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
                f"    {{\"QUOTE\": \"This is the first quotation.\"}},\n"
                f"    {{\"QUOTE\": \"This is the second quotation.\"}},\n"
                f"    {{\"QUOTE\": \"This is the third quotation.\"}}\n"
                f"]\n"
                f"```\n"
                f"Additionally, categorize each quotation under appropriate types and functions, and state the purpose of selecting these quotations.\n"
                f"Ensure that the response is a valid JSON object containing the quotations, their types and functions, and the overall purpose. "
                f"If no quotations are available, respond with empty lists and an empty string."
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
            types_and_functions = parsed_response.get("types_and_functions", [])
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
                "types_and_functions": types_and_functions,
                "purpose": purpose
            }
        except Exception as e:
            logger.error(f"Error in SelectQuotationSignature.forward: {e}", exc_info=True)
            return {
                "quotations": [],
                "types_and_functions": [],
                "purpose": ""
            }

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """
        Parses the JSON response to extract quotations, types/functions, and purpose.
        """
        import json
        try:
            parsed = json.loads(response)
            return parsed
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response. Returning empty fields.")
            return {
                "quotations": [],
                "types_and_functions": [],
                "purpose": ""
            }
