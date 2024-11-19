import logging
from typing import List, Dict, Any
import dspy
import json

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
                f"        \"quotation\": \"This is the first quotation.\",\n"
                f"        \"type\": \"Descriptive\",\n"
                f"        \"function\": \"Evidence\"\n"
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
            quotations = self._standardize_quotations(parsed_response.get("quotations", []))
            purpose = parsed_response.get("purpose", "")

            # Validate quotations
            self._validate_quotations(quotations, research_objectives, transcript_chunks)

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

    def _standardize_quotations(self, quotations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Standardize quotation format to ensure consistency."""
        standardized = []
        for q in quotations:
            standardized_quote = {}
            # Ensure "quotation" is the standard key
            if "quote" in q:
                standardized_quote["quotation"] = q.pop("quote")
            elif "quotation" in q:
                standardized_quote["quotation"] = q["quotation"]
            else:
                logger.warning("Quotation missing required text field")
                continue

            # Copy other fields
            standardized_quote["type"] = q.get("type", "")
            standardized_quote["function"] = q.get("function", "")
            
            standardized.append(standardized_quote)
        return standardized

    def _validate_quotations(self, quotations: List[Dict[str, Any]], research_objectives: str, context: List[str]) -> None:
        """Validate quotations meet quality standards."""
        if not quotations:
            return

        context_text = " ".join(context)
        
        for q in quotations:
            # Validate quotation exists in context
            if q["quotation"] not in context_text:
                raise ValueError(f"Quotation not found in original context: {q['quotation'][:50]}...")
            
            # Validate required fields
            required_fields = ["quotation", "type", "function"]
            missing_fields = [field for field in required_fields if not q.get(field)]
            if missing_fields:
                raise ValueError(f"Quotation missing required fields: {missing_fields}")
            
            # Validate quotation length
            if len(q["quotation"]) < 10:
                raise ValueError(f"Quotation too short: {q['quotation']}")

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        try:
            # Attempt to parse the JSON array of quotations
            start_index = response.find('[')
            end_index = response.rfind(']') + 1
            if start_index == -1 or end_index == 0:
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