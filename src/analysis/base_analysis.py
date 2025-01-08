# analysis/base_analysis.py
import json
import logging
import re
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class BaseAnalysisSignature:
    """
    Abstract base class for thematic analysis signatures.
    Provides common functionality for creating prompts, parsing responses,
    and validating codes, which can be extended by concrete analysis implementations.
    """
    def create_prompt(self, *args, **kwargs) -> str:
        """Abstract method to generate a prompt; to be implemented by subclass."""
        raise NotImplementedError("Subclasses must implement create_prompt.")

    def parse_response(self, response: str) -> Dict[str, Any]:
        """Extracts and parses the JSON content from the language model's response."""
        try:
            json_match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
            if not json_match:
                logger.error("No valid JSON found in the response.")
                logger.debug(f"Full response received: {response}")
                return {}
            json_string = json_match.group(1)
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decoding failed: {e}. Response: {response}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error during response parsing: {e}")
            return {}

    def validate_codes(self, codes: List[Dict[str, Any]], *args, **kwargs) -> None:
        """
        Validates the developed codes against various assertions.
        Should be extended if specific validation is needed.
        """
        # Base implementation can be a no-op or common validation.
        pass
