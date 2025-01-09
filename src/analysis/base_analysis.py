#analysis/base_analysis.py
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
        # Base implementation is a no-op, subclasses should override as needed
        pass

    def handle_failed_validation(self, validation_error: Exception, *args, **kwargs) -> Dict[str, Any]:
        """
        Handles validation failures by attempting to refine the analysis.
        Should be implemented by subclasses for specific error handling.
        """
        logger.error(f"Validation failed: {validation_error}")
        return {}

    def preprocess_inputs(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Preprocesses input data before analysis.
        Should be implemented by subclasses for specific preprocessing needs.
        """
        return {}

    def postprocess_outputs(self, outputs: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:
        """
        Postprocesses analysis outputs before returning.
        Should be implemented by subclasses for specific postprocessing needs.
        """
        return outputs

    def execute_analysis(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Template method defining the overall analysis workflow.
        Subclasses should implement specific steps as needed.
        """
        try:
            # Preprocess inputs
            processed_inputs = self.preprocess_inputs(*args, **kwargs)
            
            # Create prompt
            prompt = self.create_prompt(**processed_inputs)
            
            # Generate response
            response = self.generate_response(prompt)
            
            # Parse response
            parsed_response = self.parse_response(response)
            
            # Validate results
            self.validate_codes(parsed_response.get("codes", []), *args, **kwargs)
            
            # Postprocess outputs
            final_output = self.postprocess_outputs(parsed_response, *args, **kwargs)
            
            return final_output
            
        except Exception as e:
            logger.error(f"Error in execute_analysis: {e}", exc_info=True)
            return {}

    def generate_response(self, prompt: str) -> str:
        """
        Abstract method to generate response from the language model.
        Should be implemented by subclasses based on their specific needs.
        """
        raise NotImplementedError("Subclasses must implement generate_response.")