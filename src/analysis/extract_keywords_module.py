# analysis/extract_keywords_module.py

import logging
from typing import Dict, Any, List
import dspy
import os

from .extract_keywords import KeywordExtractionSignature, load_quotations, save_keywords

logger = logging.getLogger(__name__)

class KeywordExtractionModule(dspy.Module):
    """
    DSPy module to extract keywords from quotations.
    """
    def __init__(self, input_file: str = "data/quotation.json", output_file: str = "data/keywords.json"):
        super().__init__()
        self.chain = dspy.TypedChainOfThought(KeywordExtractionSignature)
        self.input_file = input_file
        self.output_file = output_file

    def process_file(self, input_file: str, research_objectives: str) -> Dict[str, Any]:
        """Process a single input file and extract keywords for each quote."""
        try:
            logger.debug(f"Processing file: {input_file}")
            
            # Load quotations from input file
            quotations = load_quotations(input_file)
            if not quotations:
                logger.warning(f"No quotations found in {input_file}")
                return {"keywords": []}

            keywords_mapping = []
            for idx, quote_dict in enumerate(quotations):
                quote = quote_dict.get("quote", "")
                if not quote:
                    logger.warning(f"Quote at index {idx} is empty.")
                    keywords_mapping.append({"quote": quote, "keywords": []})
                    continue

                logger.debug(f"Extracting keywords for quote {idx+1}: {quote}")
                # Extract keywords for the current quote
                response = self.chain(
                    research_objectives=research_objectives,
                    quotations=[quote_dict]
                )
                keywords = response.get("keywords", [])

                keywords_mapping.append({"quote": quote, "keywords": keywords})

            # Save the mapping of quotes to keywords
            save_keywords(keywords_mapping, self.output_file)
            
            return {"keywords": keywords_mapping, "output_file": self.output_file}
            
        except Exception as e:
            logger.error(f"Error processing file {input_file}: {e}", exc_info=True)
            return {"keywords": []}
