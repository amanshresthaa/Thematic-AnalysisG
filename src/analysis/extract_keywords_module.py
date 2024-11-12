# src/analysis/extract_keywords_module.py
import logging
from typing import Dict, Any, List
import dspy
import os
from datetime import datetime

from .extract_keywords import KeywordExtractionSignature, load_quotations, save_keywords  # Updated relative import

# Rest of the code remains the same
logger = logging.getLogger(__name__)

class KeywordExtractionModule(dspy.Module):
    """
    DSPy module to extract keywords from quotations.
    """
    def __init__(self, input_dir: str = "data/quotations", output_dir: str = "data/keywords"):
        super().__init__()
        self.chain = dspy.TypedChainOfThought(KeywordExtractionSignature)
        self.input_dir = input_dir
        self.output_dir = output_dir

    def process_file(self, input_file: str, research_objectives: str) -> Dict[str, Any]:
        """Process a single input file and extract keywords."""
        try:
            logger.debug(f"Processing file: {input_file}")
            
            # Load quotations from input file
            quotations = load_quotations(input_file)
            if not quotations:
                logger.warning(f"No quotations found in {input_file}")
                return {"keywords": []}

            # Extract keywords
            response = self.chain(
                research_objectives=research_objectives,
                quotations=quotations
            )
            keywords = response.get("keywords", [])
            
            # Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            basename = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(
                self.output_dir,
                f"keywords_{basename}_{timestamp}.json"
            )
            
            # Save keywords
            save_keywords(keywords, output_file)
            
            return {"keywords": keywords, "output_file": output_file}
            
        except Exception as e:
            logger.error(f"Error processing file {input_file}: {e}", exc_info=True)
            return {"keywords": []}