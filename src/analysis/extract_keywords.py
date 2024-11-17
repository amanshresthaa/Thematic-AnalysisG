import logging
import json
from typing import List, Dict, Any
import dspy
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class KeywordExtractionSignature(dspy.Signature):
    """
    Extract keywords from quotations with simplified input/output format.
    """
    research_objectives: str = dspy.InputField(
        desc="The research objectives guiding the extraction of keywords."
    )
    quotations: List[Dict[str, str]] = dspy.InputField(
        desc="List containing a single quotation with 'quote' field."
    )
    keywords: List[str] = dspy.OutputField(
        desc="List of extracted keywords for the provided quote."
    )
    
    def forward(self, research_objectives: str, quotations: List[Dict[str, str]]) -> Dict[str, List[str]]:
        try:
            if not quotations:
                logger.warning("No quotations provided for keyword extraction.")
                return {"keywords": []}
            
            quote = quotations[0].get("quote", "")
            if not quote:
                logger.warning("Empty quote provided for keyword extraction.")
                return {"keywords": []}

            logger.debug("Starting keyword extraction process.")
            prompt = (
                f"You are an expert in qualitative research and thematic analysis.\n\n"
                f"Research Objectives: {research_objectives}\n\n"
                f"Analyze the following quote about potato farming and supply chain:\n\"{quote}\"\n\n"
                f"Extract key terms, concepts, and themes from this quotation. "
                f"Return ONLY a list of single words or short phrases (2-3 words maximum) "
                f"that represent the main concepts, without any additional metadata.\n"
                f"Focus on business concepts, agricultural terms, supply chain terminology, "
                f"and important themes mentioned in the quote."
            )

            response = self.language_model.generate(
                prompt=prompt,
                max_tokens=150,
                temperature=0.5,
                top_p=0.9,
                n=1,
                stop=None
            ).strip()

            # Process the response to extract clean keywords
            keywords = [
                keyword.strip().lower()
                for keyword in response.split('\n')
                if keyword.strip() and not keyword.startswith(('-', '*', '•', '1.', '2.'))
            ]
            
            # Remove duplicates while preserving order
            seen = set()
            unique_keywords = [x for x in keywords if not (x in seen or seen.add(x))]

            logger.info(f"Extracted {len(unique_keywords)} unique keywords for the quote.")
            return {"keywords": unique_keywords}
            
        except Exception as e:
            logger.error(f"Error in keyword extraction: {e}", exc_info=True)
            return {"keywords": []}

def load_quotations(input_file: str) -> List[Dict[str, str]]:
    """Load quotations from input JSON file."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading quotations from {input_file}: {e}")
        return []

def save_keywords(keywords_mapping: List[Dict[str, Any]], output_file: str):
    """Save extracted keywords mapping to output JSON file."""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(keywords_mapping, f, indent=4)
        logger.info(f"Keywords mapping saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving keywords to {output_file}: {e}")
