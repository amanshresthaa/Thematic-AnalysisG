import logging
import json
from typing import List, Dict, Any
import dspy
import os

logger = logging.getLogger(__name__)

class KeywordExtractionSignature(dspy.Signature):
    """
    Extract keywords from quotations by evaluating them against the 6Rs criteria.
    """
    research_objectives: str = dspy.InputField(
        desc="The research objectives guiding the extraction of keywords."
    )
    quotations: List[Dict[str, str]] = dspy.InputField(
        desc="List containing a single quotation with 'quote' field."
    )
    keywords_data: List[Dict[str, Any]] = dspy.OutputField(
        desc="List of extracted keywords with 6Rs scores and justifications."
    )
    
    def forward(self, research_objectives: str, quotations: List[Dict[str, str]]) -> Dict[str, Any]:
        try:
            if not quotations:
                logger.warning("No quotations provided for keyword extraction.")
                return {"keywords_data": []}
            
            quote = quotations[0].get("quote", "")
            if not quote:
                logger.warning("Empty quote provided for keyword extraction.")
                return {"keywords_data": []}

            logger.debug("Starting keyword extraction process with 6Rs enhancement.")
            prompt = (
                f"You are an expert in qualitative research and thematic analysis.\n\n"
                f"**Research Objectives**: {research_objectives}\n\n"
                f"**Quotation**:\n\"{quote}\"\n\n"
                f"Based on the 6Rs criteria (Realness, Richness, Repetition, Rationale, Repartee, Regal), extract key terms, concepts, and themes from the quotation.\n"
                f"For each keyword, provide scores from 1 to 5 for each of the 6Rs and a brief justification.\n"
                f"Return the results in JSON format like this:\n"
                f"[\n"
                f"  {{\n"
                f"    \"keyword\": \"...\",\n"
                f"    \"realness\": 4,\n"
                f"    \"richness\": 5,\n"
                f"    \"repetition\": 3,\n"
                f"    \"rationale\": 4,\n"
                f"    \"repartee\": 2,\n"
                f"    \"regal\": 5,\n"
                f"    \"justification\": \"...\"\n"
                f"  }},\n"
                f"  ...\n"
                f"]\n"
            )

            response = self.language_model.generate(
                prompt=prompt,
                max_tokens=800,
                temperature=0.5,
                top_p=0.9,
                n=1,
                stop=None
            ).strip()

            # Parse the response to extract keywords data
            keywords_data = json.loads(response)

            # Validate the parsed data
            if not isinstance(keywords_data, list):
                logger.error("The response is not a list of keywords data.")
                return {"keywords_data": []}

            logger.info(f"Extracted {len(keywords_data)} keywords with 6Rs scores.")
            return {"keywords_data": keywords_data}
                
        except Exception as e:
            logger.error(f"Error in keyword extraction: {e}", exc_info=True)
            return {"keywords_data": []}

def load_quotations(input_file: str) -> List[Dict[str, str]]:
    """Load quotations from input JSON file."""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading quotations from {input_file}: {e}")
        return []

def save_keywords(keywords_mapping: List[Dict[str, Any]], output_file: str):
    """Save extracted keywords data mapping to output JSON file."""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(keywords_mapping, f, indent=4)
        logger.info(f"Keywords data mapping saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving keywords to {output_file}: {e}")