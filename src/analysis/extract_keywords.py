import logging
import json
from typing import List, Dict, Any
import dspy
import os

logger = logging.getLogger(__name__)

class KeywordExtractionSignature(dspy.Signature):
    """
    Extract keywords from quotations based on the 6Rs framework.
    """
    research_objectives: str = dspy.InputField(
        desc="The research objectives guiding the extraction of keywords."
    )
    quotations: List[Dict[str, str]] = dspy.InputField(
        desc="List containing a single quotation with 'quote' field."
    )
    keywords_data: List[Dict[str, Any]] = dspy.OutputField(
        desc="List of dictionaries containing extracted keywords and their corresponding 6Rs categories."
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
                f"Research Objectives: {research_objectives}\n\n"
                f"Analyze the following quote:\n\"{quote}\"\n\n"
                f"Extract key terms, concepts, and themes from this quotation based on the 6Rs framework:\n"
                f"- **Realness**: Words reflecting real experiences and perceptions.\n"
                f"- **Richness**: Words rich in meaning providing detailed understanding.\n"
                f"- **Repetition**: Words or concepts repeated throughout the data.\n"
                f"- **Rationale**: Words informed by theoretical foundations or research objectives.\n"
                f"- **Repartee**: Witty or poignant words adding depth.\n"
                f"- **Regal**: Crucial words for understanding the core phenomena.\n\n"
                f"Return ONLY a list of keywords that align with these categories. For each keyword, indicate which of the 6Rs it satisfies in the format:\n"
                f"Keyword: Category1, Category2, ...\n"
            )

            response = self.language_model.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.5,
                top_p=0.9,
                n=1,
                stop=None
            ).strip()

            # Process the response to extract keywords and their categories
            keywords_data = []
            lines = response.split('\n')
            for line in lines:
                if line.strip() and not line.startswith(('-', '*', '•')):
                    parts = line.split(':')
                    if len(parts) == 2:
                        keyword = parts[0].strip()
                        categories = [cat.strip() for cat in parts[1].split(',')]
                        keywords_data.append({'keyword': keyword, 'categories': categories})

            # Remove duplicates while preserving order
            seen = set()
            unique_keywords_data = []
            for item in keywords_data:
                key = item['keyword'].lower()
                if key not in seen:
                    seen.add(key)
                    unique_keywords_data.append(item)

            logger.info(f"Extracted {len(unique_keywords_data)} unique keywords with 6Rs categories.")
            return {"keywords_data": unique_keywords_data}
            
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
    """Save extracted keywords mapping to output JSON file."""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(keywords_mapping, f, indent=4)
        logger.info(f"Keywords mapping saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving keywords to {output_file}: {e}")