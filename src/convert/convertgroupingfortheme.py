# src/convert/convertgroupingfortheme.py

import json
import os
import logging

logger = logging.getLogger(__name__)

def convert_query_results(input_file: str, output_dir: str, output_file: str):
    """
    Converts query_results_grouping.json to queries_theme.json.

    Args:
        input_file (str): Path to query_results_grouping.json.
        output_dir (str): Directory where queries_theme.json will be saved.
        output_file (str): Name of the output file (e.g., queries_theme.json).
    """
    try:
        logger.info(f"Starting conversion from {input_file} to {os.path.join(output_dir, output_file)}")

        if not os.path.exists(input_file):
            logger.error(f"Input file {input_file} does not exist.")
            return

        with open(input_file, 'r', encoding='utf-8') as f:
            grouping_results = json.load(f)

        if not grouping_results:
            logger.error(f"Input file {input_file} is empty.")
            return

        # Process each result from grouping analysis
        queries_theme = []
        
        for result in grouping_results:
            # Extract data from the grouping result
            research_objectives = result.get("research_objectives", "")
            
            quotation = result.get("quotation", "") 
            
            codes = result.get("grouping_info", {}).get("codes", [])
            
            groupings = result.get("groupings", [])
            
            keywords = result.get("keywords", [])
            
            transcript_chunk = result.get("transcript_chunk", "") 
            
            theoretical_framework = result.get("theoretical_framework", {})
            
            
            # If theoretical_framework is missing or incomplete, create a default structure
            if not theoretical_framework:
                theoretical_framework = {
                    "theory": "Thematic analysis",
                    "philosophical_approach": "Inductive approach",
                    "rationale": "To identify patterns and themes in qualitative data"
                }

            # Create a properly formatted theme query item
            theme_query = {
                "quotation": quotation,
                "keywords": keywords,
                "codes": codes,
                "research_objectives": research_objectives,
                "theoretical_framework": theoretical_framework,
                "transcript_chunk": transcript_chunk,
                "groupings": groupings,
                "document_id": result.get("document_id", "unknown"),
                "id": result.get("id", f"theme_{len(queries_theme)}")
            }
            
            queries_theme.append(theme_query)
        
        logger.info(f"Processed {len(queries_theme)} grouping results into theme queries")

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(queries_theme, f, indent=4)

        logger.info(f"Successfully converted and saved to {output_path}")
    except Exception as e:
        logger.error(f"Error during conversion: {e}", exc_info=True)
