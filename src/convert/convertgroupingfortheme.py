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

        first_result = grouping_results[0]

        # Assuming info.json has already been handled in main.py's generate_theme_input
        # Here, we're just copying necessary fields
        codes = first_result.get("grouping_info", {}).get("codes", [])
        groupings = first_result.get("groupings", [])

        # Placeholders; actual implementation should retrieve these from the pipeline
        quotation = "Original quotation from previous step."
        keywords = ["improvement", "reasoning", "innovation"]
        transcript_chunk = "A relevant transcript chunk providing context."

        queries_theme = [
            {
                "quotation": quotation,
                "keywords": keywords,
                "codes": codes,
                "research_objectives": "To develop advanced reasoning capabilities in AI models...",
                "theoretical_framework": {
                    "theory": "Scaling laws of deep learning suggest...",
                    "philosophical_approach": "A pragmatic, engineering-driven approach...",
                    "rationale": "Reasoning is identified as a crucial bottleneck..."
                },
                "transcript_chunk": transcript_chunk,
                "groupings": groupings
            }
        ]

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(queries_theme, f, indent=4)

        logger.info(f"Successfully converted and saved to {output_path}")
    except Exception as e:
        logger.error(f"Error during conversion: {e}", exc_info=True)
