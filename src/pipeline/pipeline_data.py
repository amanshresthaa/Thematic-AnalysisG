# src/pipeline/pipeline_data.py

import logging
import os
import json
from typing import List

def create_directories(dir_paths: List[str]) -> None:
    """
    Creates directories if they do not exist.
    """
    logger = logging.getLogger(__name__)
    for path in dir_paths:
        try:
            os.makedirs(path, exist_ok=True)
            logger.info(f"Directory ensured: {path}")
        except Exception as e:
            logger.error(f"Failed to create directory {path}: {e}", exc_info=True)
            raise

def generate_theme_input(info_path: str, grouping_path: str, output_path: str) -> None:
    """
    Generates queries_theme.json from info.json and query_results_grouping.json.
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating queries_theme.json...")

    if not os.path.exists(info_path):
        logger.error("info.json not found.")
        return

    if not os.path.exists(grouping_path):
        logger.error("query_results_grouping.json not found.")
        return

    with open(info_path, 'r', encoding='utf-8') as f:
        info = json.load(f)

    with open(grouping_path, 'r', encoding='utf-8') as f:
        grouping_results = json.load(f)

    if not grouping_results:
        logger.error("query_results_grouping.json is empty.")
        return

    first_result = grouping_results[0]
    research_objectives = info.get("research_objectives", "")
    theoretical_framework = info.get("theoretical_framework", {})

    codes = first_result.get("grouping_info", {}).get("codes", [])
    groupings = first_result.get("groupings", [])

    # Assume placeholders for demonstration
    quotation = "Original quotation from previous step."
    keywords = ["improvement", "reasoning", "innovation"]
    transcript_chunk = "Some relevant transcript chunk."

    queries_theme = [
        {
            "quotation": quotation,
            "keywords": keywords,
            "codes": codes,
            "research_objectives": research_objectives,
            "theoretical_framework": theoretical_framework,
            "transcript_chunk": transcript_chunk,
            "groupings": groupings
        }
    ]

    create_directories([os.path.dirname(output_path)])
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(queries_theme, f, indent=4)
    logger.info(f"Generated {output_path} successfully.")