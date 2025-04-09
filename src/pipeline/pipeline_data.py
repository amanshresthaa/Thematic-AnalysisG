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

    # Check if grouping results exist
    if not os.path.exists(grouping_path):
        logger.error(f"Grouping results file not found at {grouping_path}")
        return
        
    # Get chunker info from latest chunker output if available
    chunker_info = {}
    latest_path = os.path.join("data", "chunker_output", "latest")
    if os.path.exists(latest_path):
        try:
            with open(latest_path, 'r') as f:
                timestamp = f.read().strip()
                info_file = os.path.join("data", "chunker_output", timestamp, "info.json")
                if os.path.exists(info_file):
                    with open(info_file, 'r', encoding='utf-8') as f:
                        chunker_info = json.load(f)
                        logger.info(f"Loaded chunker info from {info_file}")
        except Exception as e:
            logger.warning(f"Could not load chunker info: {e}")
    
    # Load grouping results
    try:
        with open(grouping_path, 'r', encoding='utf-8') as f:
            grouping_results = json.load(f)
    except Exception as e:
        logger.error(f"Error loading grouping results: {e}")
        return

    if not grouping_results:
        logger.error("Grouping results file is empty.")
        return
    
    logger.info(f"Processing {len(grouping_results)} grouping results for theme input")
    
    # Create theme queries from each grouping result
    queries_theme = []
    
    for result in grouping_results:
        # Extract data from the grouping result
        research_objectives = result.get("research_objectives", chunker_info.get("research_objectives", ""))
        
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
    
    logger.info(f"Created {len(queries_theme)} theme queries")
    
    # Ensure the output directory exists
    create_directories([os.path.dirname(output_path)])
    
    # Write the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(queries_theme, f, indent=4)
    
    logger.info(f"Generated {output_path} successfully with {len(queries_theme)} items")