# src/convert/convertquotationforkeyword.py

import json
import os
from typing import List, Dict, Any

def convert_query_results(input_file: str, output_dir: str, output_file: str):
    """
    Convert query results to a simplified format and save them in the specified directory.

    The simplified format includes:
    - quotation: The actual quotation text.
    - research_objectives: The research objectives from transcript_info.
    - theoretical_framework: The theoretical framework details from transcript_info (preserved as a nested dictionary).
    - transcript_chunk: (Optional) The full transcript chunk for context.

    If a result contains multiple quotations, each quotation will be a separate entry in the simplified data.
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(output_dir, 'input'), exist_ok=True)

    try:
        # Read the query results file
        with open(input_file, 'r', encoding='utf-8') as f:
            results = json.load(f)

        # Convert to simplified format
        simplified_data: List[Dict[str, Any]] = []
        for result in results:
            transcript_info = result.get('transcript_info', {})
            research_objectives = transcript_info.get('research_objectives', '')
            theoretical_framework = transcript_info.get('theoretical_framework', {})
            transcript_chunk = transcript_info.get('transcript_chunk', '')

            quotations = result.get('quotations', [])
            for quotation_entry in quotations:
                quotation = quotation_entry.get('quotation', '')
                simplified_entry = {
                    "quotation": quotation,
                    "research_objectives": research_objectives,
                    "theoretical_framework": theoretical_framework,  # Preserved as a nested dictionary
                    "transcript_chunk": transcript_chunk  # Optional: Include for additional context
                }
                simplified_data.append(simplified_entry)

        # Save to output file with '_standard' suffix
        output_path = os.path.join(output_dir, 'input', output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_data, f, indent=4, ensure_ascii=False)

        print(f"Successfully converted and saved to {output_path}")

    except Exception as e:
        print(f"Error converting file: {e}")

if __name__ == "__main__":
    # Example usage; this will only run when the script is executed directly
    convert_query_results(
        input_file='data/output/query_results_quotation.json',  # Ensure this path is correct
        output_dir='data',
        output_file='queries_keyword_standard.json'  # Modified filename
    )
