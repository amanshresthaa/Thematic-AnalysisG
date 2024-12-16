# src/convert/convertcodingfortheme.py

import json
import os
from typing import List, Dict, Any, Optional

def extract_coding_info(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts and simplifies coding information from a single JSON entry.
    This revised version keeps codes in dictionary format instead of just extracting their 'code' strings.
    """
    try:
        coding_info = entry.get('coding_info', {})
        quotation = coding_info.get('quotation', '').strip()
        keywords = coding_info.get('keywords', [])
        research_objectives = coding_info.get('research_objectives', '').strip()
        theoretical_framework = coding_info.get('theoretical_framework', {})

        # Extract codes as dictionaries if possible
        codes = entry.get('codes', [])
        processed_codes = []
        for code_entry in codes:
            # Ensure the code_entry is a dictionary and has at least the 'code' key
            if isinstance(code_entry, dict) and 'code' in code_entry:
                processed_codes.append(code_entry)
            else:
                # If the code entry is not in the expected format, skip it or handle as needed
                print(f"Warning: Code entry is not a valid dictionary or missing 'code' key: {code_entry}")

        # Combine transcript chunks
        retrieved_chunks = entry.get('retrieved_chunks', [])
        transcript_contents = [chunk.get('original_content', '').strip() for chunk in retrieved_chunks]
        transcript_chunk = ' '.join([quotation] + transcript_contents)

        simplified_entry = {
            "quotation": quotation,
            "keywords": keywords,
            "codes": processed_codes,  # Keep codes as dictionaries
            "research_objectives": research_objectives,
            "theoretical_framework": theoretical_framework,
            "transcript_chunk": transcript_chunk
        }

        return simplified_entry

    except Exception as e:
        print(f"Error processing entry: {e}")
        return {}

def convert_query_results(input_file: str, output_dir: str, output_file: str, sub_dir: Optional[str] = 'input'):
    """
    Convert query results to a simplified format and save them in the specified directory.
    
    The simplified format includes:
    - quotation: The actual quotation text.
    - keywords: List of keywords associated with the quotation.
    - codes: List of code dictionaries associated with the quotation.
    - research_objectives: The research objectives.
    - theoretical_framework: The theoretical framework details.
    - transcript_chunk: The combined transcript chunk for context.
    """
    # Ensure the output directory exists
    if sub_dir:
        target_dir = os.path.join(output_dir, sub_dir)
    else:
        target_dir = output_dir
    os.makedirs(target_dir, exist_ok=True)

    try:
        # Read the input JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        # Check if input_data is a list; if not, make it a list
        if not isinstance(input_data, list):
            input_data = [input_data]

        simplified_data: List[Dict[str, Any]] = []

        for idx, entry in enumerate(input_data):
            simplified_entry = extract_coding_info(entry)
            if simplified_entry:
                simplified_data.append(simplified_entry)
            else:
                print(f"Entry at index {idx} could not be processed and was skipped.")

        # Define the output path
        output_path = os.path.join(target_dir, output_file)

        # Write the simplified data to the output JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_data, f, indent=4, ensure_ascii=False)

        print(f"Successfully converted and saved to {output_path}")

    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except json.JSONDecodeError as jde:
        print(f"Error decoding JSON: {jde}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Example usage; this will only run when the script is executed directly
    convert_query_results(
        input_file='data/output/query_results_coding_analysis.json',  # Replace with your input file path
        output_dir='data',                                           # Set to 'data' to avoid double 'input' subdirectory
        output_file='queries_theme.json'                            # Replace with your desired output file name
        # sub_dir='input'  # Optional: Specify if needed
    )
