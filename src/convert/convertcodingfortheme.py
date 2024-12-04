import json
import os
from typing import List, Dict, Any

def extract_coding_info(entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts and simplifies coding information from a single JSON entry.

    Args:
        entry (Dict[str, Any]): A dictionary representing a single JSON entry.

    Returns:
        Dict[str, Any]: A simplified dictionary containing the extracted information.
    """
    try:
        coding_info = entry.get('coding_info', {})
        quotation = coding_info.get('quotation', '').strip()
        keywords = coding_info.get('keywords', [])
        research_objectives = coding_info.get('research_objectives', '').strip()
        theoretical_framework = coding_info.get('theoretical_framework', {})

        # Extract codes
        codes = entry.get('codes', [])
        coding = [code_entry.get('code', '').strip() for code_entry in codes if 'code' in code_entry]

        # Combine transcript chunks
        retrieved_chunks = entry.get('retrieved_chunks', [])
        transcript_contents = [chunk.get('original_content', '').strip() for chunk in retrieved_chunks]
        transcript_chunk = ' '.join([quotation] + transcript_contents)

        simplified_entry = {
            "quotation": quotation,
            "keywords": keywords,
            "coding": coding,
            "research_objectives": research_objectives,
            "theoretical_framework": theoretical_framework,
            "transcript_chunk": transcript_chunk
        }

        return simplified_entry

    except Exception as e:
        print(f"Error processing entry: {e}")
        return {}

def process_input_file(input_file: str, output_dir: str, output_file: str):
    """
    Processes the input JSON file to extract and simplify coding information.

    Args:
        input_file (str): Path to the input JSON file.
        output_dir (str): Directory where the output will be saved.
        output_file (str): Name of the output JSON file.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

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
        output_path = os.path.join(output_dir, output_file)

        # Write the simplified data to the output JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(simplified_data, f, indent=4, ensure_ascii=False)

        print(f"Successfully processed and saved to {output_path}")

    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except json.JSONDecodeError as jde:
        print(f"Error decoding JSON: {jde}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Usage Example
if __name__ == "__main__":
    process_input_file(
        input_file='query_results_coding_analysis.json',  # Replace with your input file path
        output_dir='data/input',                  # Replace with your desired output directory
        output_file='queries_theme.json'  # Replace with your desired output file name
    )
