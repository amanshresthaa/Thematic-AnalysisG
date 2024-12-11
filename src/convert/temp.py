import json
import os
from typing import List, Dict, Any

def convert_query_results(input_file: str, output_dir: str, output_file: str):
    """
    Convert query results to only extract the code arrays from all entries and write them to an output JSON file.
    If multiple entries have the same 'code' name, only the first instance is included in the final output.
    """

    # Ensure the output directory exists
    os.makedirs(os.path.join(output_dir, 'input'), exist_ok=True)

    try:
        # Read the input JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        # Ensure we are dealing with a list of entries
        if not isinstance(input_data, list):
            input_data = [input_data]

        # A dictionary to keep track of codes we've already seen to avoid duplicates
        seen_codes = {}
        final_codes = []

        # Iterate over each entry
        for entry in input_data:
            codes = entry.get('codes', [])
            for code_entry in codes:
                # code_entry should be a dictionary containing 'code' and other keys
                code_name = code_entry.get('code', '')
                if code_name and code_name not in seen_codes:
                    seen_codes[code_name] = True
                    final_codes.append(code_entry)

        # Write the unique codes to the output JSON file
        output_path = os.path.join(output_dir, 'input', output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_codes, f, indent=4, ensure_ascii=False)

        print(f"Successfully extracted and saved unique codes to {output_path}")

    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except json.JSONDecodeError as jde:
        print(f"Error decoding JSON: {jde}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Example usage; adjust paths as needed
    convert_query_results(
        input_file='src/convert/query_results_coding_analysis.json',  # Replace with your input file path
        output_dir='data/',                               # Replace with your desired output directory
        output_file='queries_themes.json'                  # Replace with your desired output file name
    )
