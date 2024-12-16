# src/convert/convertkeywordforcoding.py

import json
import os
import re
from typing import List, Dict, Any

def split_into_sentences(text: str) -> List[str]:
    """
    Splits a given text into sentences using regular expressions.
    """
    sentence_endings = re.compile(r'(?<=[.!?]) +')
    sentences = sentence_endings.split(text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def extract_keywords(keywords_list: List[Dict[str, Any]]) -> List[str]:
    """
    Extracts the keyword strings from the keywords list.
    """
    return [keyword_entry.get('keyword', '') for keyword_entry in keywords_list if 'keyword' in keyword_entry]

def convert_query_results(input_file: str, output_dir: str, output_file: str):
    """
    Convert query results to a simplified format and save them in the specified directory.
    """
    # Ensure the output directory exists
    os.makedirs(os.path.join(output_dir, 'input'), exist_ok=True)

    try:
        # Read the input JSON file
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        simplified_data: List[Dict[str, Any]] = []

        for entry in input_data:
            quotation_info = entry.get('quotation_info', {})
            full_quotation = quotation_info.get('quotation', '')
            research_objectives = quotation_info.get('research_objectives', '')
            theoretical_framework = quotation_info.get('theoretical_framework', {})
            transcript_chunk = full_quotation

            # Split the full quotation into sentences
            sentences = split_into_sentences(full_quotation)

            # Extract keywords
            keywords = extract_keywords(entry.get('keywords', []))

            for sentence in sentences:
                simplified_entry = {
                    "quotation": sentence,
                    "keywords": keywords,
                    "research_objectives": research_objectives,
                    "theoretical_framework": theoretical_framework,
                    "transcript_chunk": transcript_chunk
                }
                simplified_data.append(simplified_entry)

        # Define the output path with '_standard' suffix
        output_path = os.path.join(output_dir, 'input', output_file)

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
        input_file='data/output/query_results_keyword_extraction.json',  # Ensure this path is correct
        output_dir='data',
        output_file='queries_coding_standard.json'  # Modified filename
    )
