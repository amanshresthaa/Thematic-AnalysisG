import json
import pandas as pd
import os

def load_json(json_path):
    """
    Load JSON data from a file.

    Parameters:
    - json_path (str): Path to the JSON file.

    Returns:
    - list: List of dictionaries representing each JSON object.
    """
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def format_retrieved_chunks(retrieved_chunks):
    """
    Format the retrieved chunks for inclusion in the prompt.

    Parameters:
    - retrieved_chunks (list): List of retrieved chunks from the JSON data.

    Returns:
    - str: Formatted string of retrieved chunks.
    """
    formatted_chunks = []
    for chunk in retrieved_chunks:
        chunk_id = chunk.get('chunk', {}).get('chunk_id', 'N/A')
        contextual_content = chunk.get('chunk', {}).get('contextualized_content', '').replace('\n', ' ').strip()
        score = chunk.get('score', 0.0)
        formatted_chunks.append(f"  - Chunk ID: {chunk_id}\n    Content: {contextual_content}\n    Score: {score}")
    return "\n".join(formatted_chunks)

def process_entry(entry, prompt_template):
    """
    Process a single JSON entry to fill the prompt template and extract the output.

    Parameters:
    - entry (dict): A dictionary representing a single JSON object.
    - prompt_template (str): The prompt template with placeholders.

    Returns:
    - dict: A dictionary with 'input' and 'output' keys.
    """
    # Extract necessary fields from the JSON entry
    query = entry.get('query', '').replace('\n', ' ').strip()
    research_objectives = entry.get('research_objectives', '').replace('\n', ' ').strip()
    theoretical_framework = entry.get('theoretical_framework', '').replace('\n', ' ').strip()
    retrieved_chunks = entry.get('retrieved_chunks', [])

    # Format retrieved chunks
    chunks_formatted = format_retrieved_chunks(retrieved_chunks)

    # Fill the prompt template with the extracted data
    input_text = prompt_template.format(
        query=query,
        research_objectives=research_objectives,
        theoretical_framework=theoretical_framework,
        chunks_formatted=chunks_formatted
    )

    # Extract the quotations and serialize them as JSON string
    quotations = entry.get('quotations', [])
    output = json.dumps(quotations, ensure_ascii=False)  # ensure_ascii=False to preserve any special characters

    return {"input": input_text, "output": output}

def convert_json_to_csv(json_path, csv_path):
    """
    Convert a JSON file to a CSV file with 'input' and 'output' columns.

    Parameters:
    - json_path (str): Path to the input JSON file.
    - csv_path (str): Path to the output CSV file.
    """
    # Load JSON data
    data = load_json(json_path)

    # Define the prompt template with escaped curly braces
    prompt_template = """
You are conducting thematic analysis following Braun and Clarke's (2006) approach.

Research Objectives:
{research_objectives}

Theoretical Framework:
{theoretical_framework}

Transcript Chunks:
{chunks_formatted}

Task: Select quotations that align with the research objectives and theoretical framework while following these criteria:

For each quotation, provide:

1. quote_data:
{{
    "quotation": str,                    # The exact quotation text
    "context": str,                  # Surrounding context from interview
    "type": str,                     # 'discrete' (brief diverse perspectives)
                                    # 'embedded' (transition phrases) 
                                    # 'longer' (complex understanding)
    "function": str,                 # 'evidence' | 'explanation' | 'illustration' | 
                                    # 'impression' | 'representation'
    "pattern_strength": float,       # 0.0-1.0 indicating robustness of pattern
    "theoretical_alignment": float   # 0.0-1.0 alignment with framework
}}

2. analysis:
{{
    "selection_rationale": str,     # Why this quote was chosen 
    "theoretical_relevance": str,   # How quote supports theoretical framework
    "objective_support": str,       # How quote advances research objectives
    "patterns": str                 # Key patterns identified
}}

Key Guidelines:
1. Focus on relevance to research objectives
2. Ensure theoretical alignment
3. Provide adequate context
4. Identify clear patterns
5. Maintain participant confidentiality
6. Balance diversity of perspectives

Return response as valid JSON with 'quotations' and 'analysis' fields.
"""

    # Process each entry and compile input-output pairs
    processed_data = []
    for idx, entry in enumerate(data):
        processed_entry = process_entry(entry, prompt_template)
        processed_data.append(processed_entry)
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1} entries...")

    # Create a DataFrame
    df = pd.DataFrame(processed_data)

    # Save to CSV
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"CSV file has been created at '{csv_path}' with {len(df)} entries.")

def main():
    """
    Main function to execute the JSON to CSV conversion.
    """
    # Define file paths
    json_input_path = 'query_results.json'       # Ensure this file exists in the same directory
    csv_output_path = 'final_output.csv'

    # Check if the JSON file exists
    if not os.path.exists(json_input_path):
        print(f"Error: The file '{json_input_path}' does not exist in the current directory.")
        return

    # Convert JSON to CSV
    convert_json_to_csv(json_input_path, csv_output_path)

if __name__ == "__main__":
    main()
