import json
import csv

# Load JSON data from a file
input_file = 'data.json'  # Replace with the path to your JSON file
output_file = 'extracted_data.csv'

try:
    # Open and load the JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Check if data is a list
    if not isinstance(data, list):
        print("Unexpected JSON structure: Expected a list at the root level.")
    else:
        # Extract data and write to CSV
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['query', 'quotations', 'purpose'])

            # Loop through each item in JSON data and extract relevant fields
            for item in data:
                # Debugging to check each item structure
                if not isinstance(item, dict):
                    print("Unexpected item structure: Expected a dictionary.")
                    continue

                query = item.get('query', '')
                quotations = '; '.join(item.get('quotations', []))  # Join quotations with semicolon
                purpose = item.get('purpose', '')

                # Debugging to confirm values extracted
                print(f"Extracted - Query: {query}, Quotations: {quotations}, Purpose: {purpose}")

                # Write to CSV file
                writer.writerow([query, quotations, purpose])

    print(f"Data has been extracted to {output_file}")

except json.JSONDecodeError:
    print("Failed to parse JSON. Please check the file format.")
except FileNotFoundError:
    print(f"File {input_file} not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
