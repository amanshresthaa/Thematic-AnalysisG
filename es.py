import csv
import json
from typing import List, Dict, Union
import logging
from dataclasses import dataclass
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class StandardQuotation:
    quote_data: Dict[str, Union[str, float]]
    analysis: Dict[str, str]

class CSVStandardizer:
    def __init__(self, input_file: str):
        self.input_file = Path(input_file)
        self.logger = logging.getLogger(__name__)

    def read_csv(self) -> pd.DataFrame:
        """Read CSV file using pandas."""
        try:
            df = pd.read_csv(self.input_file)
            return df
        except Exception as e:
            self.logger.error(f"Error reading CSV file: {e}")
            raise

    def standardize_quote_data(self, quote_dict: Dict) -> Dict:
        """Standardize quote data format."""
        return {
            "quotation": quote_dict.get("quotation", quote_dict.get("quotation", "")),
            "context": quote_dict.get("context", "No context provided"),
            "type": quote_dict.get("type", "discrete"),
            "function": quote_dict.get("function", "evidence"),
            "pattern_strength": float(quote_dict.get("pattern_strength", 0.5)),
            "theoretical_alignment": float(quote_dict.get("theoretical_alignment", 0.5))
        }

    def create_analysis(self, quote_dict: Dict) -> Dict:
        """Create standardized analysis section."""
        return {
            "selection_rationale": quote_dict.get("selection_rationale", "Not provided"),
            "theoretical_relevance": quote_dict.get("theoretical_relevance", "Not provided"),
            "objective_support": quote_dict.get("objective_support", "Not provided"),
            "patterns": quote_dict.get("patterns", "Not provided")
        }

    def process_row(self, row: Dict) -> Dict:
        """Process a single row from the CSV."""
        try:
            # Parse input and output JSON strings
            input_data = json.loads(row['input'])
            output_data = json.loads(row['output'])

            # Create standardized quotation
            standardized_quotes = []
            for quote in output_data:
                quote_data = self.standardize_quote_data(quote)
                analysis = self.create_analysis(quote)
                standardized_quotes.append({
                    "quote_data": quote_data,
                    "analysis": analysis
                })

            return {
                "input": input_data,
                "output": standardized_quotes
            }
        except Exception as e:
            self.logger.warning(f"Error processing row: {e}")
            return None

    def standardize_dataset(self) -> List[Dict]:
        """Main method to standardize the CSV dataset."""
        try:
            # Read CSV
            df = self.read_csv()
            
            # Process each row
            standardized_data = []
            for _, row in df.iterrows():
                processed_row = self.process_row(row)
                if processed_row:
                    standardized_data.append(processed_row)

            return standardized_data

        except Exception as e:
            self.logger.error(f"Error standardizing dataset: {e}")
            raise

    def save_standardized_csv(self, output_file: str):
        """Save standardized data back to CSV."""
        try:
            standardized_data = self.standardize_dataset()
            
            # Convert to DataFrame
            rows = []
            for item in standardized_data:
                rows.append({
                    'input': json.dumps(item['input'], ensure_ascii=False),
                    'output': json.dumps(item['output'], ensure_ascii=False)
                })
            
            df = pd.DataFrame(rows)
            df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
            self.logger.info(f"Standardized dataset saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving standardized CSV: {e}")
            raise

def main():
    # Configure input and output files
    input_file = "data/new_training_data.csv"
    output_file = "standardized_dataset.csv"
    
    # Create standardizer instance
    standardizer = CSVStandardizer(input_file)
    
    try:
        # Process and save the standardized dataset
        standardizer.save_standardized_csv(output_file)
        print(f"Dataset standardization completed. Output saved to {output_file}")
        
        # Print sample of standardized format
        df = pd.read_csv(output_file)
        print("\nSample of standardized format:")
        print(df.head(1).to_string())
        
    except Exception as e:
        print(f"Error during standardization: {e}")

if __name__ == "__main__":
    main()