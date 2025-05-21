"""
Partial Correct Answer Extractor.

This module identifies and extracts questions from model responses where
only some of the answers are correct (partially correct). These are questions
where the model is not consistently correct or incorrect across all responses.

python src/data/partial_correct_extractor.py \
  --input data/results/qwen_72b.json \
  --output data/partial/qwen_72b_partial_correct.json \
  --responses_per_question 5
"""

import os
import json
import logging
import argparse
from typing import List, Dict, Any
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PartialCorrectExtractor:
    """
    Extracts partially correct data from model responses.
    
    Identifies questions where only some of the model's responses are correct
    (not all correct and not all incorrect).
    """
    
    def __init__(self, responses_per_question: int = 5):
        """
        Initialize the extractor.
        
        Args:
            responses_per_question: Number of responses per question
        """
        self.responses_per_question = responses_per_question
    
    def extract_partial_correct(self, data: List[Dict]) -> List[Dict]:
        """
        Extract questions with partially correct answers.
        
        Args:
            data: List of model response data
            
        Returns:
            List of response groups with partially correct answers
        """
        partial_correct_data = []
        total_groups = len(data) // self.responses_per_question
        
        logger.info(f"Processing {total_groups} question groups")
        
        # Process data in groups based on responses_per_question
        for i in tqdm(range(0, len(data), self.responses_per_question), desc="Extracting partial correct data"):
            # Get the current group of responses (all responses for one question)
            group = data[i:i + self.responses_per_question]
            
            # Skip incomplete groups
            if len(group) < self.responses_per_question:
                logger.warning(f"Skipping incomplete group at index {i}")
                continue
            
            # Extract correctness values for the group
            correct_values = [item.get('correct', False) for item in group]
            
            # Check if the group has both correct and incorrect answers
            # (not all correct and not all incorrect)
            if not (all(correct_values) or not any(correct_values)):
                partial_correct_data.extend(group)
        
        return partial_correct_data
    
    def process_file(self, input_file: str, output_file: str) -> None:
        """
        Process a file to extract partially correct data.
        
        Args:
            input_file: Path to input file
            output_file: Path to output file
        """
        try:
            # Load input data
            logger.info(f"Loading data from {input_file}")
            with open(input_file, 'r', encoding='utf-8') as infile:
                data = json.load(infile)
            
            logger.info(f"Loaded {len(data)} responses for {len(data) // self.responses_per_question} questions")
            
            # Extract partially correct data
            partial_correct_data = self.extract_partial_correct(data)
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save the extracted data
            logger.info(f"Saving {len(partial_correct_data)} responses to {output_file}")
            with open(output_file, 'w', encoding='utf-8') as outfile:
                json.dump(partial_correct_data, outfile, ensure_ascii=False, indent=4)
            
            # Log summary
            original_question_count = len(data) // self.responses_per_question
            partial_correct_count = len(partial_correct_data) // self.responses_per_question
            percentage = (partial_correct_count / original_question_count) * 100 if original_question_count > 0 else 0
            
            logger.info(f"Original data: {original_question_count} questions")
            logger.info(f"Partially correct data: {partial_correct_count} questions ({percentage:.2f}%)")
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")


def main():
    """Main entry point with command line argument handling."""
    parser = argparse.ArgumentParser(description="Extract questions with partially correct answers")
    parser.add_argument("--input", required=True, help="Input file path")
    parser.add_argument("--output", required=True, help="Output file path")
    parser.add_argument("--responses_per_question", type=int, default=5,
                       help="Number of responses per question (default: 5)")
    
    args = parser.parse_args()
    
    extractor = PartialCorrectExtractor(responses_per_question=args.responses_per_question)
    extractor.process_file(args.input, args.output)


if __name__ == "__main__":
    main()