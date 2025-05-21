"""
Format Converter module for C2RM.

This module converts data from Pair (Direct Preference Optimization) format
to SFT (Supervised Fine-Tuning) format, specifically for confidence assessment training.
It processes pairs of chosen and rejected responses into individual training samples.

python src/data/format_converter.py \
  --input data/pair/scaling/data.json \
  --output data/sft/scaling/data.json \
  --seed 42
"""

import os
import json
import re
import random
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

class FormatConverter:
    """
    Converts data from Pair format to SFT format for confidence assessment training.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the format converter.
        
        Args:
            seed: Random seed for reproducibility when shuffling data
        """
        self.seed = seed
        random.seed(seed)
    
    def remove_explanation_prefix(self, text: str) -> str:
        """
        Remove "Explanation:" prefix from text.
        
        Args:
            text: Input text that may contain explanation prefix
            
        Returns:
            Text with explanation prefix removed
        """
        explanation_pattern = r"^(```\s*)?Explanation:\s*"
        return re.sub(explanation_pattern, '', text)
    
    def remove_question_prefix(self, text: str) -> str:
        """
        Remove "Question:" prefix from text.
        
        Args:
            text: Input text that may contain question prefix
            
        Returns:
            Text with question prefix removed
        """
        question_pattern = r"^(```\s*)?Question:\s*"
        return re.sub(question_pattern, '', text)
    
    def load_data(self, input_file: str) -> List[Dict[str, Any]]:
        """
        Load data from input file.
        
        Args:
            input_file: Path to input file
            
        Returns:
            List of data items
        """
        try:
            logger.info(f"Loading data from {input_file}")
            with open(input_file, 'r', encoding='utf-8') as infile:
                data = json.load(infile)
            logger.info(f"Loaded {len(data)} items")
            return data
        except Exception as e:
            logger.error(f"Error loading file {input_file}: {e}")
            return []
    
    def save_data(self, data: List[Dict[str, Any]], output_file: str) -> None:
        """
        Save data to output file.
        
        Args:
            data: Data to save
            output_file: Path to output file
        """
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as outfile:
                json.dump(data, outfile, indent=4, ensure_ascii=False)
            logger.info(f"Saved {len(data)} items to {output_file}")
        except Exception as e:
            logger.error(f"Error saving to {output_file}: {e}")
    
    def convert_to_sft_format(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert Pair format data to SFT format.
        
        Args:
            data: Input data in Pair format
            
        Returns:
            Data converted to SFT format
        """
        sft_data = []
        
        for item in tqdm(data, desc="Converting data format"):
            # Extract fields
            instruction = item.get('instruction', '')
            chosen = item.get('chosen', '')
            rejected = item.get('rejected', '')
            
            # Process text
            instruction = self.remove_question_prefix(instruction)
            chosen = self.remove_explanation_prefix(chosen)
            rejected = self.remove_explanation_prefix(rejected)
            
            # Create prompt template
            prompt_template = (
                "Given the following Question and the corresponding Answer provided by a model, "
                "you are required to assess whether the model is certain about its answer. "
                "If the model is certain about its answer, output 'Yes'. "
                "If the model is uncertain about its answer, output 'No'."
                "\n\nQuestion:\n{question}\n\nModel's Answer:\n{answer}"
            )
            
            # Create entry for chosen answer (marked as certain)
            new_entry_1 = {
                "instruction": prompt_template.format(question=instruction, answer=chosen),
                "input": "",
                "output": "Yes"
            }
            sft_data.append(new_entry_1)
            
            # Create entry for rejected answer (marked as uncertain)
            new_entry_2 = {
                "instruction": prompt_template.format(question=instruction, answer=rejected),
                "input": "",
                "output": "No"
            }
            sft_data.append(new_entry_2)
        
        return sft_data
    
    def process_file(self, input_file: str, output_file: str, shuffle: bool = True) -> None:
        """
        Process a file to convert from Pair to SFT format.
        
        Args:
            input_file: Path to input file in Pair format
            output_file: Path to output file for SFT format
            shuffle: Whether to shuffle the output data
        """
        # Load data
        data = self.load_data(input_file)
        if not data:
            return
        
        # Convert format
        sft_data = self.convert_to_sft_format(data)
        
        # Shuffle data if requested
        if shuffle:
            logger.info(f"Shuffling {len(sft_data)} examples")
            random.shuffle(sft_data)
        
        # Save data
        self.save_data(sft_data, output_file)
        
        # Log summary
        logger.info(f"Processed {len(data)} Pair pairs into {len(sft_data)} SFT examples")


def main():
    """Main entry point with command line argument handling."""
    parser = argparse.ArgumentParser(description="Convert Pair format data to SFT format")
    parser.add_argument("--input", required=True, help="Input file path (Pair format)")
    parser.add_argument("--output", required=True, help="Output file path (SFT format)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable shuffling of output data")
    
    args = parser.parse_args()
    
    converter = FormatConverter(seed=args.seed)
    converter.process_file(
        input_file=args.input, 
        output_file=args.output,
        shuffle=not args.no_shuffle
    )


if __name__ == "__main__":
    main()