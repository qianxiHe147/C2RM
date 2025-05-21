"""
Data classification module.

This module classifies model responses based on correctness and confidence
to prepare training data for the reward model.

Label descriptions:
- class_1: Correct and High Confidence (True & Certain, T&C)
- class_2: Incorrect and High Confidence (False & Certain, F&C)  
- class_3: Correct but Low Confidence (True & Uncertain, T&U)
- class_4: Incorrect and Low Confidence (False & Uncertain, F&U)

# Process a single file
python src/data/data_classification.py --input data/results/dataset.json --threshold 50.0

# Process an entire directory
python src/data/data_classification.py --input data/results/scieval/ --threshold 50.0
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define label constants for easy reference in code
CLASS_TRUE_CERTAIN = "class_1"    # Correct and High Confidence 
CLASS_FALSE_CERTAIN = "class_2"   # Incorrect and High Confidence
CLASS_TRUE_UNCERTAIN = "class_3"  # Correct but Low Confidence
CLASS_FALSE_UNCERTAIN = "class_4" # Incorrect and Low Confidence


class DataClassifier:
    """
    Classifies model responses based on correctness and confidence.
    
    Label descriptions:
    - class_1: Correct and High Confidence (True & Certain, T&C)
    - class_2: Incorrect and High Confidence (False & Certain, F&C)  
    - class_3: Correct but Low Confidence (True & Uncertain, T&U)
    - class_4: Incorrect and Low Confidence (False & Uncertain, F&U)
    """
    
    def __init__(self, confidence_threshold: float = 50.0):
        """
        Initialize the data classifier.
        
        Args:
            confidence_threshold: Threshold for distinguishing between certain and uncertain responses
        """
        self.confidence_threshold = confidence_threshold
    
    def classify_item(self, item: Dict) -> str:
        """
        Classify a single data item based on correctness and confidence.
        
        Args:
            item: Data item with correctness and confidence information
            
        Returns:
            Classification label (class_1, class_2, class_3, or class_4)
        """
        # Get correctness
        correct = item.get('correct', False)
        
        # Parse confidence value
        confidence_str = item.get('model_confidence', '0%')
        confidence_str = confidence_str.replace('%', '')
        try:
            confidence = float(confidence_str)
        except (ValueError, TypeError):
            confidence = 0.0
            logger.warning("Could not parse confidence value, defaulting to 0.0")
        
        # Apply classification logic
        if correct and confidence >= self.confidence_threshold:
            return CLASS_TRUE_CERTAIN       # class_1: Correct and High Confidence
        elif not correct and confidence >= self.confidence_threshold:
            return CLASS_FALSE_CERTAIN      # class_2: Incorrect and High Confidence
        elif correct and confidence < self.confidence_threshold:
            return CLASS_TRUE_UNCERTAIN     # class_3: Correct but Low Confidence
        else:  # not correct and confidence < threshold
            return CLASS_FALSE_UNCERTAIN    # class_4: Incorrect and Low Confidence
    
    def process_file(self, input_file: str, output_file: Optional[str] = None) -> None:
        """
        Process a JSON file to classify data items.
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to output JSON file (defaults to input file if None)
        """
        # Use input file as output if not specified
        if output_file is None:
            output_file = input_file
        
        logger.info(f"Processing file: {input_file}")
        
        try:
            # Load data
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Add classification to each item
            for item in tqdm(data, desc="Classifying items"):
                class_label = self.classify_item(item)
                item['class_label'] = class_label
            
            # Save the processed data
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            
            logger.info(f"Classified data written to {output_file}")
            
            # Log class distribution
            self.log_class_distribution(data)
            
        except Exception as e:
            logger.error(f"Error processing file {input_file}: {e}")
    
    def log_class_distribution(self, data: List[Dict]) -> Dict[str, int]:
        """
        Calculate and log the distribution of response types.
        
        Args:
            data: List of classified data items
            
        Returns:
            Dictionary with counts for each response type
        """
        # Count instances of each response type
        class_counts = {
            CLASS_TRUE_CERTAIN: 0,
            CLASS_FALSE_CERTAIN: 0,
            CLASS_TRUE_UNCERTAIN: 0,
            CLASS_FALSE_UNCERTAIN: 0
        }
        
        for item in data:
            class_label = item.get('class_label')
            if class_label in class_counts:
                class_counts[class_label] += 1
        
        # Calculate percentages
        total = len(data)
        percentages = {
            cls: f"{count / total * 100:.2f}%" if total > 0 else "0.00%" 
            for cls, count in class_counts.items()
        }
        
        # Log the distribution
        logger.info("Class label distribution:")
        logger.info(f"  {CLASS_TRUE_CERTAIN} (Correct and High Confidence): {class_counts[CLASS_TRUE_CERTAIN]} ({percentages[CLASS_TRUE_CERTAIN]})")
        logger.info(f"  {CLASS_FALSE_CERTAIN} (Incorrect and High Confidence): {class_counts[CLASS_FALSE_CERTAIN]} ({percentages[CLASS_FALSE_CERTAIN]})")
        logger.info(f"  {CLASS_TRUE_UNCERTAIN} (Correct but Low Confidence): {class_counts[CLASS_TRUE_UNCERTAIN]} ({percentages[CLASS_TRUE_UNCERTAIN]})")
        logger.info(f"  {CLASS_FALSE_UNCERTAIN} (Incorrect and Low Confidence): {class_counts[CLASS_FALSE_UNCERTAIN]} ({percentages[CLASS_FALSE_UNCERTAIN]})")
        
        return class_counts
    
    def process_directory(self, directory: str) -> None:
        """
        Process all JSON files in a directory.
        
        Args:
            directory: Path to directory containing JSON files
        """
        logger.info(f"Processing directory: {directory}")
        
        # Get all JSON files in the directory
        json_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(directory)
            for file in files
            if file.endswith('.json')
        ]
        
        logger.info(f"Found {len(json_files)} JSON files")
        
        # Process each file
        for file_path in json_files:
            self.process_file(file_path)


def main():
    """Main entry point with command line argument handling."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Classify model responses for reward model training")
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument("--output", help="Output file (only used if input is a file)")
    parser.add_argument("--threshold", type=float, default=50.0, 
                        help="Confidence threshold for classification (default: 50.0)")
    
    args = parser.parse_args()
    
    classifier = DataClassifier(confidence_threshold=args.threshold)
    
    if os.path.isdir(args.input):
        classifier.process_directory(args.input)
    else:
        classifier.process_file(args.input, args.output)


if __name__ == "__main__":
    main()