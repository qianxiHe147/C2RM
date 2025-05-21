"""
Answer extraction and confidence calculation module.

This module processes model outputs from various dataset types,
extracts answers, and calculates confidence scores based on
the consistency of model responses.

# SciEval
python src/data/answer_extraction.py --dir data/results/scieval --type scieval --responses 5

# NuminaMath
python src/data/answer_extraction.py --dir data/results/numina --type numina_math --responses 5

# LogiQA, SciKnowEval
python src/data/answer_extraction.py --dir data/results/logiqa --type multiple_choice --responses 5

# LogicNLI
python src/data/answer_extraction.py --dir data/results/logicnli --type logicnli --responses 5
"""

import os
import json
import re
import logging
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import Counter
from enum import Enum

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetType(str, Enum):
    """Dataset types supported by the answer extractor."""
    SCIEVAL = "scieval"
    NUMINA_MATH = "numina_math" 
    MULTIPLE_CHOICE = "multiple_choice"  # General multiple choice (LogiQA, SciKnowEval, etc.)
    LOGICNLI = "logicnli"

class AnswerExtractor:
    """
    Extracts answers from model outputs and calculates confidence 
    for various dataset types.
    """
    
    def __init__(self, 
                 dataset_type: DatasetType,
                 responses_per_question: int = 5):
        """
        Initialize the answer extractor.
        
        Args:
            dataset_type: Type of dataset being processed
            responses_per_question: Number of responses generated per question
        """
        self.dataset_type = dataset_type
        self.responses_per_question = responses_per_question
    
    def extract_answer(self, model_output: str) -> Optional[str]:
        """
        Extract answer from model output based on dataset type.
        
        Args:
            model_output: Model generated output text
            
        Returns:
            Extracted answer or None if not found
        """
        if self.dataset_type == DatasetType.SCIEVAL:
            return self._extract_scieval_answer(model_output)
        elif self.dataset_type == DatasetType.NUMINA_MATH:
            return self._extract_numina_math_answer(model_output)
        elif self.dataset_type == DatasetType.MULTIPLE_CHOICE:
            return self._extract_multiple_choice_answer(model_output)
        elif self.dataset_type == DatasetType.LOGICNLI:
            return self._extract_logicnli_answer(model_output)
        else:
            logger.warning(f"Unsupported dataset type: {self.dataset_type}")
            return None
    
    def _extract_scieval_answer(self, model_output: str) -> Optional[str]:
        """
        Extract answer for SciEval datasets.
        """
        # SciEval supports both multiple-choice and yes/no/maybe answers
        # First try to extract A/B/C/D
        multiple_choice_match = re.findall(r'\b([A-D])\b', model_output)
        if multiple_choice_match:
            return multiple_choice_match[-1]
        
        # If not found, try to extract yes/no/maybe
        yes_no_match = re.findall(r'\b(yes|no|maybe)\b', model_output, re.IGNORECASE)
        if yes_no_match:
            return yes_no_match[-1].lower()
            
        return None
    
    def _extract_numina_math_answer(self, model_output: str) -> Optional[str]:
        """
        Extract answer for NuminaMath-TIR datasets.
        """
        # First try to extract from \boxed{}
        boxed_match = re.findall(r'\\boxed\{((?:[^{}]|\{[^}]*})*)\}', model_output)
        if boxed_match:
            return boxed_match[-1]  # Take the last boxed answer
            
        # If no boxed answer, look for numerical values
        number_match = re.findall(r'\d+\.\d+|\d+', model_output)
        if number_match:
            return number_match[-1].replace(" ", "")
            
        return None
    
    def _extract_multiple_choice_answer(self, model_output: str) -> Optional[str]:
        """
        Extract answer for multiple choice datasets.
        """
        match = re.findall(r'\b([A-Z])\b', model_output)
        if match:
            return match[-1]  # Take the last option letter
        return None
    
    def _extract_logicnli_answer(self, model_output: str) -> Optional[str]:
        """
        Extract answer for LogicNLI datasets.
        """
        match = re.findall(r'\b(entailment|neutral|contradiction|self_contradiction|self-contradiction)\b', 
                          model_output, re.IGNORECASE)
        if match:
            return match[-1].lower()
        return None
    
    def get_correct_answer(self, item: Dict) -> str:
        """
        Get the correct answer from the dataset item.
        
        Args:
            item: Dataset item with ground truth
            
        Returns:
            Correct answer in standardized format
        """
        if self.dataset_type == DatasetType.SCIEVAL:
            # SciEval answer format handling
            answer_value = item.get('answer', [''])[0].strip().upper()
            return answer_value
        
        elif self.dataset_type == DatasetType.NUMINA_MATH:
            # Extract answer from solution
            solution = item.get('answer', '')
            match = re.search(r'\\boxed\{((?:[^{}]|\{[^}]*})*)\}', solution)
            return match.group(1) if match else ''
        
        elif self.dataset_type == DatasetType.MULTIPLE_CHOICE:
            # Handle different multiple choice dataset formats
            if 'answerKey' in item:  # SciKnowEval
                return str(item.get('answerKey', '')).upper()
            elif 'answer' in item and isinstance(item.get('answer'), int):  # LogiQA
                answer_idx = int(item.get('answer', 0))
                return chr(65 + answer_idx)  # Convert 0->A, 1->B, etc.
            elif 'answer_letter' in item:
                return str(item.get('answer_letter', '')).upper()
            elif 'correct_answer' in item:  # TruthfulQA
                return str(item.get('correct_answer', '')).upper()
            elif 'Correct Option' in item:  # GPQA
                return str(item.get('Correct Option', '')).upper()
            else:
                return str(item.get('answer', '')).upper()
        
        elif self.dataset_type == DatasetType.LOGICNLI:
            return str(item.get('output', '')).lower()
        
        return ''
    
    def process_data(self, data: List[Dict]) -> Tuple[float, int, int]:
        """
        Process a batch of data to extract answers and calculate confidence.
        
        Args:
            data: List of data items with model outputs
            
        Returns:
            Tuple of (accuracy percentage, correct count, total count)
        """
        correct_count = 0
        total_count = len(data)
        
        # Process data in groups of responses_per_question
        for i in range(0, total_count, self.responses_per_question):
            group = data[i:i+self.responses_per_question]
            
            # Extract answers for the group
            extracted_answers = [self.extract_answer(item.get('model_output', '')) for item in group]
            valid_answers = [ans for ans in extracted_answers if ans]  # Filter out None values
            
            # Count answer frequencies
            answer_counts = Counter(valid_answers)
            
            # Process each item in the group
            for item in group:
                model_answer = self.extract_answer(item.get('model_output', ''))
                
                # Calculate confidence as percentage of matching answers in the group
                if model_answer and valid_answers:
                    model_confidence = (answer_counts.get(model_answer, 0) / len(valid_answers)) * 100
                else:
                    model_confidence = 0
                
                # Store results in the item
                item['model_answer'] = model_answer
                item['model_confidence'] = f"{model_confidence:.2f}%"
                
                # Check correctness
                correct_answer = self.get_correct_answer(item)
                is_correct = model_answer == correct_answer
                item['correct'] = is_correct
                
                if is_correct:
                    correct_count += 1
        
        # Calculate overall accuracy
        accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0
        return accuracy, correct_count, total_count


class FileProcessor:
    """
    Processes JSON files containing model outputs to extract answers 
    and calculate confidence scores.
    """
    
    @staticmethod
    def get_all_json_files(directory: str) -> List[str]:
        """
        Get all JSON files in a directory and its subdirectories.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of JSON file paths
        """
        json_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
        return json_files
    
    @staticmethod
    def read_json(file_path: str) -> Any:
        """
        Read JSON data from a file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Parsed JSON data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            logger.error(f"Error reading JSON file {file_path}: {e}")
            return []
    
    @staticmethod
    def write_json(data: Any, file_path: str) -> None:
        """
        Write JSON data to a file.
        
        Args:
            data: Data to write
            file_path: Output file path
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
            logger.info(f"Data written to {file_path}")
        except Exception as e:
            logger.error(f"Error writing JSON file {file_path}: {e}")
    
    @staticmethod
    def process_directory(directory: str, dataset_type: DatasetType, responses_per_question: int = 5) -> None:
        """
        Process all JSON files in a directory.
        
        Args:
            directory: Directory containing JSON files
            dataset_type: Type of dataset to process
            responses_per_question: Number of responses per question
        """
        logger.info(f"Processing directory: {directory}")
        json_files = FileProcessor.get_all_json_files(directory)
        logger.info(f"Found {len(json_files)} JSON files")
        
        extractor = AnswerExtractor(dataset_type, responses_per_question)
        
        for json_file in json_files:
            logger.info(f"Processing file: {json_file}")
            data = FileProcessor.read_json(json_file)
            
            # Skip metadata entry if present
            if (isinstance(data, list) and len(data) > 0 and 
                isinstance(data[0], dict) and 'accuracy' in data[0]):
                data = data[1:]
            
            # Process data
            accuracy, correct_count, total_count = extractor.process_data(data)
            
            # Create accuracy metadata
            accuracy_info = {
                "accuracy": f"{accuracy:.2f}%",
                "total": total_count,
                "correct": correct_count
            }
            
            logger.info(f"File: {os.path.basename(json_file)}, Accuracy: {accuracy:.2f}%, " +
                       f"Correct: {correct_count}/{total_count}")
            
            # Write processed data back to file
            # Uncomment this line if you want to add accuracy info at the beginning
            # updated_data = [accuracy_info] + data
            updated_data = data
            FileProcessor.write_json(updated_data, json_file)


def main():
    """Main entry point with command line argument handling."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Process model outputs to extract answers and calculate confidence")
    parser.add_argument("--dir", required=True, help="Directory containing JSON files")
    parser.add_argument("--type", required=True, choices=[t.value for t in DatasetType], 
                        help="Dataset type to process")
    parser.add_argument("--responses", type=int, default=5, 
                        help="Number of responses per question")
    
    args = parser.parse_args()
    
    FileProcessor.process_directory(
        directory=args.dir,
        dataset_type=DatasetType(args.type),
        responses_per_question=args.responses
    )


if __name__ == "__main__":
    main()