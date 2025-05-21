"""
Correctness-Only Pair Construction module for C2RM.

This module builds training pairs from model responses based only on correctness,
generating various combinations of pairs (1-2, 1-4, 3-2, 3-4) where:
- Class 1: Correct & High Confidence
- Class 2: Incorrect & High Confidence
- Class 3: Correct & Low Confidence 
- Class 4: Incorrect & Low Confidence

The pairing focuses primarily on correctness, with classes 1 and 3 (both correct) 
being used as positive examples, and classes 2 and 4 (both incorrect) as negative examples.
"""

import os
import json
import random
import logging
from typing import Dict, List, Tuple, Any, Set, Optional
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CorrectnessBasedPairConstructor:
    """
    Constructs training pairs from model responses based on correctness.
    
    Creates four types of pairs:
    - class_1 (Correct & Certain) vs class_2 (Incorrect & Certain)
    - class_1 (Correct & Certain) vs class_4 (Incorrect & Uncertain)
    - class_3 (Correct & Uncertain) vs class_2 (Incorrect & Certain)
    - class_3 (Correct & Uncertain) vs class_4 (Incorrect & Uncertain)
    """
    
    def __init__(self, responses_per_question: int = 5, seed: int = 42):
        """
        Initialize the pair constructor.
        
        Args:
            responses_per_question: Number of responses generated per question
            seed: Random seed for reproducibility
        """
        self.responses_per_question = responses_per_question
        self.seed = seed
        random.seed(seed)
    
    def load_data(self, file_path: str) -> List[Dict]:
        """
        Load data from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of data items
        """
        try:
            logger.info(f"Loading data from {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} items")
            return data
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return []
    
    def save_data(self, data: List[Dict], file_path: str) -> None:
        """
        Save data to a JSON file.
        
        Args:
            data: Data to save
            file_path: Output file path
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(data)} items to {file_path}")
        except Exception as e:
            logger.error(f"Error saving to {file_path}: {e}")
    
    def format_question(self, item: Dict, dataset_type: str) -> str:
        """
        Format a question based on dataset type.
        
        Args:
            item: Dataset item
            dataset_type: Type of dataset
            
        Returns:
            Formatted question string
        """
        if dataset_type == "numina_math":
            question = item.get("problem", "No question provided")
            return f"Question:\n{question}"
        
        elif dataset_type == "sciknoweval":
            question = item.get("question", "No question provided")
            choices_text = " ".join(
                [f"{label}. {text}" for label, text in zip(item["choices"]["label"], item["choices"]["text"])]
            )
            return f"Question:\n{question}\nOptions:\n{choices_text}"
        
        elif dataset_type == "scieval":
            question = item.get("question", "No question provided")
            return f"Question:\n{question}"
        
        elif dataset_type == "logicnli":
            premise = item.get("premise", "No question provided")
            hypothesis = item.get("hypothesis", "No question provided")
            return f"Question:\n{premise}\n\nHypothesis:\n{hypothesis}"
        
        elif dataset_type == "logiqa":
            text = item.get("text", "No background information provided")
            question = item.get("question", "No question provided")
            options = item.get("options", [])
            options_text = " ".join([f"{chr(65 + i)}. {option}" for i, option in enumerate(options)])
            return f"Background:\n{text}\n\nQuestion:\n{question}\nOptions:\n{options_text}"
        
        else:
            # Default format for other dataset types
            if "question" in item:
                return f"Question:\n{item['question']}"
            return f"Question:\n{item.get('problem', 'No question provided')}"
    
    def create_diverse_pairs(
        self, 
        pos_class_items: List[Dict], 
        neg_class_items: List[Dict],
        used_positive_outputs: Optional[Set[str]] = None,
        used_negative_outputs: Optional[Set[str]] = None
    ) -> Tuple[List[Dict], List[str], Set[str], Set[str]]:
        """
        Create diverse positive-negative pairs, attempting to use different examples.
        
        Args:
            pos_class_items: List of positive class items (class_1 or class_3)
            neg_class_items: List of negative class items (class_2 or class_4)
            used_positive_outputs: Set of already used positive outputs
            used_negative_outputs: Set of already used negative outputs
            
        Returns:
            Tuple of (pairs, pair_sources, updated_used_positive_outputs, updated_used_negative_outputs)
        """
        if used_positive_outputs is None:
            used_positive_outputs = set()
        
        if used_negative_outputs is None:
            used_negative_outputs = set()
        
        pairs = []
        pair_sources = []
        
        # If no positive or negative items, return empty results
        if not pos_class_items or not neg_class_items:
            return pairs, pair_sources, used_positive_outputs, used_negative_outputs
        
        # First try to use positive items that haven't been used yet
        available_pos_items = [item for item in pos_class_items if item["model_output"] not in used_positive_outputs]
        
        # If no unused positive items are available, use all positive items
        if not available_pos_items:
            available_pos_items = pos_class_items[:]
        
        # First try to use negative items that haven't been used yet
        available_neg_items = [item for item in neg_class_items if item["model_output"] not in used_negative_outputs]
        
        # If no unused negative items are available, use all negative items
        if not available_neg_items:
            available_neg_items = neg_class_items[:]
        
        # Choose one positive and one negative item
        pos_item = random.choice(available_pos_items)
        neg_item = random.choice(available_neg_items)
        
        # Track used outputs
        used_positive_outputs.add(pos_item["model_output"])
        used_negative_outputs.add(neg_item["model_output"])
        
        # Get source models
        chosen_source = pos_item.get("source_model", "unknown")
        rejected_source = neg_item.get("source_model", "unknown")
        
        # Create the pair
        pair = {
            "instruction": self.format_question(pos_item, self.dataset_type),
            "input": "",
            "chosen": pos_item["model_output"],
            "rejected": neg_item["model_output"],
        }
        
        pairs.append(pair)
        pair_sources.append(f"{chosen_source}-{rejected_source}")
        
        return pairs, pair_sources, used_positive_outputs, used_negative_outputs
    
    def generate_all_pairs_for_question(
        self, 
        class_1_items: List[Dict],
        class_2_items: List[Dict],
        class_3_items: List[Dict],
        class_4_items: List[Dict]
    ) -> Tuple[Dict[str, List[Dict]], Dict[str, List[str]]]:
        """
        Generate all possible pair types for a question.
        
        Args:
            class_1_items: List of class_1 items (Correct & Certain)
            class_2_items: List of class_2 items (Incorrect & Certain)
            class_3_items: List of class_3 items (Correct & Uncertain)
            class_4_items: List of class_4 items (Incorrect & Uncertain)
            
        Returns:
            Tuple of (all_pairs, all_sources)
        """
        # Initialize results
        all_pairs = {'12': [], '14': [], '32': [], '34': []}
        all_sources = {'12': [], '14': [], '32': [], '34': []}
        
        # Track used outputs separately
        used_positive_outputs = set()  # For both class_1 and class_3
        used_negative_outputs = {'2': set(), '4': set()}  # For class_2 and class_4
        
        # Generate class_1 vs class_2 pairs
        if class_1_items and class_2_items:
            pairs_12, sources_12, used_pos, used_neg_2 = self.create_diverse_pairs(
                class_1_items, class_2_items, used_positive_outputs, used_negative_outputs['2'])
            all_pairs['12'] = pairs_12
            all_sources['12'] = sources_12
            used_positive_outputs = used_pos
            used_negative_outputs['2'] = used_neg_2
        
        # Generate class_1 vs class_4 pairs
        if class_1_items and class_4_items:
            pairs_14, sources_14, used_pos, used_neg_4 = self.create_diverse_pairs(
                class_1_items, class_4_items, used_positive_outputs, used_negative_outputs['4'])
            all_pairs['14'] = pairs_14
            all_sources['14'] = sources_14
            used_positive_outputs = used_pos
            used_negative_outputs['4'] = used_neg_4
        
        # Generate class_3 vs class_2 pairs (class_3 as positive)
        if class_3_items and class_2_items:
            pairs_32, sources_32, used_pos, used_neg_2 = self.create_diverse_pairs(
                class_3_items, class_2_items, used_positive_outputs, used_negative_outputs['2'])
            all_pairs['32'] = pairs_32
            all_sources['32'] = sources_32
            used_positive_outputs = used_pos
            used_negative_outputs['2'] = used_neg_2
        
        # Generate class_3 vs class_4 pairs (class_3 as positive)
        if class_3_items and class_4_items:
            pairs_34, sources_34, used_pos, used_neg_4 = self.create_diverse_pairs(
                class_3_items, class_4_items, used_positive_outputs, used_negative_outputs['4'])
            all_pairs['34'] = pairs_34
            all_sources['34'] = sources_34
            used_positive_outputs = used_pos
            used_negative_outputs['4'] = used_neg_4
        
        return all_pairs, all_sources
    
    def match_question_across_models(self, llama_item: Dict, qwen_item: Dict, mistral_item: Dict) -> bool:
        """
        Check if questions match across different model outputs based on dataset type.
        
        Args:
            llama_item: Item from LLaMA model data
            qwen_item: Item from Qwen model data
            mistral_item: Item from Mistral model data
            
        Returns:
            Boolean indicating if questions match
        """
        if self.dataset_type == "numina_math":
            return (llama_item.get("problem", "") == 
                    qwen_item.get("problem", "") == 
                    mistral_item.get("problem", ""))
        
        elif self.dataset_type == "sciknoweval" or self.dataset_type == "scieval":
            return (llama_item.get("question", "") == 
                    qwen_item.get("question", "") == 
                    mistral_item.get("question", ""))
        
        elif self.dataset_type == "logicnli":
            return (llama_item.get("premise", "") == qwen_item.get("premise", "") == mistral_item.get("premise", "") and
                   llama_item.get("hypothesis", "") == qwen_item.get("hypothesis", "") == mistral_item.get("hypothesis", ""))
        
        elif self.dataset_type == "logiqa":
            return (llama_item.get("text", "") == qwen_item.get("text", "") == mistral_item.get("text", "") and
                   llama_item.get("question", "") == qwen_item.get("question", "") == mistral_item.get("question", ""))
        
        else:
            # Default check based on question field
            return (llama_item.get("question", "") == 
                    qwen_item.get("question", "") == 
                    mistral_item.get("question", ""))
    
    def construct_pairs(
        self, 
        model_data_files: Dict[str, str],
        output_files: Dict[str, str],
        dataset_type: str
    ) -> Dict[str, Any]:
        """
        Construct pairs from multiple model data files.
        
        Args:
            model_data_files: Dictionary mapping model names to data files
            output_files: Dictionary mapping pair types to output files
            dataset_type: Type of dataset
            
        Returns:
            Statistics about the constructed pairs
        """
        self.dataset_type = dataset_type
        
        # Load data from each model
        model_data = {}
        for model_name, file_path in model_data_files.items():
            data = self.load_data(file_path)
            
            # Add source model information to each item
            for item in data:
                item["source_model"] = model_name
            
            model_data[model_name] = data
        
        # Get the model names for convenient access
        model_names = list(model_data.keys())
        
        # Find the minimum length across all datasets
        min_length = min(len(data) for data in model_data.values())
        logger.info(f"Using {min_length} items from each model")
        
        # Initialize containers for pairs and statistics
        pairs = {
            '12': [],
            '14': [],
            '32': [],
            '34': []
        }
        
        sources = {
            '12': [],
            '14': [],
            '32': [],
            '34': []
        }
        
        # Process data in groups based on responses_per_question
        progress_bar = tqdm(range(0, min_length, self.responses_per_question), desc="Constructing pairs")
        for i in progress_bar:
            # Skip if we've reached the end of any dataset
            if i + self.responses_per_question > min_length:
                continue
            
            # Get the same question group from each model
            model_groups = {}
            for model_name in model_names:
                model_groups[model_name] = model_data[model_name][i:i+self.responses_per_question]
            
            # Check if questions match across models
            question_match = False
            if len(model_names) == 3:  # For three models
                question_match = self.match_question_across_models(
                    model_groups[model_names[0]][0],
                    model_groups[model_names[1]][0],
                    model_groups[model_names[2]][0]
                )
            
            # Skip if questions don't match
            if not question_match:
                logger.debug(f"Question mismatch at index {i}, skipping")
                continue
            
            # Combine all responses for this question
            all_items = []
            for group in model_groups.values():
                all_items.extend(group)
            
            # Categorize by class label
            class_items = {
                "class_1": [],
                "class_2": [],
                "class_3": [],
                "class_4": []
            }
            
            for item in all_items:
                label = item.get("class_label")
                if label in class_items:
                    class_items[label].append(item)
            
            # Generate all pair types for this question
            question_pairs, question_sources = self.generate_all_pairs_for_question(
                class_items["class_1"], class_items["class_2"], 
                class_items["class_3"], class_items["class_4"]
            )
            
            # Add pairs to results
            for pair_type in ['12', '14', '32', '34']:
                pairs[pair_type].extend(question_pairs[pair_type])
                sources[pair_type].extend(question_sources[pair_type])
        
        # Save pairs to output files
        for pair_type, file_path in output_files.items():
            if pair_type in pairs:
                self.save_data(pairs[pair_type], file_path)
        
        # Calculate statistics
        stats = self.calculate_statistics(sources, pairs)
        return stats
    
    def calculate_statistics(
        self, 
        sources: Dict[str, List[str]], 
        pairs: Dict[str, List[Dict]]
    ) -> Dict[str, Any]:
        """
        Calculate statistics about the constructed pairs.
        
        Args:
            sources: Dictionary of source model pairs for each class combination
            pairs: Dictionary of constructed pairs for each class combination
            
        Returns:
            Statistics dictionary
        """
        stats = {}
        
        # Count model combinations for each class pair type
        for pair_type, source_list in sources.items():
            if pair_type in ['12', '14', '32', '34']:
                combo_counts = self.count_model_combinations(source_list)
                
                pair_name = {
                    '12': "class_1_2",
                    '14': "class_1_4",
                    '32': "class_3_2",
                    '34': "class_3_4"
                }[pair_type]
                
                stats[pair_name] = {
                    "total": len(pairs[pair_type]),
                    "combinations": combo_counts
                }
        
        # Add total pair count
        total_pairs = sum(len(p) for p in pairs.values())
        stats["total_pairs"] = total_pairs
        
        return stats
    
    def count_model_combinations(self, source_list: List[str]) -> Dict[str, int]:
        """
        Count the occurrences of model combinations in source list.
        
        Args:
            source_list: List of model source combinations (e.g., ["llama-qwen", "mistral-llama"])
            
        Returns:
            Dictionary with counts for each combination type
        """
        # Initialize counters
        combinations = {
            "llama-qwen": 0, "qwen-llama": 0,
            "llama-mistral": 0, "mistral-llama": 0,
            "qwen-mistral": 0, "mistral-qwen": 0,
            "other": 0
        }
        
        # Count occurrences
        for source in source_list:
            if source in combinations:
                combinations[source] += 1
            else:
                combinations["other"] += 1
        
        # Group bidirectional combinations (e.g., llama-qwen and qwen-llama)
        simplified = {
            "llama-qwen": combinations["llama-qwen"] + combinations["qwen-llama"],
            "llama-mistral": combinations["llama-mistral"] + combinations["mistral-llama"],
            "qwen-mistral": combinations["qwen-mistral"] + combinations["mistral-qwen"],
            "other": combinations["other"]
        }
        
        return simplified


def main():
    """Main entry point with command line argument handling."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Construct training pairs based on correctness from classified model responses"
    )
    parser.add_argument("--dataset_type", required=True, 
                        choices=["numina_math", "sciknoweval", "scieval", "logicnli", "logiqa", "other"],
                        help="Type of dataset being processed")
    parser.add_argument("--model1_file", required=True, help="Path to first model's data file")
    parser.add_argument("--model2_file", required=True, help="Path to second model's data file")
    parser.add_argument("--model3_file", required=True, help="Path to third model's data file")
    parser.add_argument("--model1_name", default="model1", help="Name for first model")
    parser.add_argument("--model2_name", default="model2", help="Name for second model")
    parser.add_argument("--model3_name", default="model3", help="Name for third model")
    parser.add_argument("--output_dir", required=True, help="Output directory for pair files")
    parser.add_argument("--responses_per_question", type=int, default=5, 
                        help="Number of responses generated per question")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Construct output file paths
    output_files = {
        '12': os.path.join(args.output_dir, "conf_dpo_12.json"),
        '14': os.path.join(args.output_dir, "conf_dpo_14.json"),
        '32': os.path.join(args.output_dir, "conf_dpo_32.json"),
        '34': os.path.join(args.output_dir, "conf_dpo_34.json")
    }
    
    # Set up model data files
    model_data_files = {
        args.model1_name: args.model1_file,
        args.model2_name: args.model2_file,
        args.model3_name: args.model3_file
    }
    
    # Create pair constructor and build pairs
    constructor = CorrectnessBasedPairConstructor(
        responses_per_question=args.responses_per_question,
        seed=args.seed
    )
    
    stats = constructor.construct_pairs(
        model_data_files=model_data_files,
        output_files=output_files,
        dataset_type=args.dataset_type
    )
    
    # Save statistics
    stats_file = os.path.join(args.output_dir, "pair_statistics.json")
    constructor.save_data(stats, stats_file)
    
    # Print summary
    logger.info("\nPair construction summary:")
    logger.info(f"Class 1-2 pairs: {stats['class_1_2']['total']}")
    logger.info(f"Class 1-4 pairs: {stats['class_1_4']['total']}")
    logger.info(f"Class 3-2 pairs: {stats['class_3_2']['total']}")
    logger.info(f"Class 3-4 pairs: {stats['class_3_4']['total']}")
    logger.info(f"Total pairs: {stats['total_pairs']}")
    logger.info(f"Statistics saved to: {stats_file}")


if __name__ == "__main__":
    main()