"""
python src/inference/model_inference.py \
    --model /path/to/models/Qwen2.5-72B-Instruct \
    --input data/raw/LogiQA/train.jsonl \
    --output data/results/qwen_72b_logiqa.json \
    --dataset_type logiqa \
    --gpu_ids 0,1,2,3 \
    --batch_size 500 \
    --num_samples 5 \
    --temperature 0.7
"""
import os
import json
import argparse
from typing import List, Dict, Tuple, Any
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetProcessor:
    """Processes different dataset formats for model inference."""
    
    @staticmethod
    def process_logiqa(item: Dict) -> str:
        """Process LogiQA dataset format."""
        prompt_description = (
            "Read the question, analyze step by step and provide your answer. "
            "Use the following format to answer:\n"
            "Explanation: [insert step-by-step analysis here]\n"
            "Answer: [ONLY the A/B/C/D...; not a complete sentence]\n\n"
            "Only give me the reply according to this format, don't give me any other words. "
            "Please make sure to analyze step by step before giving the answer."
        )
        
        text = item.get("text", "No background information provided")
        question = item.get("question", "No question provided")
        options = item.get("options", [])

        options_text = " ".join([f"{chr(65 + i)}. {option}" for i, option in enumerate(options)])
        
        return f"{prompt_description}\n\nBackground:\n{text}\n\nQuestion:\n{question}\nOptions:\n{options_text}"
    
    @staticmethod
    def process_sciknoweval(item: Dict) -> str:
        """Process SciKnowEval dataset format."""
        prompt_description = (
            "Read the question, analyze step by step and provide your answer. "
            "Use the following format to answer:\n"
            "Explanation: [insert step-by-step analysis here]\n"
            "Answer: [ONLY the A/B/C/D...; not a complete sentence]\n\n"
            "Only give me the reply according to this format, don't give me any other words. "
            "Please make sure to analyze step by step before giving the answer."
        )
        
        question = item.get("question", "No question provided")
        choices_text = " ".join(
            [f"{label}. {text}" for label, text in zip(item["choices"]["label"], item["choices"]["text"])]
        )
        
        return f"{prompt_description}\n\nQuestion:\n{question}\nOptions:\n{choices_text}"
    
    @staticmethod
    def process_scieval(item: Dict) -> str:
        """Process SciEval dataset format."""
        prompt_description = (
            "Read the question, analyze step by step and provide your answer. "
            "Use the following format to answer:\n"
            "Explanation: [insert step-by-step analysis here]\n"
            "Answer: [ONLY the final answer; not a complete sentence]\n\n"
            "Only give me the reply according to this format, don't give me any other words. "
            "Please make sure to analyze step by step before giving the answer."
        )
        
        question = item.get("question", "No question provided")
        return f"{prompt_description}\n\nQuestion:\n{question}"
    
    @staticmethod
    def process_numina_math(item: Dict) -> str:
        """Process NuminaMath-TIR dataset format."""
        prompt_description = (
            "Read the question, analyze step by step and provide your answer. "
            "Use the following format to answer:\n"
            "Explanation: [insert step-by-step analysis here]\n"
            "Answer: [ONLY the numerical number be enclosed within \\boxed{}; not a complete sentence]\n\n"
            "Only give me the reply according to this format, don't give me any other words. "
            "Please make sure to analyze step by step before giving the answer."
        )
        
        question = item.get("problem", "No question provided")
        return f"{prompt_description}\n\nQuestion:\n{question}"
    
    @staticmethod
    def process_logicnli(item: Dict) -> str:
        """Process LogicNLI dataset format."""
        prompt_description = (
            "Please determine whether the hypothesis is entailment/neutral/self_contradiction/self-contradiction/contradiction "
            "based on these premises. Read the question, analyze step by step and provide your answer. "
            "Use the following format to answer:\n"
            "Explanation: [insert step-by-step analysis here]\n"
            "Answer: [ONLY the entailment/neutral/self_contradiction/self-contradiction/contradiction; not a complete sentence]\n\n"
            "Only give me the reply according to this format, don't give me any other words. "
            "Please make sure to analyze step by step and give me your evidence before giving the answer."
        )
        
        premise = item.get("premise", "No question provided")
        hypothesis = item.get("hypothesis", "No question provided")
        
        return f"{prompt_description}\n\nQuestion:\n{premise}\n\nHypothesis:\n{hypothesis}"


class ModelInference:
    """Handles batch inference using vLLM for multiple datasets."""
    
    def __init__(
        self, 
        model_path: str,
        gpu_ids: str = "0,1,2,3",
        batch_size: int = 500,
        num_samples: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ):
        """
        Initialize the ModelInference class.
        
        Args:
            model_path: Path to the model
            gpu_ids: Comma-separated GPU IDs to use
            batch_size: Batch size for inference
            num_samples: Number of samples to generate per question
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
        """
        # Set GPU devices
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        
        self.model_path = model_path
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize model and tokenizer
        logger.info(f"Loading model from {model_path}")
        self.model = LLM(
            model_path, 
            dtype='bfloat16', 
            gpu_memory_utilization=0.8, 
            tensor_parallel_size=len(gpu_ids.split(',')),
            max_num_seqs=256,
            enforce_eager=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    def load_data(self, input_file: str) -> List[Dict]:
        """
        Load data from a JSONL file.
        
        Args:
            input_file: Path to the input file
            
        Returns:
            List of data items
        """
        logger.info(f"Loading data from {input_file}")
        data = []
        with open(input_file, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
        logger.info(f"Loaded {len(data)} items")
        return data
    
    def generate_batches(
        self, 
        data: List[Dict], 
        process_function
    ) -> Tuple[List[List[List[Dict]]], int]:
        """
        Generate message batches for inference.
        
        Args:
            data: List of data items
            process_function: Function to process each item
            
        Returns:
            Tuple of (batches, number of batches)
        """
        all_batches = []
        num_batches = (len(data) + self.batch_size - 1) // self.batch_size

        for i in range(num_batches):
            start_index = i * self.batch_size
            end_index = min(start_index + self.batch_size, len(data))
            data_batch = data[start_index:end_index]

            batch_messages = []
            for item in data_batch:
                instruction = process_function(item)
                message = [{"role": "user", "content": instruction}]
                batch_messages.append(message)

            all_batches.append(batch_messages)

        return all_batches, num_batches
    
    def run_inference(
        self,
        input_file: str,
        output_file: str,
        dataset_type: str
    ) -> None:
        """
        Run inference on a dataset.
        
        Args:
            input_file: Path to the input file
            output_file: Path to the output file
            dataset_type: Type of dataset (logiqa, sciknoweval, etc.)
        """
        # Select the appropriate processing function based on dataset type
        process_functions = {
            "logiqa": DatasetProcessor.process_logiqa,
            "sciknoweval": DatasetProcessor.process_sciknoweval,
            "scieval": DatasetProcessor.process_scieval,
            "numina_math": DatasetProcessor.process_numina_math,
            "logicnli": DatasetProcessor.process_logicnli
        }
        
        if dataset_type not in process_functions:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        
        process_function = process_functions[dataset_type]
        
        # Load data and generate batches
        data = self.load_data(input_file)
        messages_all, num_batches = self.generate_batches(data, process_function)
        
        # Set up sampling parameters
        sampling_params = SamplingParams(
            n=self.num_samples, 
            temperature=self.temperature, 
            max_tokens=self.max_tokens, 
            stop="<|eot_id|>"
        )
        
        # Run inference
        logger.info(f"Starting inference with {self.num_samples} samples per question")
        all_outputs = []
        
        for i in tqdm(range(num_batches), desc="Processing batches"):
            # Apply chat template and generate responses
            formatted_prompt = [
                self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                for messages in messages_all[i]
            ]
            
            batch_output = self.model.generate(formatted_prompt, sampling_params)
            
            # Process outputs
            for data_idx, output in enumerate(batch_output):
                item_idx = i * self.batch_size + data_idx
                
                # Process each generated sample
                for generated_idx, text_output in enumerate(output.outputs):
                    generated_text = text_output.text
                    result = {
                        **data[item_idx],
                        'model_output': generated_text,
                        'generated_idx': generated_idx + 1
                    }
                    all_outputs.append(result)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Write results to file
        logger.info(f"Writing {len(all_outputs)} results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(all_outputs, file, ensure_ascii=False, indent=4)
        
        logger.info("Inference complete")


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run model inference on datasets")
    
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--input", type=str, required=True, help="Path to the input dataset")
    parser.add_argument("--output", type=str, required=True, help="Path to the output file")
    parser.add_argument("--dataset_type", type=str, required=True, 
                        choices=["logiqa", "sciknoweval", "scieval", "numina_math", "logicnli"],
                        help="Type of dataset to process")
    parser.add_argument("--gpu_ids", type=str, default="0,1,2,3", help="Comma-separated GPU IDs")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size for inference")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples per question")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Maximum tokens to generate")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = get_args()
    
    # Initialize model inference
    inference = ModelInference(
        model_path=args.model,
        gpu_ids=args.gpu_ids,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        temperature=args.temperature,
        max_tokens=args.max_tokens
    )
    
    # Run inference
    inference.run_inference(
        input_file=args.input,
        output_file=args.output,
        dataset_type=args.dataset_type
    )


if __name__ == "__main__":
    main()