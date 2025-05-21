# Beyond Correctness: Confidence-Aware Reward Modeling for Enhancing Large Language Model Reasoning

## Introduction

This repository contains the code and implementation details for our paper **"Beyond Correctness: Confidence-Aware Reward Modeling for Enhancing Large Language Model Reasoning"**.

Recent advancements in large language models (LLMs) have shifted focus toward reinforcement learning methods that enhance reasoning capabilities. However, conventional rule-based reward approaches often lead to poor-quality reasoning chains or inconsistencies between reasoning processes and final answers, especially with smaller-scale models.

Our work proposes a novel **Confidence-based Reward Model (C2RM)** tailored for enhancing STEM reasoning capabilities. Unlike conventional approaches that only consider correctness, our model also accounts for confidence expressed in model responses, penalizing both incorrect answers and low-confidence correct responses. This approach promotes more robust and logically consistent reasoning.

## Methodology Overview

![Methodology Overview](method_diagram.png)

Our approach consists of several steps:
1. Data collection from various STEM datasets
2. Filtering to identify partially correct response sets
3. Multi-model inference to generate diverse responses
4. Response classification based on correctness and confidence
5. Pair construction for training data preparation
6. Reward model training

## Data Preparation Pipeline

The data preparation workflow involves the following steps:

### 1. Model Inference

We first use Qwen2.5-72B-Instruct to generate responses for selected datasets. This serves as our initial filtering step.

```bash
python src/model_inference.py \
  --model_name "qwen2.5-72b-instruct" \
  --dataset "datasets/SciEval/train.json" \
  --output_dir "data/results/SciEval/qwen_72b" \
  --num_responses 5
```

### 2. Answer Extraction & Dataset Filtering

We extract answers and correctness rates, focusing on datasets with accuracy between 40%-70%.

```bash
python src/answer_extraction.py \
  --input_file "data/results/SciEval/qwen_72b.json" \
  --output_file "data/processed/SciEval/qwen_72b_extracted.json"
```

### 3. Partial Correctness Extraction

We identify questions where only some of Qwen2.5-72B-Instruct's 5 answers are correct.

```bash
python src/partial_correct_extractor.py \
  --input "data/processed/SciEval/qwen_72b_extracted.json" \
  --output "data/partial/SciEval/qwen_72b_partial_correct.json"
```

### 4. Multi-Model Inference

For the partially correct questions, we collect responses from multiple models (LLaMA-3 and Mistral) to create a diverse dataset.

```bash
python src/model_inference.py \
  --model_name "llama3_1_8b" \
  --dataset "data/partial/SciEval/qwen_72b_partial_correct.json" \
  --output_dir "data/results/SciEval/llama3_1_8b"

python src/model_inference.py \
  --model_name "mistral87" \
  --dataset "data/partial/SciEval/qwen_72b_partial_correct.json" \
  --output_dir "data/results/SciEval/mistral87"
```

### 5. Response Classification

We classify all model responses based on:

- **Correctness**: Whether the answer is correct  
- **Confidence**: The level of certainty expressed in the answer  

This creates 4 classes:
- Class 1: Correct & High Confidence (True & Certain, T&C)  
- Class 2: Incorrect & High Confidence (False & Certain, F&C)  
- Class 3: Correct & Low Confidence (True & Uncertain, T&U)  
- Class 4: Incorrect & Low Confidence (False & Uncertain, F&U)  

```bash
python src/data_classification.py \
  --input "data/results/SciEval/llama3_1_8b.json" \
  --threshold 50.0

python src/data_classification.py \
  --input "data/results/SciEval/qwen_72b.json" \
  --threshold 50.0

python src/data_classification.py \
  --input "data/results/SciEval/mistral87.json" \
  --threshold 50.0
```

### 6. Pair Construction

#### 6.1 Standard Pair Construction (Main Approach)

We construct preference pairs focusing on both correctness and confidence.

```bash
python src/pair_construction.py \
  --dataset_type "scieval" \
  --model1_file "data/results/SciEval/llama3_1_8b.json" \
  --model2_file "data/results/SciEval/qwen_72b.json" \
  --model3_file "data/results/SciEval/mistral87.json" \
  --model1_name "llama" \
  --model2_name "qwen" \
  --model3_name "mistral" \
  --output_dir "data/pairs/SciEval/main"
```

#### 6.2 Confidence-Only Pairs (Ablation Study)

```bash
python src/pair_construction_confidence_only.py \
  --dataset_type "scieval" \
  --model1_file "data/results/SciEval/llama3_1_8b.json" \
  --model2_file "data/results/SciEval/qwen_72b.json" \
  --model3_file "data/results/SciEval/mistral87.json" \
  --model1_name "llama" \
  --model2_name "qwen" \
  --model3_name "mistral" \
  --output_dir "data/pairs/SciEval/conf"
```

#### 6.3 Correctness-Only Pairs (Ablation Study)

```bash
python src/pair_construction_correctness_only.py \
  --dataset_type "scieval" \
  --model1_file "data/results/SciEval/llama3_1_8b.json" \
  --model2_file "data/results/SciEval/qwen_72b.json" \
  --model3_file "data/results/SciEval/mistral87.json" \
  --model1_name "llama" \
  --model2_name "qwen" \
  --model3_name "mistral" \
  --output_dir "data/pairs/SciEval/tf"
```

### 7. Format Conversion for Training

```bash
python src/format.py \
  --input "data/pairs/SciEval/main/pair_12.json" \
  --output "data/training/SciEval/sft_12.json"
```

## Code Overview

This repository includes the following main components:

- `model_inference.py`: Performs inference with specified language models  
- `answer_extraction.py`: Extracts answers and correctness rates from model outputs  
- `partial_correct_extractor.py`: Identifies questions with partially correct answers  
- `data_classification.py`: Classifies responses based on correctness and confidence  
- `pair_construction.py`: Constructs preference pairs for reward model training  
- `pair_construction_confidence_only.py`: Ablation study with confidence-only signals  
- `pair_construction_correctness_only.py`: Ablation study with correctness-only signals  
- `format.py`: Converts pairs to the format needed for reward model training  

## Dataset and Model Availability

All model outputs, processed data, and reward model checkpoints have been uploaded to HuggingFace.  
However, since our paper is under review at ARR and follows anonymous submission guidelines, we are currently unable to share the direct links.
