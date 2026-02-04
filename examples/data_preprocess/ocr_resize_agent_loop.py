# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess OCR dataset for verl Agentic RL training with resize tool.

Input format (JSON/JSONL):
{
    "img_path_sh": "/path/to/image.jpg",   # or "image"
    "groundtruth": "识别结果文本",           # or "ground_truth"
    "data_id": "text_block_000001",         # optional, used to infer tag
    "tag": "text"                           # optional: text/table/formula
}

Output format (Parquet):
{
    "data_source": "ocr_dataset",
    "agent_name": "tool_agent",
    "prompt": [...],
    "extra_info": {...},
    "reward_model": {...}
}

Usage:
    python ocr_resize_agent_loop.py \
        --input_path /path/to/dataset.json \
        --output_dir ~/data/ocr_verl \
        --train_ratio 0.9
"""

import argparse
import json
import os
from pathlib import Path

import datasets

from verl.utils.hdfs_io import copy, makedirs

# ============================================================================
# System prompts for different OCR tasks (consistent with SFT training data)
# ============================================================================

# Common tool instruction shared across all task types
TOOL_INSTRUCTION = """
You have access to a resize tool that can adjust the image resolution:
<tool_call>{"name": "resize", "arguments": {"scale": N}}</tool_call>
where N can be: 0.125, 0.143, 0.167, 0.2, 0.25, 0.333, 0.5, 2, 3, 4, 5, 6, 7, 8
- scale < 1: Decrease resolution. For example: scale=0.5 means half size, scale=0.125 means 1/8 size
- scale > 1: Increase resolution. For example: scale=2 means 2x larger, scale=8 means 8x larger

Different images may have different optimal resolutions for accurate recognition. Analyze the image and decide whether adjusting the resolution would help improve recognition accuracy.

When you complete the extraction, wrap your result with <final_answer> tags.
"""

# System prompts for each tag type
SYSTEM_PROMPTS = {
    "text": (
        "You are an expert OCR assistant. Your task is to accurately extract all text content from the given image.\n"
        + TOOL_INSTRUCTION
        + """
Output format:
- Output text in reading order (top to bottom, left to right)
- Preserve paragraph structure with newlines
- Keep list markers (-, ·, • etc.) and numbering
- Use LaTeX for math formulas: $x^2 + y^2 = z^2$
- Preserve all special characters and punctuation
"""
    ),
    "table": (
        "You are an expert OCR assistant. Your task is to accurately extract the table content from the given image and output it in HTML format.\n"
        + TOOL_INSTRUCTION
        + """
Output format:
- Use <table border="1"> as the opening tag
- Use <tr> for rows, <td> for cells
- Use rowspan/colspan for merged cells
- Use LaTeX for formulas: $x^2$
- Preserve line breaks with \\n
"""
    ),
    "equation": (
        "You are an expert OCR assistant. Your task is to accurately extract the mathematical equation from the given image and output it in LaTeX format.\n"
        + TOOL_INSTRUCTION
        + """
Output format:
- Output the equation in LaTeX format
- Use standard LaTeX math notation
- Preserve the exact structure of the equation
"""
    ),
    "formula": (
        "You are an expert OCR assistant. Your task is to accurately extract the mathematical equation from the given image and output it in LaTeX format.\n"
        + TOOL_INSTRUCTION
        + """
Output format:
- Output the equation in LaTeX format
- Use standard LaTeX math notation
- Preserve the exact structure of the equation
"""
    ),
}

# User prompts for each tag type
USER_PROMPTS = {
    "text": "Extract all text content from this image.",
    "table": "Extract the table content from this image in HTML format.",
    "equation": "Extract the mathematical equation from this image in LaTeX format.",
    "formula": "Extract the mathematical equation from this image in LaTeX format.",
}

# Default fallback for unknown tags
DEFAULT_SYSTEM_PROMPT = SYSTEM_PROMPTS["text"]
DEFAULT_USER_PROMPT = USER_PROMPTS["text"]


def infer_tag(data_id: str) -> str:
    """Infer data type tag from data_id.

    Args:
        data_id: The data identifier string

    Returns:
        Tag string: "equation", "table", or "text"
    """
    if data_id:
        data_id_lower = data_id.lower()
        if "equation" in data_id_lower or "formula" in data_id_lower:
            return "equation"
        elif "table" in data_id_lower:
            return "table"
    return "text"


def load_raw_dataset(data_path: str) -> list:
    """Load raw JSON/JSONL dataset.

    Args:
        data_path: Path to the dataset file

    Returns:
        List of data dictionaries

    Raises:
        ValueError: If file format is not supported
    """
    data = []
    path = Path(data_path)

    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            content = json.load(f)
            if isinstance(content, dict):
                data = [content]
            else:
                data = content
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}. Use .json or .jsonl")

    return data


def process_sample(example: dict, idx: int, split: str, data_source: str) -> dict:
    """Process a single sample to verl training format.

    Args:
        example: Raw data dictionary
        idx: Sample index
        split: Dataset split (train/test)
        data_source: Name of the data source

    Returns:
        Processed data dictionary in verl format

    Raises:
        ValueError: If required fields are missing
    """
    # Field mapping: support multiple naming conventions
    image_path = example.get("img_path_sh") or example.get("image") or example.get("image_path")
    ground_truth = example.get("groundtruth") or example.get("ground_truth") or example.get("label")
    data_id = example.get("data_id", f"{split}_{idx:06d}")
    tag = example.get("tag") or infer_tag(data_id)

    if not image_path:
        raise ValueError(f"Missing image path in sample {idx}. Expected 'img_path_sh', 'image', or 'image_path'")
    if ground_truth is None:
        raise ValueError(f"Missing ground truth in sample {idx}. Expected 'groundtruth', 'ground_truth', or 'label'")

    # Get tag-specific prompts (consistent with SFT training data)
    system_prompt = SYSTEM_PROMPTS.get(tag, DEFAULT_SYSTEM_PROMPT)
    user_prompt = USER_PROMPTS.get(tag, DEFAULT_USER_PROMPT)

    # Build verl training format
    # Note: Using "<image>" placeholder format consistent with SFT data
    return {
        "data_source": data_source,
        "agent_name": "tool_agent",  # Enable ToolAgentLoop
        # Provide images list for <image> placeholders (required by RLHFDataset._build_messages)
        "images": [
            {"image": image_path},
        ],
        "prompt": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"<image>{user_prompt}",
            },
        ],
        "ability": "ocr",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
            "tag": tag,
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "data_id": data_id,
            "tag": tag,
            "ground_truth": ground_truth,
            "image_path": image_path,
            "need_tools_kwargs": True,
            "tools_kwargs": {
                "resize": {
                    "create_kwargs": {"image": image_path},
                },
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess OCR dataset for verl Agentic RL training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a JSONL file
    python ocr_resize_agent_loop.py --input_path data.jsonl --output_dir ~/data/ocr_verl

    # Process with custom train/test split
    python ocr_resize_agent_loop.py --input_path data.json --train_ratio 0.8

    # Upload to HDFS
    python ocr_resize_agent_loop.py --input_path data.json --hdfs_dir hdfs://path/to/output
        """,
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input JSON/JSONL file path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="~/data/ocr_verl",
        help="Output directory for processed parquet files (default: ~/data/ocr_verl)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Train split ratio (default: 0.9)",
    )
    parser.add_argument(
        "--data_source",
        type=str,
        default="ocr_dataset",
        help="Data source name for tracking (default: ocr_dataset)",
    )
    parser.add_argument(
        "--hdfs_dir",
        type=str,
        default=None,
        help="Optional HDFS directory to copy processed data",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle data before splitting",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)",
    )

    args = parser.parse_args()

    # Load raw data
    print(f"Loading data from {args.input_path}...")
    raw_data = load_raw_dataset(args.input_path)
    print(f"Loaded {len(raw_data)} samples")

    # Shuffle if requested
    if args.shuffle:
        import random

        random.seed(args.seed)
        random.shuffle(raw_data)
        print(f"Shuffled data with seed {args.seed}")

    # Split into train/test
    split_idx = int(len(raw_data) * args.train_ratio)
    train_data = raw_data[:split_idx]
    test_data = raw_data[split_idx:]

    print(f"Split: {len(train_data)} train, {len(test_data)} test")

    # Process samples
    print("Processing train samples...")
    train_processed = []
    for i, ex in enumerate(train_data):
        try:
            processed = process_sample(ex, i, "train", args.data_source)
            train_processed.append(processed)
        except ValueError as e:
            print(f"Warning: Skipping train sample {i}: {e}")

    print("Processing test samples...")
    test_processed = []
    for i, ex in enumerate(test_data):
        try:
            processed = process_sample(ex, i, "test", args.data_source)
            test_processed.append(processed)
        except ValueError as e:
            print(f"Warning: Skipping test sample {i}: {e}")

    print(f"Processed: {len(train_processed)} train, {len(test_processed)} test")

    # Convert to HuggingFace Dataset
    train_dataset = datasets.Dataset.from_list(train_processed)
    test_dataset = datasets.Dataset.from_list(test_processed)

    # Count tags for statistics
    tag_counts = {"train": {}, "test": {}}
    for item in train_processed:
        tag = item["extra_info"]["tag"]
        tag_counts["train"][tag] = tag_counts["train"].get(tag, 0) + 1
    for item in test_processed:
        tag = item["extra_info"]["tag"]
        tag_counts["test"][tag] = tag_counts["test"].get(tag, 0) + 1

    print(f"Tag distribution - Train: {tag_counts['train']}, Test: {tag_counts['test']}")

    # Save to parquet
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")

    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

    print(f"Saved train dataset to {train_path}")
    print(f"Saved test dataset to {test_path}")

    # Copy to HDFS if specified
    if args.hdfs_dir:
        print(f"Copying to HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=output_dir, dst=args.hdfs_dir)
        print("HDFS copy complete")

    print("\nDone! Dataset is ready for verl training.")
    print(f"Use with: data.train_path={train_path}")


if __name__ == "__main__":
    main()
