# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
Unit tests for OCR Resize Agent data preprocessing script.

Tests cover:
- Tag inference from data_id
- Raw dataset loading (JSON/JSONL)
- Sample processing to verl format
- Output format validation
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add examples directory to path for import
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES_DIR / "data_preprocess"))

from ocr_resize_agent_loop import (
    SYSTEM_PROMPTS,
    USER_PROMPTS,
    infer_tag,
    load_raw_dataset,
    process_sample,
)


class TestInferTag:
    """Test infer_tag function."""

    def test_infer_formula_from_equation(self):
        """Test inferring formula tag from data_id containing 'equation'."""
        assert infer_tag("equation_isolated_000020") == "equation"
        assert infer_tag("EQUATION_123") == "equation"
        assert infer_tag("some_equation_data") == "equation"

    def test_infer_formula_from_formula(self):
        """Test inferring formula tag from data_id containing 'formula'."""
        assert infer_tag("formula_block_001") == "equation"
        assert infer_tag("FORMULA_TEST") == "equation"

    def test_infer_table_tag(self):
        """Test inferring table tag from data_id containing 'table'."""
        assert infer_tag("table_block_000001") == "table"
        assert infer_tag("TABLE_DATA_123") == "table"
        assert infer_tag("some_table_item") == "table"

    def test_infer_text_as_default(self):
        """Test that text is the default tag."""
        assert infer_tag("text_block_000001") == "text"
        assert infer_tag("random_id_123") == "text"
        assert infer_tag("image_001") == "text"
        assert infer_tag("") == "text"
        assert infer_tag(None) == "text"

    def test_infer_case_insensitive(self):
        """Test that tag inference is case insensitive."""
        assert infer_tag("EQUATION_UPPER") == "equation"
        assert infer_tag("Table_Mixed") == "table"
        assert infer_tag("formula_lower") == "equation"


class TestLoadRawDataset:
    """Test load_raw_dataset function."""

    def test_load_json_single_object(self):
        """Test loading JSON file with single object."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"img_path_sh": "/path/to/image.jpg", "groundtruth": "text"}, f)
            temp_path = f.name

        try:
            data = load_raw_dataset(temp_path)
            assert len(data) == 1
            assert data[0]["img_path_sh"] == "/path/to/image.jpg"
            assert data[0]["groundtruth"] == "text"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_json_array(self):
        """Test loading JSON file with array of objects."""
        samples = [
            {"img_path_sh": "/path/1.jpg", "groundtruth": "text1"},
            {"img_path_sh": "/path/2.jpg", "groundtruth": "text2"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(samples, f)
            temp_path = f.name

        try:
            data = load_raw_dataset(temp_path)
            assert len(data) == 2
            assert data[0]["groundtruth"] == "text1"
            assert data[1]["groundtruth"] == "text2"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_jsonl(self):
        """Test loading JSONL file."""
        samples = [
            {"img_path_sh": "/path/1.jpg", "groundtruth": "text1"},
            {"img_path_sh": "/path/2.jpg", "groundtruth": "text2"},
            {"img_path_sh": "/path/3.jpg", "groundtruth": "text3"},
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
            temp_path = f.name

        try:
            data = load_raw_dataset(temp_path)
            assert len(data) == 3
            assert data[2]["groundtruth"] == "text3"
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_jsonl_with_empty_lines(self):
        """Test loading JSONL file with empty lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"img_path_sh": "/path/1.jpg", "groundtruth": "text1"}\n')
            f.write("\n")  # empty line
            f.write('{"img_path_sh": "/path/2.jpg", "groundtruth": "text2"}\n')
            f.write("   \n")  # whitespace line
            temp_path = f.name

        try:
            data = load_raw_dataset(temp_path)
            assert len(data) == 2
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_unsupported_format(self):
        """Test that unsupported file format raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("some text")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                load_raw_dataset(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestProcessSample:
    """Test process_sample function."""

    def test_process_basic_sample(self):
        """Test processing a basic sample."""
        example = {
            "img_path_sh": "/path/to/image.jpg",
            "groundtruth": "识别结果",
            "data_id": "text_block_000001",
        }

        result = process_sample(example, idx=0, split="train", data_source="test_source")

        # Check required fields
        assert result["data_source"] == "test_source"
        assert result["agent_name"] == "tool_agent"
        assert result["ability"] == "ocr"

        # Check prompt structure
        assert len(result["prompt"]) == 2
        assert result["prompt"][0]["role"] == "system"
        assert result["prompt"][0]["content"] == SYSTEM_PROMPTS["text"]
        assert result["prompt"][1]["role"] == "user"
        assert result["prompt"][1]["content"] == f"<image>{USER_PROMPTS['text']}"

        # Check reward_model
        assert result["reward_model"]["style"] == "rule"
        assert result["reward_model"]["ground_truth"] == "识别结果"
        assert result["reward_model"]["tag"] == "text"

        # Check extra_info
        assert result["extra_info"]["split"] == "train"
        assert result["extra_info"]["index"] == 0
        assert result["extra_info"]["data_id"] == "text_block_000001"
        assert result["extra_info"]["tag"] == "text"
        assert result["extra_info"]["ground_truth"] == "识别结果"
        assert result["extra_info"]["image_path"] == "/path/to/image.jpg"
        assert result["extra_info"]["need_tools_kwargs"] is True
        assert result["extra_info"]["tools_kwargs"]["resize"]["create_kwargs"]["image"] == "/path/to/image.jpg"

    def test_process_sample_with_explicit_tag(self):
        """Test processing sample with explicit tag."""
        example = {
            "img_path_sh": "/path/to/table.jpg",
            "groundtruth": "<table>...</table>",
            "tag": "table",
        }

        result = process_sample(example, idx=1, split="test", data_source="test")

        assert result["reward_model"]["tag"] == "table"
        assert result["extra_info"]["tag"] == "table"

    def test_process_sample_infers_tag_from_data_id(self):
        """Test that tag is inferred from data_id when not explicitly provided."""
        example = {
            "img_path_sh": "/path/to/formula.jpg",
            "groundtruth": "$$x^2$$",
            "data_id": "equation_isolated_000020",
        }

        result = process_sample(example, idx=0, split="train", data_source="test")

        assert result["reward_model"]["tag"] == "equation"
        assert result["extra_info"]["tag"] == "equation"

    def test_process_sample_alternative_field_names(self):
        """Test processing sample with alternative field names."""
        # Using 'image' instead of 'img_path_sh'
        example1 = {
            "image": "/path/to/image1.jpg",
            "ground_truth": "text1",
        }
        result1 = process_sample(example1, idx=0, split="train", data_source="test")
        assert result1["extra_info"]["image_path"] == "/path/to/image1.jpg"
        assert result1["reward_model"]["ground_truth"] == "text1"

        # Using 'image_path' and 'label'
        example2 = {
            "image_path": "/path/to/image2.jpg",
            "label": "text2",
        }
        result2 = process_sample(example2, idx=1, split="train", data_source="test")
        assert result2["extra_info"]["image_path"] == "/path/to/image2.jpg"
        assert result2["reward_model"]["ground_truth"] == "text2"

    def test_process_sample_missing_image_raises_error(self):
        """Test that missing image path raises ValueError."""
        example = {
            "groundtruth": "text",
        }

        with pytest.raises(ValueError, match="Missing image path"):
            process_sample(example, idx=0, split="train", data_source="test")

    def test_process_sample_missing_ground_truth_raises_error(self):
        """Test that missing ground truth raises ValueError."""
        example = {
            "img_path_sh": "/path/to/image.jpg",
        }

        with pytest.raises(ValueError, match="Missing ground truth"):
            process_sample(example, idx=0, split="train", data_source="test")

    def test_process_sample_generates_data_id(self):
        """Test that data_id is generated when not provided."""
        example = {
            "img_path_sh": "/path/to/image.jpg",
            "groundtruth": "text",
        }

        result = process_sample(example, idx=42, split="test", data_source="test")

        assert result["extra_info"]["data_id"] == "test_000042"

    def test_process_sample_with_empty_ground_truth(self):
        """Test processing sample with empty ground truth (should raise)."""
        example = {
            "img_path_sh": "/path/to/image.jpg",
            "groundtruth": "",
        }

        with pytest.raises(ValueError, match="Missing ground truth"):
            process_sample(example, idx=0, split="train", data_source="test")


class TestSystemAndUserPrompts:
    """Test the system and user prompts."""

    def test_system_prompt_contains_guidelines(self):
        """Test that system prompt contains necessary guidelines."""
        text_prompt = SYSTEM_PROMPTS["text"]
        assert "OCR" in text_prompt
        assert "resize" in text_prompt.lower()
        assert "tool" in text_prompt.lower()
        assert "scale" in text_prompt.lower()

    def test_user_prompt_is_instruction(self):
        """Test that user prompt is a clear instruction."""
        text_user_prompt = USER_PROMPTS["text"]
        assert "image" in text_user_prompt.lower()
        assert "text" in text_user_prompt.lower() or "extract" in text_user_prompt.lower()


class TestOutputFormat:
    """Test that output format is compatible with verl training."""

    def test_output_has_all_required_fields(self):
        """Test that output has all fields required by verl."""
        example = {
            "img_path_sh": "/path/to/image.jpg",
            "groundtruth": "text",
            "data_id": "test_001",
        }

        result = process_sample(example, idx=0, split="train", data_source="test")

        # Required top-level fields
        required_fields = ["data_source", "agent_name", "prompt", "reward_model", "extra_info"]
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    def test_prompt_structure_is_valid(self):
        """Test that prompt structure is valid for chat models."""
        example = {
            "img_path_sh": "/path/to/image.jpg",
            "groundtruth": "text",
        }

        result = process_sample(example, idx=0, split="train", data_source="test")

        # Check prompt is a list
        assert isinstance(result["prompt"], list)

        # Check each message has role and content
        for msg in result["prompt"]:
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ["system", "user", "assistant"]

    def test_tools_kwargs_structure(self):
        """Test that tools_kwargs has correct structure for ImageResizeTool."""
        example = {
            "img_path_sh": "/path/to/image.jpg",
            "groundtruth": "text",
        }

        result = process_sample(example, idx=0, split="train", data_source="test")

        # Check tools_kwargs structure
        tools_kwargs = result["extra_info"]["tools_kwargs"]
        assert "resize" in tools_kwargs
        assert "create_kwargs" in tools_kwargs["resize"]
        assert "image" in tools_kwargs["resize"]["create_kwargs"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
