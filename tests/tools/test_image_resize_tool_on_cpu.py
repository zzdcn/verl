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
Unit tests for ImageResizeTool.

Tests cover:
- Tool initialization with various configs
- Instance creation and release
- Image resize operations (scale up/down)
- Error handling for invalid parameters
- Reward calculation
"""

import asyncio
import io
import tempfile
from pathlib import Path

import pytest
from PIL import Image
from qwen_vl_utils import fetch_image

from verl.tools.image_resize_tool import ImageResizeTool
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
)


def create_test_image(width: int = 100, height: int = 100, color: str = "red") -> Image.Image:
    """Create a test image for testing."""
    return Image.new("RGB", (width, height), color)


def create_test_image_path(width: int = 100, height: int = 100, color: str = "red") -> str:
    """Create a test image file and return its path."""
    img = create_test_image(width, height, color)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f, format="PNG")
        return f.name


def get_fetch_image_size(image_path: str) -> tuple[int, int]:
    """Get the size after qwen_vl_utils fetch_image normalization."""
    return fetch_image({"image": image_path}).size


def get_default_tool_schema() -> OpenAIFunctionToolSchema:
    """Get the default tool schema for testing."""
    return OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="resize",
            description="Resize the image by a scale factor",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "scale": OpenAIFunctionPropertySchema(
                        type="number",
                        description="Scale factor for resizing",
                    )
                },
                required=["scale"],
            ),
        ),
    )


def get_default_config() -> dict:
    """Get the default config for testing."""
    return {
        "allowed_scales": [0.25, 0.5, 1.0, 2.0, 4.0],
        "interpolation": "LANCZOS",
        "step_reward": 0.0,
        "invalid_scale_penalty": -0.05,
        "max_resize_calls": 5,
    }


class TestImageResizeToolInit:
    """Test ImageResizeTool initialization."""

    def test_init_with_default_config(self):
        """Test initialization with default config values."""
        config = {}
        schema = get_default_tool_schema()
        tool = ImageResizeTool(config, schema)

        assert tool.allowed_scales == [0.25, 0.5, 1.0, 2.0, 4.0]
        assert tool.interpolation == "LANCZOS"
        assert tool.step_reward == 0.0
        assert tool.invalid_scale_penalty == -0.05
        assert tool.max_resize_calls == 5
        assert tool.name == "resize"

    def test_init_with_custom_config(self):
        """Test initialization with custom config values."""
        config = {
            "allowed_scales": [0.5, 1.0, 1.5, 2.0],
            "interpolation": "BILINEAR",
            "step_reward": 0.01,
            "invalid_scale_penalty": -0.1,
            "max_resize_calls": 3,
        }
        schema = get_default_tool_schema()
        tool = ImageResizeTool(config, schema)

        assert tool.allowed_scales == [0.5, 1.0, 1.5, 2.0]
        assert tool.interpolation == "BILINEAR"
        assert tool.step_reward == 0.01
        assert tool.invalid_scale_penalty == -0.1
        assert tool.max_resize_calls == 3

    def test_init_with_invalid_interpolation_fallback(self):
        """Test that invalid interpolation method falls back to LANCZOS."""
        config = {"interpolation": "INVALID_METHOD"}
        schema = get_default_tool_schema()
        tool = ImageResizeTool(config, schema)

        # Should fall back to LANCZOS
        assert tool._resample_method == Image.Resampling.LANCZOS


class TestImageResizeToolCreate:
    """Test ImageResizeTool.create() method."""

    @pytest.fixture
    def tool(self):
        """Create a tool instance for testing."""
        return ImageResizeTool(get_default_config(), get_default_tool_schema())

    @pytest.fixture
    def test_image_path(self):
        """Create a test image file."""
        path = create_test_image_path(200, 150, "blue")
        yield path
        Path(path).unlink(missing_ok=True)

    def test_create_with_image_path(self, tool, test_image_path):
        """Test creating an instance with an image file path."""
        instance_id, response = asyncio.run(tool.create(image=test_image_path))

        assert instance_id is not None
        assert len(instance_id) > 0
        assert response.is_empty()

        # Check instance was stored
        stats = tool.get_instance_stats(instance_id)
        assert stats is not None
        expected_size = get_fetch_image_size(test_image_path)
        assert stats["original_size"] == expected_size
        assert stats["current_size"] == expected_size
        assert stats["resize_count"] == 0

    def test_create_with_pil_image(self, tool):
        """Test creating an instance with a PIL Image object."""
        img = create_test_image(300, 200, "green")

        # Save to temp file first since ImageResizeTool uses fetch_image
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f, format="PNG")
            temp_path = f.name

        try:
            instance_id, response = asyncio.run(tool.create(image=temp_path))
            assert instance_id is not None
            stats = tool.get_instance_stats(instance_id)
            expected_size = get_fetch_image_size(temp_path)
            assert stats["original_size"] == expected_size
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_create_with_custom_instance_id(self, tool, test_image_path):
        """Test creating an instance with a custom instance_id."""
        custom_id = "my-custom-id-123"
        instance_id, response = asyncio.run(tool.create(instance_id=custom_id, image=test_image_path))

        assert instance_id == custom_id
        assert tool.get_instance_stats(custom_id) is not None

    def test_create_with_create_kwargs(self, tool, test_image_path):
        """Test creating an instance with create_kwargs parameter."""
        instance_id, response = asyncio.run(tool.create(create_kwargs={"image": test_image_path}))

        assert instance_id is not None
        stats = tool.get_instance_stats(instance_id)
        assert stats is not None

    def test_create_without_image_raises_error(self, tool):
        """Test that creating without image parameter raises ValueError."""
        with pytest.raises(ValueError, match="Missing required 'image' parameter"):
            asyncio.run(tool.create())


class TestImageResizeToolExecute:
    """Test ImageResizeTool.execute() method."""

    @pytest.fixture
    def tool(self):
        """Create a tool instance for testing."""
        return ImageResizeTool(get_default_config(), get_default_tool_schema())

    @pytest.fixture
    def instance_with_image(self, tool):
        """Create an instance with a test image."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = create_test_image(100, 100, "red")
            img.save(f, format="PNG")
            temp_path = f.name

        instance_id, _ = asyncio.run(tool.create(image=temp_path))
        yield instance_id, temp_path
        Path(temp_path).unlink(missing_ok=True)
        asyncio.run(tool.release(instance_id))

    def test_execute_scale_up(self, tool, instance_with_image):
        """Test scaling up an image."""
        instance_id, _ = instance_with_image
        original_size = tool.get_instance_stats(instance_id)["current_size"]
        expected_new_size = (int(original_size[0] * 2.0), int(original_size[1] * 2.0))

        response, reward, metrics = asyncio.run(tool.execute(instance_id, {"scale": 2.0}))

        assert metrics["success"] is True
        assert metrics["scale"] == 2.0
        assert metrics["original_size"] == original_size
        assert metrics["new_size"] == expected_new_size
        assert metrics["resize_count"] == 1
        assert reward == 0.0  # default step_reward
        assert "Resize complete" in response.text
        assert response.image is not None
        assert len(response.image) == 1
        assert response.image[0].size == expected_new_size

    def test_execute_scale_down(self, tool, instance_with_image):
        """Test scaling down an image."""
        instance_id, _ = instance_with_image
        original_size = tool.get_instance_stats(instance_id)["current_size"]
        expected_new_size = (int(original_size[0] * 0.5), int(original_size[1] * 0.5))

        response, reward, metrics = asyncio.run(tool.execute(instance_id, {"scale": 0.5}))

        assert metrics["success"] is True
        assert metrics["scale"] == 0.5
        assert metrics["new_size"] == expected_new_size
        assert "Resize complete" in response.text

    def test_execute_scale_unchanged(self, tool, instance_with_image):
        """Test with scale factor 1.0 (no change)."""
        instance_id, _ = instance_with_image
        original_size = tool.get_instance_stats(instance_id)["current_size"]

        response, reward, metrics = asyncio.run(tool.execute(instance_id, {"scale": 1.0}))

        assert metrics["success"] is True
        assert metrics["scale"] == 1.0
        assert metrics["new_size"] == original_size
        assert "Resize complete" in response.text

    def test_execute_invalid_instance_id(self, tool):
        """Test executing with invalid instance_id."""
        response, reward, metrics = asyncio.run(tool.execute("nonexistent-id", {"scale": 2.0}))

        assert metrics["success"] is False
        assert metrics["error"] == "invalid_instance"
        assert reward < 0  # penalty

    def test_execute_missing_scale_parameter(self, tool, instance_with_image):
        """Test executing without scale parameter."""
        instance_id, _ = instance_with_image

        response, reward, metrics = asyncio.run(tool.execute(instance_id, {}))

        assert metrics["success"] is False
        assert metrics["error"] == "missing_scale"
        assert reward < 0

    def test_execute_invalid_scale_type(self, tool, instance_with_image):
        """Test executing with invalid scale type."""
        instance_id, _ = instance_with_image

        response, reward, metrics = asyncio.run(tool.execute(instance_id, {"scale": "large"}))

        assert metrics["success"] is False
        assert metrics["error"] == "invalid_scale_type"

    def test_execute_scale_not_allowed(self, tool, instance_with_image):
        """Test executing with scale not in allowed_scales."""
        instance_id, _ = instance_with_image

        response, reward, metrics = asyncio.run(tool.execute(instance_id, {"scale": 3.0}))

        assert metrics["success"] is False
        assert metrics["error"] == "scale_not_allowed"
        assert "3.0 not supported" in response.text

    def test_execute_max_calls_exceeded(self, tool, instance_with_image):
        """Test that max_resize_calls limit is enforced."""
        instance_id, _ = instance_with_image

        # Execute max_resize_calls times
        for i in range(tool.max_resize_calls):
            response, reward, metrics = asyncio.run(tool.execute(instance_id, {"scale": 1.0}))
            assert metrics["success"] is True

        # Next call should fail
        response, reward, metrics = asyncio.run(tool.execute(instance_id, {"scale": 1.0}))
        assert metrics["success"] is False
        assert metrics["error"] == "max_calls_exceeded"

    def test_execute_multiple_resizes(self, tool, instance_with_image):
        """Test multiple resize operations."""
        instance_id, _ = instance_with_image

        first_original = tool.get_instance_stats(instance_id)["current_size"]
        first_expected = (int(first_original[0] * 2.0), int(first_original[1] * 2.0))
        asyncio.run(tool.execute(instance_id, {"scale": 2.0}))

        second_expected = (int(first_expected[0] * 0.5), int(first_expected[1] * 0.5))
        response, reward, metrics = asyncio.run(tool.execute(instance_id, {"scale": 0.5}))

        assert metrics["success"] is True
        assert metrics["original_size"] == first_expected
        assert metrics["new_size"] == second_expected
        assert metrics["resize_count"] == 2

        stats = tool.get_instance_stats(instance_id)
        assert len(stats["resize_history"]) == 2


class TestImageResizeToolCalcReward:
    """Test ImageResizeTool.calc_reward() method."""

    @pytest.fixture
    def tool(self):
        """Create a tool instance for testing."""
        return ImageResizeTool(get_default_config(), get_default_tool_schema())

    def test_calc_reward_no_penalty(self, tool):
        """Test calc_reward with default (no penalty)."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = create_test_image(100, 100)
            img.save(f, format="PNG")
            temp_path = f.name

        try:
            instance_id, _ = asyncio.run(tool.create(image=temp_path))
            asyncio.run(tool.execute(instance_id, {"scale": 2.0}))

            reward = asyncio.run(tool.calc_reward(instance_id))
            assert reward == 0.0
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_calc_reward_with_penalty(self, tool):
        """Test calc_reward with resize penalty."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = create_test_image(100, 100)
            img.save(f, format="PNG")
            temp_path = f.name

        try:
            instance_id, _ = asyncio.run(tool.create(image=temp_path))
            asyncio.run(tool.execute(instance_id, {"scale": 2.0}))
            asyncio.run(tool.execute(instance_id, {"scale": 0.5}))

            reward = asyncio.run(tool.calc_reward(instance_id, resize_penalty=0.01))
            assert reward == pytest.approx(-0.02)  # 2 calls * 0.01 penalty
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_calc_reward_invalid_instance(self, tool):
        """Test calc_reward with invalid instance_id."""
        reward = asyncio.run(tool.calc_reward("nonexistent-id"))
        assert reward == 0.0


class TestImageResizeToolRelease:
    """Test ImageResizeTool.release() method."""

    @pytest.fixture
    def tool(self):
        """Create a tool instance for testing."""
        return ImageResizeTool(get_default_config(), get_default_tool_schema())

    def test_release_instance(self, tool):
        """Test releasing an instance."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = create_test_image(100, 100)
            img.save(f, format="PNG")
            temp_path = f.name

        try:
            instance_id, _ = asyncio.run(tool.create(image=temp_path))
            assert tool.get_instance_stats(instance_id) is not None

            asyncio.run(tool.release(instance_id))
            assert tool.get_instance_stats(instance_id) is None
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_release_nonexistent_instance(self, tool):
        """Test releasing a nonexistent instance (should not raise)."""
        asyncio.run(tool.release("nonexistent-id"))  # Should not raise


class TestImageResizeToolGetInstanceStats:
    """Test ImageResizeTool.get_instance_stats() method."""

    @pytest.fixture
    def tool(self):
        """Create a tool instance for testing."""
        return ImageResizeTool(get_default_config(), get_default_tool_schema())

    def test_get_instance_stats(self, tool):
        """Test getting instance stats."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = create_test_image(100, 80)
            img.save(f, format="PNG")
            temp_path = f.name

        try:
            instance_id, _ = asyncio.run(tool.create(image=temp_path))
            original_size = tool.get_instance_stats(instance_id)["current_size"]
            expected_new_size = (int(original_size[0] * 2.0), int(original_size[1] * 2.0))
            asyncio.run(tool.execute(instance_id, {"scale": 2.0}))

            stats = tool.get_instance_stats(instance_id)
            assert stats["original_size"] == original_size
            assert stats["current_size"] == expected_new_size
            assert stats["resize_count"] == 1
            assert len(stats["resize_history"]) == 1
            assert stats["resize_history"][0]["scale"] == 2.0
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_get_instance_stats_nonexistent(self, tool):
        """Test getting stats for nonexistent instance."""
        stats = tool.get_instance_stats("nonexistent-id")
        assert stats is None


class TestImageResizeToolStepReward:
    """Test step reward configuration."""

    def test_positive_step_reward(self):
        """Test positive step reward for successful operations."""
        config = get_default_config()
        config["step_reward"] = 0.05
        tool = ImageResizeTool(config, get_default_tool_schema())

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img = create_test_image(100, 100)
            img.save(f, format="PNG")
            temp_path = f.name

        try:
            instance_id, _ = asyncio.run(tool.create(image=temp_path))
            response, reward, metrics = asyncio.run(tool.execute(instance_id, {"scale": 2.0}))

            assert metrics["success"] is True
            assert reward == 0.05
        finally:
            Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
