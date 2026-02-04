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
Image Resize Tool for OCR Agent.

This tool provides image scaling functionality to optimize OCR recognition.
The agent can choose to scale up (for small/blurry details) or scale down
(for unnecessarily large images) based on the image content.
"""

import logging
import os
from typing import Any, Optional
from uuid import uuid4

from PIL import Image
from qwen_vl_utils import fetch_image

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ImageResizeTool(BaseTool):
    """Image resize tool for OCR optimization.

    This tool allows an agent to resize images by a scale factor to improve
    OCR accuracy. The agent can:
    - Scale UP (e.g., 2.0, 4.0) when details are small or blurry
    - Scale DOWN (e.g., 0.5, 0.25) when the image is unnecessarily large
    - Keep original size (1.0) when the image is already optimal

    Attributes:
        allowed_scales: List of allowed scale factors
        interpolation: PIL resampling method name
        step_reward: Reward for each successful tool call
        invalid_scale_penalty: Penalty for invalid scale values
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize the ImageResizeTool.

        Args:
            config: Tool configuration dictionary containing:
                - allowed_scales: List of allowed scale factors (default: [0.25, 0.5, 1.0, 2.0, 4.0])
                - interpolation: PIL resampling method (default: "LANCZOS")
                - step_reward: Reward for successful resize (default: 0.0)
                - invalid_scale_penalty: Penalty for invalid scale (default: -0.05)
                - max_resize_calls: Maximum resize calls per instance (default: 5)
            tool_schema: OpenAI function tool schema
        """
        super().__init__(config, tool_schema)
        self._instances: dict[str, dict[str, Any]] = {}

        # Configuration with defaults matching eval_ocr implementation
        self.allowed_scales = config.get("allowed_scales", [0.25, 0.5, 1.0, 2.0, 4.0])
        self.interpolation = config.get("interpolation", "LANCZOS")
        self.step_reward = config.get("step_reward", 0.0)
        self.invalid_scale_penalty = config.get("invalid_scale_penalty", -0.05)
        self.max_resize_calls = config.get("max_resize_calls", 5)

        # Validate interpolation method
        self._resample_method = getattr(Image.Resampling, self.interpolation, Image.Resampling.LANCZOS)

        logger.info(
            f"Initialized ImageResizeTool with config: "
            f"allowed_scales={self.allowed_scales}, "
            f"interpolation={self.interpolation}, "
            f"step_reward={self.step_reward}, "
            f"invalid_scale_penalty={self.invalid_scale_penalty}"
        )

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a tool instance and cache the original image.

        Args:
            instance_id: Optional unique identifier for this instance
            **kwargs: Should contain 'image' key with image data, or 'create_kwargs'
                containing {'image': image_data}. Image can be:
                - A PIL.Image.Image object
                - A string containing an HTTP/HTTPS URL
                - A string containing a local file path
                - A string containing a file URI
                - A string containing a base64-encoded image

        Returns:
            Tuple of (instance_id, ToolResponse)

        Raises:
            ValueError: If 'image' parameter is missing
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Handle create_kwargs parameter if passed
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        # Get image from kwargs
        image = kwargs.get("image")
        if image is None:
            raise ValueError("Missing required 'image' parameter in kwargs")

        # Fetch and cache the image
        img = fetch_image({"image": image})
        self._instances[instance_id] = {
            "original_image": img,
            "current_image": img,
            "resize_count": 0,
            "resize_history": [],
        }

        logger.info(f"Created ImageResizeTool instance {instance_id} with image size {img.size}")
        return instance_id, ToolResponse()

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> tuple[ToolResponse, float, dict]:
        """Execute image resize operation.

        Args:
            instance_id: The instance identifier
            parameters: Dictionary containing:
                - scale: Scale factor (must be in allowed_scales)

        Returns:
            Tuple of (ToolResponse, reward, metrics):
                - ToolResponse: Contains resized image and feedback text
                - reward: Step reward (0.0 for success, negative for errors)
                - metrics: Dictionary with success status and details
        """
        # Validate instance
        instance_data = self._instances.get(instance_id)
        if not instance_data:
            return (
                ToolResponse(text="Error: Invalid instance_id. Tool instance not found."),
                self.invalid_scale_penalty,
                {"success": False, "error": "invalid_instance"},
            )

        # Check resize count limit
        if instance_data["resize_count"] >= self.max_resize_calls:
            return (
                ToolResponse(
                    text=f"Error: Maximum resize calls ({self.max_resize_calls}) exceeded. "
                    "Please provide your OCR result directly."
                ),
                self.invalid_scale_penalty,
                {"success": False, "error": "max_calls_exceeded"},
            )

        # Validate scale parameter
        scale = parameters.get("scale")
        if scale is None:
            return (
                ToolResponse(text="Error: 'scale' parameter is required."),
                self.invalid_scale_penalty,
                {"success": False, "error": "missing_scale"},
            )

        try:
            scale = float(scale)
        except (ValueError, TypeError):
            return (
                ToolResponse(text=f"Error: 'scale' must be a number, got {type(scale).__name__}."),
                self.invalid_scale_penalty,
                {"success": False, "error": "invalid_scale_type"},
            )

        if scale not in self.allowed_scales:
            return (
                ToolResponse(
                    text=f"Error: scale {scale} not supported. Allowed values: {self.allowed_scales}"
                ),
                self.invalid_scale_penalty,
                {"success": False, "error": "scale_not_allowed", "scale": scale},
            )

        # Execute resize operation
        current_image = instance_data["current_image"]
        original_size = current_image.size
        new_size = (int(current_image.width * scale), int(current_image.height * scale))

        # Validate new size
        if new_size[0] < 1 or new_size[1] < 1:
            return (
                ToolResponse(
                    text=f"Error: Resulting image size {new_size} is too small. "
                    f"Current size: {original_size}, scale: {scale}"
                ),
                self.invalid_scale_penalty,
                {"success": False, "error": "size_too_small"},
            )

        try:
            resized_image = current_image.resize(new_size, resample=self._resample_method)
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return (
                ToolResponse(text=f"Error resizing image: {str(e)}"),
                self.invalid_scale_penalty,
                {"success": False, "error": "resize_failed"},
            )

        # Update instance state
        instance_data["current_image"] = resized_image
        instance_data["resize_count"] += 1
        instance_data["resize_history"].append({"scale": scale, "from_size": original_size, "to_size": new_size})

        # Generate feedback text (consistent with SFT training data format)
        # Format: "Resize complete. New scale: {scale}x\nResult:"
        feedback = f"Resize complete. New scale: {scale}x\nResult:"

        logger.info(f"Instance {instance_id}: resized from {original_size} to {new_size} (scale={scale})")

        return (
            ToolResponse(image=[resized_image], text=feedback),
            self.step_reward,
            {
                "success": True,
                "scale": scale,
                "original_size": original_size,
                "new_size": new_size,
                "resize_count": instance_data["resize_count"],
            },
        )

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate the reward based on tool usage.

        The reward can be customized based on:
        - Number of resize calls (efficiency)
        - Final image quality
        - OCR accuracy (if ground truth available)

        Args:
            instance_id: The instance identifier

        Returns:
            Reward value (default: 0.0)
        """
        instance_data = self._instances.get(instance_id)
        if not instance_data:
            return 0.0

        # Base reward is 0, can be extended for custom reward logic
        # e.g., penalize excessive resize calls
        resize_count = instance_data["resize_count"]
        resize_penalty = kwargs.get("resize_penalty", 0.0)

        return -resize_count * resize_penalty

    async def release(self, instance_id: str, **kwargs) -> None:
        """Release the tool instance and free resources.

        Args:
            instance_id: The instance identifier to release
        """
        if instance_id in self._instances:
            del self._instances[instance_id]
            logger.info(f"Released ImageResizeTool instance {instance_id}")

    def get_instance_stats(self, instance_id: str) -> Optional[dict]:
        """Get statistics for an instance (for debugging/monitoring).

        Args:
            instance_id: The instance identifier

        Returns:
            Dictionary with instance statistics or None if not found
        """
        instance_data = self._instances.get(instance_id)
        if not instance_data:
            return None

        return {
            "original_size": instance_data["original_image"].size,
            "current_size": instance_data["current_image"].size,
            "resize_count": instance_data["resize_count"],
            "resize_history": instance_data["resize_history"],
        }
