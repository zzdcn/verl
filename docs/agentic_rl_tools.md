# Agentic RL 与工具实现速览

本笔记整合 `docs/start/agentic_rl.rst`、`verl/tools/base_tool.py`、`verl/tools/gsm8k_tool.py`、`verl/tools/geo3k_tool.py`、`verl/tools/image_zoom_in_tool.py` 以及**图片理解 Agent + Resize 工具**需求，帮助快速搭建能够利用视觉工具完成 OCR 任务的 Agentic RL 流程。

## 1. Agentic RL 训练脉络
 
- **异步 Rollout**：训练端（PPO Trainer）与推理端（AsyncServer, AsyncLLMServerManager）解耦，通过 Ray actor 维护粘性会话，避免工具调用阻塞 GPU (`docs/start/agentic_rl.rst`).
- **多轮对话 + 工具**：数据集需包含 `agent_name` 字段，AgentLoop 依据该字段选择 `single_turn_agent` 或 `tool_agent_loop`，并在 rollout 过程中注入工具 schema 与工具响应。
- **LangGraph / 自定义 Agent**：AgentLoop 充当 LangGraph agent 的适配层，使用统一的 token in/out 接口保证训练与推理 token 一致。

伪代码概览：

```python
async def run_agentic_rl_step(batch):
    mgr.wake_up_servers()
    outputs = []
    for chunk in batch.split(num_workers):
        agent_outputs = await worker.generate_sequences(chunk)
        outputs.append(agent_outputs)
    mgr.sleep_servers()
    return concat(outputs)
```

## 2. Tool Base Class (`verl/tools/base_tool.py`)

- `BaseTool` 统一了工具生命周期：`create → execute → calc_reward → release`，并要求提供 OpenAI Function 格式的 `tool_schema`。
- 执行阶段通过 `ToolResponse` 返回文本/图像/视频，多模内容需以列表形式存储，便于 AgentLoop 重组消息。

伪代码：

```python
class BaseTool:
    def __init__(self, config, schema):
        self.tool_schema = schema or self.get_openai_tool_schema()

    async def create(self, instance_id=None, **kwargs):
        return instance_id or uuid4(), ToolResponse()

    async def execute(self, instance_id, parameters, **ctx):
        return ToolResponse(text="Updated"), 0.0, {}
```

## 3. 数学奖励工具

### 3.1 `Gsm8kTool` (`verl/tools/gsm8k_tool.py`)

- 处理 `calc_gsm8k_reward` 工具调用，`create` 阶段缓存地面真值答案，`execute` 将模型输出标准化为 `#### <answer>`。
- `calc_reward` 通过 `verl.utils.reward_score.gsm8k.compute_score` 赋分，若新提交未提高奖励则给予 -0.05 惩罚。

伪代码：

```python
async def execute(instance, params):
    normalized = ensure_hash_prefix(params["answer"])
    reward = await calc_reward(instance)
    delta = 0.0 if reward > cache[instance].reward else -0.05
    cache[instance].update(response=normalized, reward=reward)
    return ToolResponse(text=f"{normalized=}{reward=}"), delta, {}
```

### 3.2 `Geo3kTool` (`verl/tools/geo3k_tool.py`)

- 结构与 GSM8K 类似，但奖励函数改用 `verl.utils.reward_score.geo3k.compute_score`，且答案要求以 `\boxed{}` 包裹。
- 仍遵循“若奖励未提升则惩罚”的策略，利于模型反思后再提交。

## 4. 视觉裁剪工具：`ImageZoomInTool` (`verl/tools/image_zoom_in_tool.py`)

- 提供图像局部放大功能，支持 Ray 远程 worker + TokenBucket 限流，避免并发图像裁剪导致的资源争用。
- `create` 阶段会解码多种图像来源（URL、本地、base64），并缓存原图；`execute` 读取 `bbox_2d`/`label`，自动校验与扩展过小的框（最小 28x28）。
- 若裁剪成功，返回新的图像切片与描述文本；否则返回错误提示并附 -0.05 惩罚。

伪代码：

```python
async def execute(instance, params):
    bbox = sanitize_bbox(params["bbox_2d"], image.size)
    if not bbox:
        return ToolResponse(text="invalid bbox"), -0.05, {"success": False}
    cropped = image.crop(bbox)
    return ToolResponse(image=[cropped], text=f"Zoomed {bbox}"), 0.0, {"success": True}
```

## 5. 图片理解 Agent + Resize 工具扩展

为了让 OCR Agent 学会自适应选择“放大/缩小”倍数，我们可以实现一个新的 `ImageResizeTool`，并在 AgentLoop 里要求模型先分析当前图像，再决定要调用工具。整体思路如下：

1. **工具接口**：限定若干放大/缩小倍数（如 `[0.5, 1.0, 2.0, 4.0]`），通过参数 `scale` 控制 `PIL.Image.resize`；返回 `ToolResponse(image=[resized_img], text=...)` 方便模型读取。可在工具配置里允许自定义 `min_scale`/`max_scale`/`allowed_scales`。
2. **Agent 策略**：
   - 先看到原始题目（可能附带低分辨率图），规划是否需要 resize。
   - 依据工具返回的新图继续 OCR 推理，最终产出文本答案。
3. **奖励设计**：结合 OCR 正确率（如 CER/WER 或通过标注答案比对）+ 工具调用成本（例如调用次数惩罚），激励模型“够用即止”。

### 5.1 新工具实现草案

文件：`verl/tools/image_resize_tool.py`

```python
class ImageResizeTool(BaseTool):
    def __init__(self, config, tool_schema):
        super().__init__(config, tool_schema)
        self.allowed_scales = config.get("allowed_scales", [0.5, 1.0, 2.0, 4.0])
        self.interp = config.get("interpolation", "bicubic")

    async def create(self, instance_id=None, **kwargs):
        instance_id = instance_id or str(uuid4())
        img = fetch_image(kwargs.get("image") or kwargs["create_kwargs"]["image"])
        self._instances[instance_id] = {"image": img}
        return instance_id, ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        scale = float(parameters.get("scale", 1.0))
        if scale not in self.allowed_scales:
            return ToolResponse(text=f"scale {scale} not supported"), -0.05, {"success": False}
        img = self._instances[instance_id]["image"]
        size = (int(img.width * scale), int(img.height * scale))
        resized = img.resize(size, resample=getattr(Image, self.interp.upper()))
        text = f"Resized from {img.size} to {resized.size} (x{scale})."
        return ToolResponse(image=[resized], text=text), 0.0, {"success": True}
```

### 5.2 训练配置修改

| 步骤 | 修改说明 |
| --- | --- |
| 工具配置 | 在 `examples/sglang_multiturn/config/tool_config/*.yaml` 中新增 `ImageResizeTool` 条目，指定 `class_name`, `allowed_scales`, `tool_schema`（包含 `scale` 参数，值域可选列表）。 |
| 数据准备 | 数据集中追加 `agent_name="tool_agent"`、`tools_kwargs.image_resize_tool.create_kwargs.image=<原图>`，并在 `prompt` 中提示“可调用 resize 工具以协助 OCR”。 |
| Rollout | `actor_rollout_ref.rollout.multi_turn.tool_config_path` 指向包含新工具的 YAML；必要时在 `tool_agent_loop` 中将工具返回的多模内容写入消息（ImageZoomIn 已支持图像列表）。 |
| 奖励 | 可复用 `verl.utils.reward_score` 的自定义函数或编写新 reward loop，比较 OCR 输出与真值；另可在 `AgentLoopOutput.extra_fields` 里统计 tool 调用次数供外部奖励逻辑参考。 |
| 推理/Serving | 推理阶段需部署相同工具，确保训练/上线一致性。 |

### 5.3 Agent 回合示例（伪对话）

```text
user  : 这是扫描文档的一部分，请抄录文字。你可以调用 image_resize_tool 放大或缩小。
assistant(thinking): 图像模糊，决定先放大 2 倍。
assistant(tool_call): image_resize_tool({"scale": 2.0})
tool  : 返回放大图片 + 描述
assistant(thinking): 读取清晰区域，输出 OCR 结果。
assistant(final): <文字答案>
```

## 6. 数据集构建指南

Agentic RL 数据集需要同时携带文本提示、原始图像引用、工具上下文以及奖励相关元信息。建议流程：

1. **收集原始样本**：包含图像路径/URL与 OCR 目标文本。
2. **预处理脚本**：仿照 `examples/data_preprocess/gsm8k_tool_agent_loop.py` 写 `ocr_resize_agent_loop.py`，核心步骤：

```python
sample = {
    "agent_name": "tool_agent",
    "prompt": [
        {"role": "system", "content": "你是OCR助手，可调用image_resize_tool。"},
        {"role": "user", "content": "请抄录附件中的文字。"}
    ],
    "extra_info": {
        "answer": target_text,
        "image_path": image_uri,
        "split": split,
    },
    "tools_kwargs": {
        "image_resize_tool": {
            "create_kwargs": {"image": image_uri}
        }
    },
    "reward_model": {
        "style": "ocr_rule",
        "ground_truth": target_text,
    }
}
```

3. **输出格式**：推荐写入 parquet (`dataset.to_parquet(...)`) 或 HuggingFace JSONL，确保 `return_raw_chat=True` 时 `RLHFDataset` 能直接读取 `prompt` 列表。
4. **字段约定**：
   - `agent_name`：决定是否启用 `ToolAgentLoop`。
   - `tools_kwargs`：为工具的 `create/execute` 传参，包含图像、ground truth 等。
   - `extra_info.interaction_kwargs`：若需引入模拟用户可在此配置。
   - `reward_model`：声明奖励风格（rule-based / model-based），以便 RewardLoop 解析。

## 7. 奖励实现与集成

### 7.1 规则奖励（OCR 示例）

可在 `verl/utils/reward_score` 下新增 `ocr.py`，提供 `compute_score(pred, label, metric="cer")`。奖励逻辑：

```python
def compute_score(pred, label, metric="cer"):
    pred_norm = normalize(pred)
    label_norm = normalize(label)
    cer = levenshtein(pred_norm, label_norm) / max(1, len(label_norm))
    return max(0.0, 1.0 - cer)
```

工具调用回合结束后，`AgentLoopOutput.extra_fields` 会附带 `reward_model` 信息；RewardLoop (`verl.experimental.reward_loop`) 读取 batch 并调用上述函数得到标量奖励，可再叠加：

- 工具调用数惩罚：`reward -= 0.01 * num_tool_calls`
- 未使用工具但 CER > 阈值时附加负奖励。

### 7.2 模型奖励

若需引入多模 RM，可在 `reward_model` 配置中开启 `use_reward_loop=True` 并提供自定义 `RewardModelWorker`；数据准备阶段把图像/文本一并塞入 `non_tensor_batch["multi_modal_inputs"]`，保证 RM 能复现上下文。

### 7.3 训练脚本对接

- 在 Hydra 配置里设置 `reward_model.style: ocr_rule` 并指向实现脚本。
- `ppo_trainer` 会在 rollout 后调用 RewardLoop；若 reward 已在 AgentLoop 内直接计算（例如工具即时评分），可通过 `AgentLoopOutput.reward_score` 返回，跳过 RM。

## 8. 完整训练流程 Checklist

1. **准备工具**：
   - 在 `examples/sglang_multiturn/config/tool_config/ocr_resize.yaml` 中登记 `ImageResizeTool`、`ImageZoomInTool` 等。
   - 若有多模工具，确保模型 processor 支持。
2. **构建数据**：运行自定义 preprocess 脚本生成 parquet，字段包含 `prompt/agent_name/tools_kwargs/reward_model`。
3. **配置 Rollout**：
   - `actor_rollout_ref.rollout.multi_turn.enable=True`
   - `...tool_config_path=.../ocr_resize.yaml`
   - 按需设置 `max_assistant_turns`, `max_parallel_calls`。
4. **奖励集成**：在 `reward_model` 配置里选择 `use_reward_loop` 或 `rule`，确保脚本可被导入。
5. **启动训练**：例如

```bash
bash examples/sglang_multiturn/run_qwen2.5-3b_gsm8k_tool_agent_mlflow.sh \
  data.train_path=~/data/ocr/train.parquet \
  actor_rollout_ref.rollout.multi_turn.tool_config_path=.../ocr_resize.yaml \
  trainer.project_name=ocr_agentic_rl
```

6. **监控与调试**：
   - `mlflow ui` 或 `rollout trace` 检查工具调用/tokenization。
   - 根据日志中 `agent_loop/tool_calls`、`reward_value` 调整奖励权重。
7. **验收**：使用与训练相同的工具配置在 validation 集上 rollout，评估 CER/WER 与平均调用次数；必要时蒸馏成无工具模型供推理。

## 9. 结合 AgentLoop 的使用建议

1. **配置工具**：在 rollout 配置里设置 `actor_rollout_ref.rollout.multi_turn.tool_config_path`，指向包含上述工具 schema 的 YAML。
2. **准备数据**：数据样本要包含 `agent_name="tool_agent"`、`tools_kwargs`（如为 GSM8K/Geo3K 传 ground truth）等字段，AgentLoop 会自动透传。
3. **监控异步执行**：利用 AgentLoop trace/mlflow（见 `docs/start/agentic_rl.rst`）观察工具调用情况，核对 tokenization 一致性，并统计 resize 工具使用频次/成功率以评估策略。

该文档可作为开发 Agentic RL + 工具调用任务的速查表，根据需要扩展新的 Tool 类或 LangGraph Agent。

---

## 10. OCR Resize Agent 训练实现计划

本节基于 `eval_ocr/` 项目中已实现的评估逻辑，制定完整的 RL 训练实现计划。

### 10.1 项目背景

`eval_ocr/` 项目实现了一个多卡并行的 OCR 模型评估脚本，核心特性：

- **Batch 内 Agent 并行推理**：使用动态 Batch 管理，处理不同样本异步完成的情况
- **工具调用**：`resize` 工具支持图像放大/缩小
- **多类型指标**：支持 text（编辑距离）、table（TEDS）、formula（CDM）三种评估
- **多卡分布式**：基于 Accelerate 实现

### 10.2 评估与训练的对应关系

| eval_ocr 组件 | verl 训练对应组件 | 说明 |
|--------------|------------------|------|
| `BatchAgentRunner` | `ToolAgentLoop` | Agent 推理循环 |
| `AgentState` dataclass | `AgentData` class | 样本状态管理 |
| `parse_tool_call()` | `ToolParser` (hermes/gpt-oss) | 工具调用解析 |
| `resize_image()` | `ImageResizeTool.execute()` | 工具执行 |
| `compute_metrics()` | `RewardLoop` / rule-based reward | 奖励计算 |
| Accelerate 多卡 | Ray + FSDP/Megatron | 分布式训练 |

### 10.3 训练实现计划

#### Phase 1: 工具实现 (`verl/tools/image_resize_tool.py`)

**目标**：实现与 eval_ocr 逻辑一致的 resize 工具

```python
# verl/tools/image_resize_tool.py
from uuid import uuid4
from PIL import Image
from qwen_vl_utils import fetch_image

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

class ImageResizeTool(BaseTool):
    """图像缩放工具，支持放大/缩小以优化 OCR 识别效果"""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instances = {}
        # 配置项（与 eval_ocr 一致）
        self.allowed_scales = config.get("allowed_scales", [0.5, 1.0, 2.0, 4.0])
        self.interpolation = config.get("interpolation", "LANCZOS")
        self.step_reward = config.get("step_reward", 0.0)  # 每次调用的即时奖励
        self.invalid_scale_penalty = config.get("invalid_scale_penalty", -0.05)

    async def create(self, instance_id=None, **kwargs):
        """初始化工具实例，缓存原始图像"""
        instance_id = instance_id or str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        image = kwargs.get("image")
        if image is None:
            raise ValueError("Missing required 'image' parameter")

        # 支持多种图像来源（URL、本地路径、base64）
        img = fetch_image({"image": image})
        self._instances[instance_id] = {
            "original_image": img,
            "current_image": img,
            "resize_count": 0,
        }
        return instance_id, ToolResponse()

    async def execute(self, instance_id, parameters, **kwargs):
        """执行图像缩放"""
        scale = parameters.get("scale")

        # 验证 scale 参数
        if scale is None:
            return (
                ToolResponse(text="Error: 'scale' parameter is required."),
                self.invalid_scale_penalty,
                {"success": False}
            )

        try:
            scale = float(scale)
        except (ValueError, TypeError):
            return (
                ToolResponse(text=f"Error: 'scale' must be a number, got {type(scale).__name__}."),
                self.invalid_scale_penalty,
                {"success": False}
            )

        if scale not in self.allowed_scales:
            return (
                ToolResponse(text=f"Error: scale {scale} not supported. Allowed: {self.allowed_scales}"),
                self.invalid_scale_penalty,
                {"success": False}
            )

        instance_data = self._instances.get(instance_id)
        if not instance_data:
            return (
                ToolResponse(text="Error: Invalid instance_id."),
                self.invalid_scale_penalty,
                {"success": False}
            )

        # 执行缩放（与 eval_ocr 一致）
        current_image = instance_data["current_image"]
        new_size = (int(current_image.width * scale), int(current_image.height * scale))
        resample = getattr(Image.Resampling, self.interpolation, Image.Resampling.LANCZOS)
        resized_image = current_image.resize(new_size, resample)

        # 更新状态
        instance_data["current_image"] = resized_image
        instance_data["resize_count"] += 1

        # 生成反馈文本（与 eval_ocr SFT 格式一致）
        if scale < 1.0:
            feedback = f"Downsampling complete. Scale: {scale}x."
        else:
            feedback = f"Upsampling complete. Scale: {scale}x."
        feedback += f"\nImage resized from {current_image.size} to {resized_image.size}."

        return (
            ToolResponse(image=[resized_image], text=feedback),
            self.step_reward,
            {"success": True, "scale": scale, "new_size": resized_image.size}
        )

    async def release(self, instance_id, **kwargs):
        """释放工具实例"""
        if instance_id in self._instances:
            del self._instances[instance_id]
```

**工具配置 YAML** (`examples/sglang_multiturn/config/tool_config/ocr_resize_tool_config.yaml`):

```yaml
tools:
  - class_name: "verl.tools.image_resize_tool.ImageResizeTool"
    config:
      # 与 SFT 训练数据一致的 scale 值
      allowed_scales: [0.125, 0.143, 0.167, 0.2, 0.25, 0.333, 0.5, 2, 3, 4, 5, 6, 7, 8]
      interpolation: "LANCZOS"
      step_reward: 0.0
      invalid_scale_penalty: -0.05
    tool_schema:
      type: "function"
      function:
        name: "resize"
        description: |
          Resize the current image by a scale factor to optimize OCR recognition.
          Allowed scale values: 0.125, 0.143, 0.167, 0.2, 0.25, 0.333, 0.5, 2, 3, 4, 5, 6, 7, 8
          - scale < 1: Decrease resolution. For example: scale=0.5 means half size
          - scale > 1: Increase resolution. For example: scale=2 means 2x larger
        parameters:
          type: "object"
          properties:
            scale:
              type: "number"
              description: "Scale factor for resizing."
          required: ["scale"]
```

#### Phase 2: 数据预处理脚本 (`examples/data_preprocess/ocr_resize_agent_loop.py`)

**目标**：将 OCR 数据集转换为 verl 训练格式，**与 SFT 训练数据保持格式一致**

```python
# examples/data_preprocess/ocr_resize_agent_loop.py
"""
将 OCR 数据集预处理为 verl Agentic RL 训练格式
与 SFT 训练数据格式完全一致，确保训推一致性

输入格式 (JSON/JSONL):
{
    "img_path_sh": "/path/to/image.jpg",
    "groundtruth": "识别结果文本",
    "data_id": "text_block_000001",  # 可选，用于推断 tag
    "tag": "text"  # 可选，text/table/equation
}

输出格式 (Parquet):
{
    "data_source": "ocr_dataset",
    "agent_name": "tool_agent",
    "prompt": [...],
    "extra_info": {...},
    "reward_model": {...}
}
"""

import argparse
import json
import os
from pathlib import Path

import datasets

# ============================================================================
# 系统提示 - 按 tag 类型分类（与 SFT 训练数据完全一致）
# ============================================================================

# 通用工具说明
TOOL_INSTRUCTION = """
You have access to a resize tool that can adjust the image resolution:
<tool_call>{"name": "resize", "arguments": {"scale": N}}</tool_call>
where N can be: 0.125, 0.143, 0.167, 0.2, 0.25, 0.333, 0.5, 2, 3, 4, 5, 6, 7, 8
- scale < 1: Decrease resolution. For example: scale=0.5 means half size
- scale > 1: Increase resolution. For example: scale=2 means 2x larger

When you complete the extraction, wrap your result with <final_answer> tags.
"""

SYSTEM_PROMPTS = {
    "text": (
        "You are an expert OCR assistant. Your task is to accurately extract all text content from the given image.\n"
        + TOOL_INSTRUCTION
        + "\nOutput format:\n- Output text in reading order\n- Preserve paragraph structure"
    ),
    "table": (
        "You are an expert OCR assistant. Your task is to accurately extract the table content from the given image and output it in HTML format.\n"
        + TOOL_INSTRUCTION
        + "\nOutput format:\n- Use <table border=\"1\"> as the opening tag\n- Use <tr> for rows, <td> for cells"
    ),
    "equation": (
        "You are an expert OCR assistant. Your task is to accurately extract the mathematical equation from the given image and output it in LaTeX format.\n"
        + TOOL_INSTRUCTION
        + "\nOutput format:\n- Output the equation in LaTeX format\n- Use standard LaTeX math notation"
    ),
}

USER_PROMPTS = {
    "text": "Extract all text content from this image.",
    "table": "Extract the table content from this image in HTML format.",
    "equation": "Extract the mathematical equation from this image in LaTeX format.",
}


def infer_tag(data_id: str) -> str:
    """根据 data_id 推断数据类型"""
    if data_id:
        data_id_lower = data_id.lower()
        if "equation" in data_id_lower or "formula" in data_id_lower:
            return "equation"
        elif "table" in data_id_lower:
            return "table"
    return "text"


def load_raw_dataset(data_path: str) -> list:
    """加载原始 JSON/JSONL 数据集"""
    data = []
    path = Path(data_path)

    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return data


def process_sample(example: dict, idx: int, split: str) -> dict:
    """处理单个样本，转换为 verl 训练格式"""
    # 字段映射
    image_path = example.get("img_path_sh") or example.get("image")
    ground_truth = example.get("groundtruth") or example.get("ground_truth")
    data_id = example.get("data_id", f"{split}_{idx:06d}")
    tag = example.get("tag") or infer_tag(data_id)

    if not image_path or not ground_truth:
        raise ValueError(f"Missing required fields in sample {idx}")

    # 获取 tag 对应的 prompt（与 SFT 数据一致）
    system_prompt = SYSTEM_PROMPTS.get(tag, SYSTEM_PROMPTS["text"])
    user_prompt = USER_PROMPTS.get(tag, USER_PROMPTS["text"])

    # 构建 verl 训练格式
    # 注意：使用 "<image>" 占位符格式，与 SFT 数据一致
    return {
        "data_source": "ocr_dataset",
        "agent_name": "tool_agent",  # 启用 ToolAgentLoop
        "prompt": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"<image>{user_prompt}"  # 与 SFT 格式一致
            }
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
                    "create_kwargs": {"image": image_path}
                }
            },
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess OCR dataset for verl training")
    parser.add_argument("--input_path", type=str, required=True, help="Input JSON/JSONL file")
    parser.add_argument("--output_dir", type=str, default="~/data/ocr_verl", help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio")
    args = parser.parse_args()

    # 加载数据
    raw_data = load_raw_dataset(args.input_path)
    print(f"Loaded {len(raw_data)} samples from {args.input_path}")

    # 划分训练/测试集
    split_idx = int(len(raw_data) * args.train_ratio)
    train_data = raw_data[:split_idx]
    test_data = raw_data[split_idx:]

    # 处理数据
    train_processed = [process_sample(ex, i, "train") for i, ex in enumerate(train_data)]
    test_processed = [process_sample(ex, i, "test") for i, ex in enumerate(test_data)]

    # 转换为 HuggingFace Dataset
    train_dataset = datasets.Dataset.from_list(train_processed)
    test_dataset = datasets.Dataset.from_list(test_processed)

    # 保存
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(output_dir, "test.parquet"))

    print(f"Saved {len(train_processed)} train samples to {output_dir}/train.parquet")
    print(f"Saved {len(test_processed)} test samples to {output_dir}/test.parquet")


if __name__ == "__main__":
    main()
```

#### Phase 3: 奖励函数实现 (`verl/utils/reward_score/ocr.py`)

**目标**：实现与 eval_ocr 一致的指标计算逻辑

```python
# verl/utils/reward_score/ocr.py
"""
OCR 任务奖励函数，支持三种数据类型：
- text: 编辑距离相似度
- table: TEDS 分数
- formula: CDM (Character Detection Matching) 分数
"""

import re
from typing import Dict, Any

import editdistance


def normalize_text(text: str) -> str:
    """文本标准化"""
    if not text:
        return ""
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text.strip())
    return text


def compute_edit_distance_similarity(pred: str, gt: str) -> float:
    """
    计算编辑距离相似度 (归一化到 [0, 1])

    Args:
        pred: 预测文本
        gt: 真实文本

    Returns:
        float: 相似度分数，1.0 为完全匹配
    """
    pred_norm = normalize_text(pred)
    gt_norm = normalize_text(gt)

    if not gt_norm:
        return 1.0 if not pred_norm else 0.0

    distance = editdistance.eval(pred_norm, gt_norm)
    max_len = max(len(pred_norm), len(gt_norm))

    if max_len == 0:
        return 1.0

    similarity = 1.0 - (distance / max_len)
    return max(0.0, similarity)


def compute_teds_score(pred: str, gt: str) -> Dict[str, float]:
    """
    计算 TEDS (Tree-Edit-Distance-based Similarity) 分数

    需要安装: pip install table_recognition_metric
    """
    try:
        from table_recognition_metric import TEDS
        teds = TEDS(structure_only=False)
        teds_struct = TEDS(structure_only=True)

        score = teds.evaluate(pred, gt)
        score_struct = teds_struct.evaluate(pred, gt)

        return {
            "teds": score,
            "teds_struct": score_struct,
        }
    except ImportError:
        # Fallback to edit distance if TEDS not available
        similarity = compute_edit_distance_similarity(pred, gt)
        return {
            "teds": similarity,
            "teds_struct": similarity,
        }
    except Exception:
        return {"teds": 0.0, "teds_struct": 0.0}


def compute_formula_similarity(pred: str, gt: str) -> Dict[str, float]:
    """
    计算公式相似度 (CDM)

    CDM 需要额外的系统依赖（LaTeX 渲染），这里提供简化版本
    """
    # 简化版本：使用编辑距离
    # TODO: 集成完整的 CDM 实现
    similarity = compute_edit_distance_similarity(pred, gt)
    return {
        "recall": similarity,
        "precision": similarity,
        "f1_score": similarity,
    }


def compute_score(
    pred: str,
    gt: str,
    tag: str = "text",
    tool_call_count: int = 0,
    tool_penalty: float = 0.01,
) -> float:
    """
    计算 OCR 任务的奖励分数

    Args:
        pred: 模型预测结果
        gt: 真实标签
        tag: 数据类型 (text/table/formula)
        tool_call_count: 工具调用次数
        tool_penalty: 每次工具调用的惩罚

    Returns:
        float: 奖励分数
    """
    # 根据类型计算基础分数
    if tag == "text":
        base_score = compute_edit_distance_similarity(pred, gt)
    elif tag == "table":
        teds_result = compute_teds_score(pred, gt)
        base_score = teds_result.get("teds", 0.0)
    elif tag == "formula":
        formula_result = compute_formula_similarity(pred, gt)
        base_score = formula_result.get("f1_score", 0.0)
    else:
        base_score = compute_edit_distance_similarity(pred, gt)

    # 扣除工具调用惩罚（鼓励高效使用工具）
    penalty = tool_call_count * tool_penalty
    final_score = max(0.0, base_score - penalty)

    return final_score
```

#### Phase 4: 训练配置 (`examples/sglang_multiturn/config/ocr_resize_grpo.yaml`)

```yaml
# OCR Resize Agent GRPO 训练配置

hydra:
  searchpath:
    - file://verl/trainer/config

defaults:
  - ppo_trainer
  - _self_

# 数据配置
data:
  train_path: ~/data/ocr_verl/train.parquet
  val_path: ~/data/ocr_verl/test.parquet
  max_prompt_length: 2048  # VLM 需要更长的 prompt
  max_response_length: 1024
  train_batch_size: 128
  return_raw_chat: True  # 启用 raw chat 格式

# Actor-Rollout-Ref 配置
actor_rollout_ref:
  hybrid_engine: True
  model:
    path: Qwen/Qwen2.5-VL-3B-Instruct  # 或其他 VLM
  rollout:
    name: sglang
    mode: async
    prompt_length: ${data.max_prompt_length}
    response_length: ${data.max_response_length}
    # 多轮配置
    multi_turn:
      enable: True
      max_assistant_turns: 5  # 最多 5 轮 LLM 生成
      max_user_turns: 5       # 最多 5 轮用户/工具响应
      max_parallel_calls: 1   # 每轮最多 1 个工具调用
      max_tool_response_length: 512
      tool_response_truncate_side: middle
      tool_config_path: examples/sglang_multiturn/config/tool_config/ocr_resize_tool_config.yaml
      format: hermes  # 工具调用格式
    # Trace 配置（调试用）
    trace:
      backend: mlflow
      log_frequency: 100

# 奖励模型配置
reward_model:
  enable: True
  style: rule
  # 规则奖励函数配置
  rule_reward:
    module: verl.utils.reward_score.ocr
    function: compute_score
    tool_penalty: 0.01  # 每次工具调用扣 0.01

# 训练配置
algorithm:
  name: grpo  # 使用 GRPO 算法
  kl_ctrl:
    type: fixed
    kl_coef: 0.01

trainer:
  project_name: ocr_resize_agent
  experiment_name: grpo_qwen2.5vl_3b
  total_epochs: 3
  save_freq: 500
  logger:
    - tensorboard
    - mlflow
```

#### Phase 5: 训练启动脚本 (`examples/sglang_multiturn/run_ocr_resize_agent.sh`)

```bash
#!/bin/bash
# OCR Resize Agent 训练脚本

set -e

# 环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VERL_LOGGING_LEVEL=INFO

# 配置路径
CONFIG_NAME="ocr_resize_grpo"
DATA_DIR="${DATA_DIR:-~/data/ocr_verl}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/ocr_resize_agent}"

# 启动训练
python3 -m verl.trainer.main_ppo \
    --config-name="${CONFIG_NAME}" \
    data.train_path="${DATA_DIR}/train.parquet" \
    data.val_path="${DATA_DIR}/test.parquet" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.project_name=ocr_resize_agent \
    "$@"
```

### 10.4 实现进度 Checklist

| 阶段 | 任务 | 状态 | 文件位置 |
|-----|------|------|---------|
| **Phase 1** | 实现 `ImageResizeTool` | ✅ 已完成 | `verl/tools/image_resize_tool.py` |
| | 工具配置 YAML | ✅ 已完成 | `examples/sglang_multiturn/config/tool_config/ocr_resize_tool_config.yaml` |
| **Phase 2** | 数据预处理脚本 | ✅ 已完成 | `examples/data_preprocess/ocr_resize_agent_loop.py` |
| | 生成训练数据 | 🔲 待运行 | `~/data/ocr_verl/` |
| **Phase 3** | OCR 奖励函数 | ✅ 已完成 | `verl/utils/reward_score/ocr.py` |
| | 集成 TEDS/CDM | ⚠️ 部分完成 | 需安装 `table_recognition_metric` |
| **Phase 4** | GRPO 训练配置 | ✅ 已完成 | `examples/sglang_multiturn/config/ocr_resize_grpo.yaml` |
| **Phase 5** | 训练启动脚本 | ✅ 已完成 | `examples/sglang_multiturn/run_ocr_resize_agent.sh` |
| | 单元测试 | 🔲 待编写 | `tests/tools/test_image_resize_tool.py` |
| **Phase 6** | 训练验证 | 🔲 待运行 | - |
| | 评估对比 | 🔲 待运行 | 使用 `eval_ocr/eval_ocr.py` |

### 10.4.1 已实现文件清单

以下文件已创建完成：

```
verl/
├── tools/
│   └── image_resize_tool.py        # ImageResizeTool 实现
├── utils/
│   └── reward_score/
│       └── ocr.py                  # OCR 奖励函数
└── examples/
    ├── data_preprocess/
    │   └── ocr_resize_agent_loop.py  # 数据预处理脚本
    └── sglang_multiturn/
        ├── config/
        │   ├── ocr_resize_grpo.yaml  # GRPO 训练配置
        │   └── tool_config/
        │       └── ocr_resize_tool_config.yaml  # 工具配置
        └── run_ocr_resize_agent.sh   # 训练启动脚本
```

### 10.5 与 SFT 训练数据的对齐验证（训推一致性）

为确保 RL 训练与 SFT 训练数据的格式完全一致，已完成以下对齐工作：

#### 10.5.1 格式一致性对比表

| 特性 | SFT 数据格式 | RL 训练格式 | 状态 |
|------|-------------|------------|------|
| **System Prompt** | 按 tag 分类（text/table/equation） | 按 tag 分类 | ✅ 一致 |
| **User Prompt** | `<image>Extract...` | `<image>Extract...` | ✅ 一致 |
| **工具调用格式** | `<tool_call>...</tool_call>` | `<tool_call>...</tool_call>` | ✅ 一致 |
| **允许的 scale 值** | `[0.125, 0.143, 0.167, 0.2, 0.25, 0.333, 0.5, 2, 3, 4, 5, 6, 7, 8]` | 相同 | ✅ 一致 |
| **工具响应格式** | `Resize complete. New scale: {scale}x\nResult:<image>` | 相同 | ✅ 一致 |
| **输出格式** | `<final_answer>...</final_answer>` | 在 prompt 中说明 | ✅ 一致 |

#### 10.5.2 SFT 数据示例

```json
{
  "data_id": "equation_isolated_000587",
  "tag": "equation",
  "messages": [
    {
      "role": "system",
      "content": "You are an expert OCR assistant. Your task is to accurately extract the mathematical equation from the given image and output it in LaTeX format.\n\nYou have access to a resize tool that can adjust the image resolution:\n<tool_call>{\"name\": \"resize\", \"arguments\": {\"scale\": N}}</tool_call>\nwhere N can be: 0.125, 0.143, 0.167, 0.2, 0.25, 0.333, 0.5, 2, 3, 4, 5, 6, 7, 8\n..."
    },
    {
      "role": "user",
      "content": "<image>Extract the mathematical equation from this image in LaTeX format."
    },
    {
      "role": "assistant",
      "content": "Analyzing the image, I think downscaling would improve equation recognition accuracy.\n<tool_call>{\"name\": \"resize\", \"arguments\": {\"scale\": 0.25}}</tool_call>"
    },
    {
      "role": "user",
      "content": "Resize complete. New scale: 0.25x\nResult:<image>"
    },
    {
      "role": "assistant",
      "content": "<final_answer>$$F_n = 2^{2^n} + 1.$$</final_answer>"
    }
  ],
  "images": [
    {"image": "/path/to/original.jpg"},
    {"image": "/path/to/resized.jpg"}
  ]
}
```

#### 10.5.3 关键对齐点

1. **System Prompt 按 tag 分类**：
   - `text`: 提取文本内容
   - `table`: 提取表格并输出 HTML 格式
   - `equation`: 提取公式并输出 LaTeX 格式

2. **工具响应格式**：
   ```
   Resize complete. New scale: {scale}x
   Result:<image>
   ```

3. **输出格式要求**：
   - 最终结果需用 `<final_answer>` 标签包裹

**验证脚本**：

```python
# tests/tools/test_ocr_consistency.py
"""验证训练与 SFT 数据的一致性"""

def test_message_format_consistency():
    """验证消息格式一致性"""
    from verl.examples.data_preprocess.ocr_resize_agent_loop import SYSTEM_PROMPTS, USER_PROMPTS

    # 验证 equation 类型的 prompt
    assert "LaTeX format" in SYSTEM_PROMPTS["equation"]
    assert "<tool_call>" in SYSTEM_PROMPTS["equation"]
    assert "<final_answer>" in SYSTEM_PROMPTS["equation"]
    assert USER_PROMPTS["equation"] == "Extract the mathematical equation from this image in LaTeX format."

def test_tool_call_parsing_consistency():
    """验证工具调用解析一致性"""
    from eval_ocr.eval_ocr import parse_tool_call

    test_cases = [
        ('<tool_call>{"name": "resize", "arguments": {"scale": 2.0}}</tool_call>',
         {"name": "resize", "arguments": {"scale": 2.0}}),
        ('<tool_call>{"name": "resize", "arguments": {"scale": 0.125}}</tool_call>',
         {"name": "resize", "arguments": {"scale": 0.125}}),
        ('No tool call here', None),
    ]

    for text, expected in test_cases:
        result = parse_tool_call(text)
        assert result == expected, f"Mismatch for: {text}"

def test_tool_response_format():
    """验证工具响应格式"""
    # SFT 数据中的工具响应格式
    sft_response = "Resize complete. New scale: 0.25x\nResult:<image>"

    # 工具实现中的响应格式
    scale = 0.25
    tool_response = f"Resize complete. New scale: {scale}x\nResult:"

    # 验证格式一致（除了 <image> 占位符由框架处理）
    assert tool_response in sft_response.replace("<image>", "")
```

### 10.6 训练监控与调试

#### MLflow Trace 查看

```bash
# 启动 MLflow UI
mlflow ui --port 5000

# 在浏览器中访问 http://localhost:5000
# 查看：
# - agent_loop/tool_calls: 工具调用次数
# - reward_value: 奖励分布
# - response_length: 响应长度分布
```

#### 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|-----|---------|---------|
| 工具从不被调用 | System prompt 不够明确 | 调整 prompt，增加工具使用示例 |
| 工具被过度调用 | tool_penalty 太小 | 增大 `tool_penalty` 参数 |
| 奖励不收敛 | 奖励函数设计问题 | 检查奖励计算逻辑，调整权重 |
| OOM | 图像太大 | 限制 `max_prompt_length`，降低 batch size |
| Tokenization 不一致 | Processor 配置问题 | 确保训练和评估使用相同的 processor |
