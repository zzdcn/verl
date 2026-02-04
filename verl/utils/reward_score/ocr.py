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
OCR reward functions supporting three data types:
- text: Edit distance similarity
- table: TEDS (Tree-Edit-Distance-based Similarity) score
- formula: CDM (Character Detection Matching) or fallback to edit distance

This module provides reward computation for OCR tasks in verl Agentic RL training.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text for comparison.

    Args:
        text: Input text string

    Returns:
        Normalized text with extra whitespace removed
    """
    if not text:
        return ""
    # Remove extra whitespace and normalize
    text = re.sub(r"\s+", " ", text.strip())
    return text


def compute_edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance (number of operations needed to transform s1 to s2)
    """
    try:
        import editdistance

        return editdistance.eval(s1, s2)
    except ImportError:
        # Fallback implementation if editdistance not installed
        if len(s1) < len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]


def compute_edit_distance_similarity(pred: str, gt: str) -> float:
    """Compute edit distance similarity normalized to [0, 1].

    Args:
        pred: Predicted text
        gt: Ground truth text

    Returns:
        Similarity score where 1.0 means exact match, 0.0 means completely different
    """
    pred_norm = normalize_text(pred)
    gt_norm = normalize_text(gt)

    if not gt_norm:
        return 1.0 if not pred_norm else 0.0

    distance = compute_edit_distance(pred_norm, gt_norm)
    max_len = max(len(pred_norm), len(gt_norm))

    if max_len == 0:
        return 1.0

    similarity = 1.0 - (distance / max_len)
    return max(0.0, similarity)


def compute_teds_score(pred: str, gt: str) -> dict[str, float]:
    """Compute TEDS (Tree-Edit-Distance-based Similarity) score for tables.

    Requires: pip install table_recognition_metric

    Args:
        pred: Predicted HTML table string
        gt: Ground truth HTML table string

    Returns:
        Dictionary with 'teds' and 'teds_struct' scores
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
        logger.warning("table_recognition_metric not installed. Falling back to edit distance for table evaluation.")
        # Fallback to edit distance if TEDS not available
        similarity = compute_edit_distance_similarity(pred, gt)
        return {
            "teds": similarity,
            "teds_struct": similarity,
        }
    except Exception as e:
        logger.error(f"Error computing TEDS score: {e}")
        return {"teds": 0.0, "teds_struct": 0.0}


def compute_formula_similarity(pred: str, gt: str) -> dict[str, float]:
    """Compute formula similarity using CDM or fallback methods.

    CDM (Character Detection Matching) requires LaTeX rendering capabilities.
    This implementation provides a simplified version using edit distance.

    Args:
        pred: Predicted formula string (LaTeX)
        gt: Ground truth formula string (LaTeX)

    Returns:
        Dictionary with 'recall', 'precision', and 'f1_score'
    """
    # Try to use CDM if available
    try:
        # CDM requires system dependencies (LaTeX, ImageMagick)
        # For now, we use edit distance as a reasonable approximation
        # TODO: Integrate full CDM implementation

        # Normalize LaTeX formulas
        pred_norm = normalize_latex(pred)
        gt_norm = normalize_latex(gt)

        similarity = compute_edit_distance_similarity(pred_norm, gt_norm)

        return {
            "recall": similarity,
            "precision": similarity,
            "f1_score": similarity,
        }
    except Exception as e:
        logger.error(f"Error computing formula similarity: {e}")
        return {
            "recall": 0.0,
            "precision": 0.0,
            "f1_score": 0.0,
        }


def normalize_latex(latex: str) -> str:
    """Normalize LaTeX formula for comparison.

    Args:
        latex: LaTeX formula string

    Returns:
        Normalized LaTeX string
    """
    if not latex:
        return ""

    # Remove common LaTeX delimiters
    latex = re.sub(r"^\$+|\$+$", "", latex)
    latex = re.sub(r"^\\[\[\(]|\\[\]\)]$", "", latex)

    # Normalize whitespace
    latex = re.sub(r"\s+", " ", latex.strip())

    # Remove common formatting commands that don't affect content
    latex = re.sub(r"\\displaystyle\s*", "", latex)
    latex = re.sub(r"\\textstyle\s*", "", latex)

    return latex


def compute_score(
    solution_str: str,
    ground_truth: str,
    tag: str = "text",
    tool_call_count: int = 0,
    tool_penalty: float = 0.01,
    **kwargs: Any,
) -> float:
    """Compute OCR reward score.

    This is the main entry point for reward computation, compatible with verl's
    rule-based reward system.

    Args:
        solution_str: Model's predicted OCR result
        ground_truth: Ground truth text
        tag: Data type tag - "text", "table", or "formula"
        tool_call_count: Number of tool calls made (for efficiency penalty)
        tool_penalty: Penalty per tool call (default: 0.01)
        **kwargs: Additional arguments (ignored)

    Returns:
        Reward score in range [0, 1] minus tool penalties
    """
    if solution_str is None:
        solution_str = ""

    # Compute base score based on data type
    if tag == "text":
        base_score = compute_edit_distance_similarity(solution_str, ground_truth)
    elif tag == "table":
        teds_result = compute_teds_score(solution_str, ground_truth)
        base_score = teds_result.get("teds", 0.0)
    elif tag == "formula":
        formula_result = compute_formula_similarity(solution_str, ground_truth)
        base_score = formula_result.get("f1_score", 0.0)
    else:
        # Default to edit distance for unknown types
        logger.warning(f"Unknown tag '{tag}', using edit distance similarity")
        base_score = compute_edit_distance_similarity(solution_str, ground_truth)

    # Apply tool call penalty (encourage efficient tool usage)
    penalty = tool_call_count * tool_penalty
    final_score = max(0.0, base_score - penalty)

    return final_score


def compute_detailed_metrics(
    solution_str: str,
    ground_truth: str,
    tag: str = "text",
    **kwargs: Any,
) -> dict[str, Any]:
    """Compute detailed metrics for analysis and logging.

    Args:
        solution_str: Model's predicted OCR result
        ground_truth: Ground truth text
        tag: Data type tag
        **kwargs: Additional arguments

    Returns:
        Dictionary with detailed metrics
    """
    if solution_str is None:
        solution_str = ""

    metrics = {
        "tag": tag,
        "pred_length": len(solution_str),
        "gt_length": len(ground_truth),
    }

    if tag == "text":
        metrics["edit_distance_similarity"] = compute_edit_distance_similarity(solution_str, ground_truth)
        metrics["edit_distance"] = compute_edit_distance(normalize_text(solution_str), normalize_text(ground_truth))
    elif tag == "table":
        teds_result = compute_teds_score(solution_str, ground_truth)
        metrics.update(teds_result)
    elif tag == "formula":
        formula_result = compute_formula_similarity(solution_str, ground_truth)
        metrics.update(formula_result)
    else:
        metrics["edit_distance_similarity"] = compute_edit_distance_similarity(solution_str, ground_truth)

    return metrics
