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
Unit tests for OCR reward functions.

Tests cover:
- Text normalization
- Edit distance computation
- Edit distance similarity
- TEDS score (table evaluation)
- Formula similarity
- LaTeX normalization
- Main compute_score function
- Detailed metrics computation
"""

import pytest

from verl.utils.reward_score.ocr import (
    compute_detailed_metrics,
    compute_edit_distance,
    compute_edit_distance_similarity,
    compute_formula_similarity,
    compute_score,
    compute_teds_score,
    normalize_latex,
    normalize_text,
)


class TestNormalizeText:
    """Test normalize_text function."""

    def test_normalize_empty_string(self):
        """Test normalization of empty string."""
        assert normalize_text("") == ""

    def test_normalize_none(self):
        """Test normalization of None."""
        assert normalize_text(None) == ""

    def test_normalize_whitespace(self):
        """Test normalization of whitespace."""
        assert normalize_text("  hello   world  ") == "hello world"
        assert normalize_text("\t\nhello\n\tworld\n") == "hello world"
        assert normalize_text("hello    world") == "hello world"

    def test_normalize_preserve_content(self):
        """Test that content is preserved."""
        assert normalize_text("Hello, World!") == "Hello, World!"
        assert normalize_text("123 abc") == "123 abc"


class TestComputeEditDistance:
    """Test compute_edit_distance function."""

    def test_identical_strings(self):
        """Test edit distance of identical strings."""
        assert compute_edit_distance("hello", "hello") == 0

    def test_empty_strings(self):
        """Test edit distance with empty strings."""
        assert compute_edit_distance("", "") == 0
        assert compute_edit_distance("hello", "") == 5
        assert compute_edit_distance("", "world") == 5

    def test_single_character_difference(self):
        """Test edit distance with single character difference."""
        assert compute_edit_distance("hello", "hallo") == 1
        assert compute_edit_distance("hello", "jello") == 1

    def test_insertion_and_deletion(self):
        """Test edit distance with insertions and deletions."""
        assert compute_edit_distance("cat", "cats") == 1
        assert compute_edit_distance("cats", "cat") == 1

    def test_complete_difference(self):
        """Test edit distance of completely different strings."""
        assert compute_edit_distance("abc", "xyz") == 3

    def test_substring(self):
        """Test edit distance with substring."""
        assert compute_edit_distance("abc", "abcd") == 1


class TestComputeEditDistanceSimilarity:
    """Test compute_edit_distance_similarity function."""

    def test_identical_strings(self):
        """Test similarity of identical strings."""
        assert compute_edit_distance_similarity("hello", "hello") == 1.0

    def test_empty_strings(self):
        """Test similarity with empty strings."""
        assert compute_edit_distance_similarity("", "") == 1.0
        assert compute_edit_distance_similarity("hello", "") == 0.0
        assert compute_edit_distance_similarity("", "world") == 0.0

    def test_partial_match(self):
        """Test similarity with partial match."""
        # "hello" vs "hallo" - 1 edit, max_len = 5, similarity = 1 - 1/5 = 0.8
        assert compute_edit_distance_similarity("hello", "hallo") == pytest.approx(0.8)

    def test_complete_difference(self):
        """Test similarity of completely different strings."""
        # "abc" vs "xyz" - 3 edits, max_len = 3, similarity = 1 - 3/3 = 0
        assert compute_edit_distance_similarity("abc", "xyz") == pytest.approx(0.0)

    def test_whitespace_normalization(self):
        """Test that whitespace is normalized before comparison."""
        assert compute_edit_distance_similarity("hello  world", "hello world") == 1.0
        assert compute_edit_distance_similarity("  hello  ", "hello") == 1.0

    def test_realistic_ocr_example(self):
        """Test with realistic OCR example."""
        pred = "The quick brown fox"
        gt = "The quick brown fox"
        assert compute_edit_distance_similarity(pred, gt) == 1.0

        # With minor error
        pred_with_error = "The quick browm fox"  # 'n' -> 'm'
        similarity = compute_edit_distance_similarity(pred_with_error, gt)
        assert 0.9 < similarity < 1.0


class TestComputeTedsScore:
    """Test compute_teds_score function."""

    def test_identical_tables(self):
        """Test TEDS score for identical tables."""
        table_html = "<table><tr><td>A</td><td>B</td></tr></table>"
        result = compute_teds_score(table_html, table_html)

        assert "teds" in result
        assert "teds_struct" in result
        # Should be perfect score or fallback to edit distance
        assert result["teds"] >= 0.0
        assert result["teds_struct"] >= 0.0

    def test_empty_tables(self):
        """Test TEDS score for empty strings."""
        result = compute_teds_score("", "")
        assert "teds" in result
        assert "teds_struct" in result

    def test_different_tables(self):
        """Test TEDS score for different tables."""
        table1 = "<table><tr><td>A</td></tr></table>"
        table2 = "<table><tr><td>B</td></tr></table>"
        result = compute_teds_score(table1, table2)

        assert "teds" in result
        assert result["teds"] >= 0.0
        assert result["teds"] <= 1.0


class TestNormalizeLatex:
    """Test normalize_latex function."""

    def test_normalize_empty(self):
        """Test normalization of empty string."""
        assert normalize_latex("") == ""
        assert normalize_latex(None) == ""

    def test_remove_dollar_signs(self):
        """Test removal of dollar sign delimiters."""
        assert normalize_latex("$x^2$") == "x^2"
        assert normalize_latex("$$x^2$$") == "x^2"

    def test_remove_latex_delimiters(self):
        """Test removal of LaTeX delimiters."""
        assert normalize_latex("\\[x^2\\]") == "x^2"
        assert normalize_latex("\\(x^2\\)") == "x^2"

    def test_remove_display_commands(self):
        """Test removal of display style commands."""
        assert normalize_latex("\\displaystyle x^2") == "x^2"
        assert normalize_latex("\\textstyle x^2") == "x^2"

    def test_normalize_whitespace(self):
        """Test whitespace normalization in LaTeX."""
        assert normalize_latex("x^2  +  y^2") == "x^2 + y^2"
        assert normalize_latex("  \\frac{a}{b}  ") == "\\frac{a}{b}"


class TestComputeFormulaSimilarity:
    """Test compute_formula_similarity function."""

    def test_identical_formulas(self):
        """Test similarity of identical formulas."""
        formula = "x^2 + y^2 = z^2"
        result = compute_formula_similarity(formula, formula)

        assert "recall" in result
        assert "precision" in result
        assert "f1_score" in result
        assert result["f1_score"] == 1.0

    def test_empty_formulas(self):
        """Test similarity with empty formulas."""
        result = compute_formula_similarity("", "")
        assert result["f1_score"] >= 0.0

    def test_different_formulas(self):
        """Test similarity of different formulas."""
        result = compute_formula_similarity("x^2", "y^3")
        assert result["f1_score"] >= 0.0
        assert result["f1_score"] <= 1.0

    def test_formula_with_delimiters(self):
        """Test that delimiters are normalized."""
        formula1 = "$x^2$"
        formula2 = "x^2"
        result = compute_formula_similarity(formula1, formula2)
        assert result["f1_score"] == 1.0


class TestComputeScore:
    """Test main compute_score function."""

    def test_text_type(self):
        """Test compute_score with text type."""
        score = compute_score(
            solution_str="Hello World",
            ground_truth="Hello World",
            tag="text",
        )
        assert score == 1.0

    def test_text_type_with_error(self):
        """Test compute_score with text type and error."""
        score = compute_score(
            solution_str="Hello Warld",  # typo
            ground_truth="Hello World",
            tag="text",
        )
        assert 0.8 < score < 1.0

    def test_table_type(self):
        """Test compute_score with table type."""
        table = "<table><tr><td>A</td></tr></table>"
        score = compute_score(
            solution_str=table,
            ground_truth=table,
            tag="table",
        )
        assert score >= 0.0

    def test_formula_type(self):
        """Test compute_score with formula type."""
        formula = "x^2 + y^2"
        score = compute_score(
            solution_str=formula,
            ground_truth=formula,
            tag="formula",
        )
        assert score == 1.0

    def test_unknown_tag(self):
        """Test compute_score with unknown tag (should use edit distance)."""
        score = compute_score(
            solution_str="test",
            ground_truth="test",
            tag="unknown",
        )
        assert score == 1.0

    def test_none_solution(self):
        """Test compute_score with None solution."""
        score = compute_score(
            solution_str=None,
            ground_truth="Hello",
            tag="text",
        )
        assert score == 0.0

    def test_tool_penalty(self):
        """Test compute_score with tool call penalty."""
        base_score = compute_score(
            solution_str="Hello",
            ground_truth="Hello",
            tag="text",
            tool_call_count=0,
        )
        assert base_score == 1.0

        penalized_score = compute_score(
            solution_str="Hello",
            ground_truth="Hello",
            tag="text",
            tool_call_count=5,
            tool_penalty=0.01,
        )
        assert penalized_score == pytest.approx(0.95)  # 1.0 - 5 * 0.01

    def test_tool_penalty_floor(self):
        """Test that tool penalty doesn't go below 0."""
        score = compute_score(
            solution_str="Hello",
            ground_truth="Hello",
            tag="text",
            tool_call_count=200,
            tool_penalty=0.01,
        )
        assert score == 0.0  # max(0.0, 1.0 - 2.0) = 0.0


class TestComputeDetailedMetrics:
    """Test compute_detailed_metrics function."""

    def test_text_metrics(self):
        """Test detailed metrics for text type."""
        metrics = compute_detailed_metrics(
            solution_str="Hello World",
            ground_truth="Hello World",
            tag="text",
        )

        assert metrics["tag"] == "text"
        assert metrics["pred_length"] == 11
        assert metrics["gt_length"] == 11
        assert "edit_distance_similarity" in metrics
        assert "edit_distance" in metrics
        assert metrics["edit_distance_similarity"] == 1.0
        assert metrics["edit_distance"] == 0

    def test_table_metrics(self):
        """Test detailed metrics for table type."""
        table = "<table><tr><td>A</td></tr></table>"
        metrics = compute_detailed_metrics(
            solution_str=table,
            ground_truth=table,
            tag="table",
        )

        assert metrics["tag"] == "table"
        assert "teds" in metrics
        assert "teds_struct" in metrics

    def test_formula_metrics(self):
        """Test detailed metrics for formula type."""
        formula = "x^2"
        metrics = compute_detailed_metrics(
            solution_str=formula,
            ground_truth=formula,
            tag="formula",
        )

        assert metrics["tag"] == "formula"
        assert "recall" in metrics
        assert "precision" in metrics
        assert "f1_score" in metrics

    def test_unknown_tag_metrics(self):
        """Test detailed metrics for unknown tag."""
        metrics = compute_detailed_metrics(
            solution_str="test",
            ground_truth="test",
            tag="unknown",
        )

        assert metrics["tag"] == "unknown"
        assert "edit_distance_similarity" in metrics

    def test_none_solution_metrics(self):
        """Test detailed metrics with None solution."""
        metrics = compute_detailed_metrics(
            solution_str=None,
            ground_truth="Hello",
            tag="text",
        )

        assert metrics["pred_length"] == 0
        assert metrics["gt_length"] == 5


class TestOCRRealWorldExamples:
    """Test with realistic OCR examples."""

    def test_chinese_text(self):
        """Test OCR evaluation with Chinese text."""
        pred = "识别结果文本"
        gt = "识别结果文本"
        score = compute_score(pred, gt, tag="text")
        assert score == 1.0

    def test_chinese_text_with_error(self):
        """Test OCR evaluation with Chinese text containing error."""
        pred = "识别结呆文本"  # 果 -> 呆
        gt = "识别结果文本"
        score = compute_score(pred, gt, tag="text")
        assert 0.8 < score < 1.0

    def test_mixed_content(self):
        """Test OCR evaluation with mixed Chinese and English."""
        pred = "Hello 世界 123"
        gt = "Hello 世界 123"
        score = compute_score(pred, gt, tag="text")
        assert score == 1.0

    def test_multiline_text(self):
        """Test OCR evaluation with multiline text."""
        pred = "Line 1\nLine 2\nLine 3"
        gt = "Line 1\nLine 2\nLine 3"
        score = compute_score(pred, gt, tag="text")
        assert score == 1.0

    def test_latex_formula(self):
        """Test OCR evaluation with LaTeX formula."""
        pred = "$$\\frac{a+b}{c}$$"
        gt = "$\\frac{a+b}{c}$"
        score = compute_score(pred, gt, tag="formula")
        assert score == 1.0  # Should be equal after normalization


if __name__ == "__main__":
    pytest.main([__file__, "-v"])