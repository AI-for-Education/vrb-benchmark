"""Test cases for the run module."""

import pytest

from vrb_benchmark.run import extract_answer


@pytest.fixture
def options():
    """Fixture providing standard options."""
    return {
        "alpha_upper": ["A", "B", "C", "D"],
        "alpha_lower": ["a", "b", "c", "d"],
        "numeric": ["1", "2", "3", "4"],
    }


class TestExtractAnswer:
    """Test answer extraction from model responses."""

    def test_extract_single_letter_last_line(self, options):
        """Test extracting answer from last line."""
        response = "Let me think about this.\nThe answer is clearly B.\nB"
        result = extract_answer(response, options, "alpha_upper")
        assert result == "B"

    def test_extract_answer_with_text(self, options):
        """Test extracting when answer is in text."""
        response = "After analysis, the answer is: C"
        result = extract_answer(response, options, "alpha_upper")
        assert result == "C"

    def test_extract_final_answer_pattern(self, options):
        """Test extracting 'final answer' pattern."""
        response = "My reasoning leads to the final answer: D"
        result = extract_answer(response, options, "alpha_upper")
        assert result == "D"

    def test_extract_lowercase_options(self, options):
        """Test extracting lowercase options."""
        response = "The correct option is b"
        result = extract_answer(response, options, "alpha_lower")
        assert result == "b"

    def test_extract_numeric_options(self, options):
        """Test extracting numeric options."""
        response = "The answer is 3"
        result = extract_answer(response, options, "numeric")
        assert result == "3"

    def test_extract_last_occurrence(self, options):
        """Test that last occurrence is extracted."""
        response = "First I thought A, but actually it's C"
        result = extract_answer(response, options, "alpha_upper")
        assert result == "C"

    def test_extract_with_markdown(self, options):
        """Test extracting answer with markdown formatting."""
        response = "The answer is **B**"
        result = extract_answer(response, options, "alpha_upper")
        assert result == "B"

    def test_no_answer_found(self, options):
        """Test when no valid answer is found."""
        response = "I cannot determine the answer from this image."
        result = extract_answer(response, options, "alpha_upper")
        assert result is None

    def test_empty_response(self, options):
        """Test with empty response."""
        result = extract_answer("", options, "alpha_upper")
        assert result is None

    def test_none_response(self, options):
        """Test with None response."""
        result = extract_answer(None, options, "alpha_upper")
        assert result is None

    def test_answer_tag_format(self, options):
        """Test extracting from answer tags."""
        response = "Let me analyze: <answer>B</answer>"
        result = extract_answer(response, options, "alpha_upper")
        assert result == "B"

    def test_case_insensitive_matching(self, options):
        """Test case insensitive for alpha_upper."""
        response = "The answer is b"
        result = extract_answer(response, options, "alpha_upper")
        # Should convert to uppercase
        assert result == "B"
