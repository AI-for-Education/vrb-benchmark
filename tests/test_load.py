"""Test cases for the load module."""

from pathlib import Path

import pandas as pd

from vrb_benchmark.load import (
    DATA_DIR,
    PROJECT_ROOT,
    get_image_path,
    load_options,
    load_questions,
    render_question_template,
)


def test_project_root_exists():
    """Test that PROJECT_ROOT points to a valid directory."""
    assert PROJECT_ROOT.exists()
    assert PROJECT_ROOT.is_dir()


def test_data_dir_exists():
    """Test that DATA_DIR exists."""
    assert DATA_DIR.exists()
    assert DATA_DIR.is_dir()


def test_load_options():
    """Test loading options from CSV."""
    options_file = DATA_DIR / "options.csv"
    assert options_file.exists(), "options.csv should exist"

    options = load_options(str(options_file))
    assert isinstance(options, dict)
    assert "alpha_upper" in options
    assert isinstance(options["alpha_upper"], list)
    assert "A" in options["alpha_upper"]
    assert "B" in options["alpha_upper"]
    assert "C" in options["alpha_upper"]
    assert "D" in options["alpha_upper"]


def test_load_questions_test_dataset():
    """Test loading test dataset questions."""
    df = load_questions(dataset="test")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "id" in df.columns
    assert "question" in df.columns
    assert "option_id" in df.columns


def test_get_image_path():
    """Test getting image path for a question."""
    image_path = get_image_path("Q001", dataset="test")
    assert isinstance(image_path, Path)
    assert image_path.name == "Q001.png"
    assert "test-images" in str(image_path)


def test_get_image_path_exists():
    """Test that test image actually exists."""
    image_path = get_image_path("Q001", dataset="test")
    assert image_path.exists(), f"Test image should exist at {image_path}"


def test_render_question_template():
    """Test rendering question template."""
    question = "How many circles are in this image?"
    options_labels = ["A", "B", "C", "D"]

    rendered = render_question_template(question, options_labels)

    assert isinstance(rendered, str)
    assert question in rendered
    assert "A" in rendered
    assert "B" in rendered
    assert "C" in rendered
    assert "D" in rendered
    assert "think step by step" in rendered.lower()


def test_render_question_template_different_options():
    """Test rendering with different option labels."""
    question = "What is the answer?"
    options_labels = ["1", "2", "3", "4"]

    rendered = render_question_template(question, options_labels)

    assert question in rendered
    assert "1" in rendered
    assert "2" in rendered
    assert "3" in rendered
    assert "4" in rendered
