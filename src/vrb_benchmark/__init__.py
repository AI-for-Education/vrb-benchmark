from pathlib import Path

from dotenv import load_dotenv
from fdllm import register_models

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

load_dotenv(PROJECT_ROOT / ".env")
register_models(PROJECT_ROOT / "custom_models.yaml")

from .load import call_model, get_image_path, load_questions, setup_models
from .run import extract_answer, run_benchmark, run_evaluation, run_question

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "call_model",
    "get_image_path",
    "load_questions",
    "setup_models",
    "extract_answer",
    "run_benchmark",
    "run_evaluation",
    "run_question",
]
