import time
from pathlib import Path

import pandas as pd
from fdllm import get_caller, register_models
from fdllm.llmtypes import LLMImage, LLMMessage
from jinja2 import Template
from PIL import Image

# configurations
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RESULTS_DIR = DATA_DIR / "results"


# dataset and model settings
DATASET = "zambia"  # or india
DEFAULT_MAX_TOKENS = 33000

# template

TEMPLATE_BOTH = """
Please answer the following question about the image.
{{ question }}

The image contains four possible answers labelled {% for label in options_labels %}{{ label }}{% if not loop.last %}, {% else %}.{% endif %}{% endfor %}
Please think step by step and consider each option carefully.
Give your final answer option (e.g. one of {% for label in options_labels %}{{ label }}{% if not loop.last %}, {% endif %}{% endfor %}) as a single character on a new line at the end of your response.
"""


def load_options(option_file: str):
    "Load options from a CSV file."
    options_df = pd.read_csv(option_file)
    option_mapping = dict(zip(options_df["option_id"], options_df["values"]))
    for key, value in option_mapping.items():
        option_mapping[key] = value.split(",")
    return option_mapping


def render_question_template(question, options_labels):
    "render the question template with options"
    template = Template(TEMPLATE_BOTH)
    return template.render(question=question, options_labels=options_labels)


def load_questions(dataset="zambia"):
    """Load gold standard questions from CSV file for specified dataset."""
    csv_path = DATA_DIR / f"{dataset}-questions-gold.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} gold standard questions from {csv_path}")
    return df


def get_image_path(question_id, dataset="zambia"):
    """Get the path to the image associated with a question."""
    image_dir = DATA_DIR / f"{dataset}-images"
    return image_dir / f"{question_id}.png"


def setup_models():
    """
    Register custom models
    """
    # First try loading from fab-benchmarks-configs submodule
    config_path = PROJECT_ROOT / "fab-benchmarks-configs" / "custom_models.yaml"

    # Fallback to local custom_models.yaml if not found in submodule
    if not config_path.exists():
        config_path = PROJECT_ROOT / "custom_models.yaml"

    register_models(config_path)
    # print("Updated registered models:")
    # available_models = list_models()
    # for model in available_models:
    #    print(model)


def call_model(
    question,
    image_path,
    model_name,
    option_id="alpha_upper",
    max_tokens=DEFAULT_MAX_TOKENS,
):
    """
    Call the LLM model with the question and image.
    """
    options = load_options(DATA_DIR / "options.csv")
    options_labels = options[option_id]
    prompt = render_question_template(question, options_labels)
    image = Image.open(image_path)
    message = LLMMessage(
        Role="user",
        Message=prompt,
        Images=LLMImage.list_from_images([image], detail="high"),
    )
    caller = get_caller(model_name)

    if "gemini" in model_name.lower():
        time.sleep(1)

    return caller.call([message], max_tokens=max_tokens)
