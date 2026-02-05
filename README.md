# VRB Benchmark

A flexible framework for evaluating vision language models (VLMs) on visual reasoning tasks with multiple-choice questions.

## Overview

VRB Benchmark provides a standardized pipeline for:
- Loading visual reasoning questions with images
- Evaluating multiple VLM providers (OpenAI, Anthropic, Google, etc.)
- Extracting and validating model answers
- Tracking results across benchmark runs

## Features

- **Multi-provider support**: Works with OpenAI, Anthropic, Google, and many other LLM providers via [fabdata-llm](https://github.com/AI-for-Education/fabdata-llm)
- **Flexible question formats**: Support for different answer labeling schemes (A/B/C/D, 1/2/3/4, etc.)
- **Resume capability**: Automatically resumes evaluation from where it left off if interrupted
- **Robust answer extraction**: Multiple regex patterns to extract model answers from various response formats
- **Data-agnostic**: Bring your own visual reasoning questions and images

## Installation

### Requirements
- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

1. Clone this repository:
```bash
git clone https://github.com/AI-for-Education/vrb-benchmark.git
cd vrb-benchmark
```

2. Install dependencies:
```bash
pipx install uv
uv sync
```

To run the analysis scripts (in `scripts/`), install the optional analysis dependencies:
```bash
uv sync --extra analysis
```

3. Set up API keys:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

Required API keys depend on which models you want to evaluate. Common ones:
- `OPENAI_API_KEY` - For GPT models
- `ANTHROPIC_API_KEY` - For Claude models
- `GEMINI_API_KEY` - For Gemini models

## Quick Start

### Test with dummy data

Run the included example with a simple test:

```bash
uv run python scripts/example_benchmark.py
```

This evaluates GPT-4o on a single test question about counting circles.

### Add your own data

1. **Prepare your questions**: Create a CSV file in `data/<dataset>-questions-gold.csv`

See [DATA_FORMAT.md](DATA_FORMAT.md) for the complete data format specification.

2. **Add images**: Place corresponding images in `data/<dataset>-images/`

Each image should be named `{question_id}.png` (e.g., `Q001.png`)

3. **Run evaluation**:

```python
from dotenv import load_dotenv
from src.vrb_benchmark.load import setup_models
from src.vrb_benchmark.run import run_evaluation

load_dotenv()
setup_models()  # Optional: load custom model configs

results = run_evaluation(
    model_name="gpt-4o",
    dataset="your-dataset-name",
    resume=True
)
```

## Usage

### Command-line tool

The easiest way to run evaluations is using the CLI script:

```bash
# Run on test dataset
uv run python scripts/run_evaluation.py --model gpt-4o --dataset test

# Run with custom settings
uv run python scripts/run_evaluation.py --model claude-3-5-sonnet-20241022 --dataset your-dataset --max-tokens 50000

# Start fresh (no resume)
uv run python scripts/run_evaluation.py --model gpt-4o --dataset test --no-resume

# See all options
uv run python scripts/run_evaluation.py --help
```

### Basic evaluation script (Python API)

```python
from dotenv import load_dotenv
from src.vrb_benchmark.load import setup_models
from src.vrb_benchmark.run import run_evaluation

# Load API keys
load_dotenv()

# Setup custom models (optional)
setup_models()

# Run benchmark
results = run_evaluation(
    model_name="gpt-4o",           # Model to evaluate
    dataset="test",                 # Dataset name
    resume=True,                    # Resume from previous run
    max_tokens=33000                # Max tokens for model response
)

# Results saved to outputs/{model_name}_{dataset}.csv
print(results)
```

### Custom model configurations

To use custom model configurations or add new providers:

1. Copy the example config:
```bash
cp custom_models.yaml.example custom_models.yaml
```

2. Edit `custom_models.yaml` to add your models

3. The `setup_models()` call will automatically load your configurations

See [fabdata-llm documentation](https://github.com/AI-for-Education/fabdata-llm) for model configuration options.

## Data Format

Your benchmark data should follow this structure:

```
data/
├── options.csv                        # Answer format definitions
├── <dataset>-questions-gold.csv       # Your questions
└── <dataset>-images/                  # Your images
    ├── Q001.png
    ├── Q002.png
    └── ...
```

See [DATA_FORMAT.md](DATA_FORMAT.md) for detailed specifications.

## Results and Outputs

The benchmark generates comprehensive outputs stored in the `outputs/` directory.

### Output Files

Results are saved to `outputs/{model_name}_{dataset}.csv` after each evaluation. The benchmark automatically resumes from where it left off if interrupted, making it safe for long-running evaluations.

### Output Format

Each CSV file contains the following columns:

- **`id`**: Unique question identifier (matches image filename)
- **`question`**: The question text presented to the model
- **`model_response`**: Full text response from the model, including reasoning and final answer
- **`model_answer`**: Extracted final answer (e.g., "A", "B", "C", "D") using robust regex patterns
- **`model`**: Model name/identifier used for evaluation
- **`dataset`**: Dataset name (e.g., "test", "zambia")
- **`error`**: Error message if the question failed (empty if successful)

### Interpreting Results

**Successful responses:**
- `model_answer` contains the extracted answer
- `error` column is empty
- Full reasoning available in `model_response`

**Failed responses:**
- `model_answer` is `None` or empty
- `error` column contains diagnostic information (e.g., "Image not found", API errors)
- Check `model_response` to see if the model generated output before failure

### Calculating Accuracy

To calculate accuracy, compare `model_answer` with ground truth solutions from your questions CSV:

```python
import pandas as pd

results = pd.read_csv("outputs/gpt-4o_test.csv")
questions = pd.read_csv("data/test-questions-gold.csv")

merged = results.merge(questions[["id", "solution"]], on="id")
merged["correct"] = merged["model_answer"] == merged["solution"]
accuracy = merged["correct"].mean() * 100

print(f"Accuracy: {accuracy:.2f}%")
```

### Using Results Utilities

The `src/vrb_benchmark/results.py` module provides utilities for:
- Loading and combining results across multiple models
- Adding model metadata (provider, cost, display names)
- Aggregating accuracy by category or skill
- Generating leaderboards

See `data/models.csv` and `data/providers.csv` for model metadata.

## Repository Structure

```
vrb-benchmark/
├── src/vrb_benchmark/           # Core package
│   ├── __init__.py              # Package initialization, environment setup
│   ├── load.py                  # Data loading, image processing, model setup
│   ├── run.py                   # Benchmark execution, answer extraction
│   └── results.py               # Results loading, metadata, analysis utilities
├── scripts/                     # Main entry points
│   ├── example_benchmark.py     # Quick example with dummy data
│   └── run_evaluation.py        # Full CLI tool for running evaluations
├── data/                        # Benchmark data (not included in repo)
│   ├── options.csv              # Answer format definitions
│   ├── models.csv               # Model metadata (provider, cost, etc.)
│   ├── providers.csv            # Provider metadata (colors, logos)
│   ├── <dataset>-questions-gold.csv    # Your questions
│   └── <dataset>-images/        # Question images
├── outputs/                     # Evaluation results (gitignored)
│   └── {model}_{dataset}.csv    # Per-model results
├── tests/                       # Unit tests
├── custom_models.yaml.example   # Template for custom model configs
├── .env.example                 # Template for API keys
├── pyproject.toml               # Python dependencies and project metadata
├── uv.lock                      # Locked dependencies for reproducibility
└── README.md                    # This file
```

### Key Files

- **`custom_models.yaml.example`**: Template for configuring 100+ LLM providers via [fabdata-llm](https://github.com/AI-for-Education/fabdata-llm)
- **`data/models.csv`**: Metadata for models (provider, cost, vision support)
- **`data/providers.csv`**: Provider information (colors, SVG logos for visualization)
- **`src/vrb_benchmark/load.py`**: Core data loading and model calling logic
- **`src/vrb_benchmark/run.py`**: Answer extraction with multiple regex patterns

## Contributing

Contributions are welcome! We appreciate:
- Bug reports and feature requests via GitHub Issues
- Pull requests for bug fixes or enhancements
- Additional model configurations
- Documentation improvements

Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests: `pytest`
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions, issues, or contributions:
- **GitHub Issues**: [github.com/vrb-benchmark/issues](https://github.com/vrb-benchmark/issues)

## Acknowledgments

This project is developed by [Fab Inc](https://fab.inc) and built on [fabdata-llm](https://github.com/AI-for-Education/fabdata-llm) for multi-provider LLM access.

Copyright (c) 2025 Fab Inc. Released under the MIT License.
