"""
Simple example script to run the benchmark on one question.

Usage:
    python scripts/example_benchmark.py

Make sure you have set your API keys in a .env file:
    OPENAI_API_KEY=your_key_here
    ANTHROPIC_API_KEY=your_key_here
    # etc.
"""

from dotenv import load_dotenv

from src.vrb_benchmark.load import setup_models
from src.vrb_benchmark.run import run_evaluation

# Load environment variables from .env
load_dotenv()

# Setup models (registers custom models if custom_models.yaml exists)
try:
    setup_models()
    print("Custom models loaded successfully")
except Exception as e:
    print(f"Note: Could not load custom models: {e}")
    print("Using default models only")

# Run evaluation on test dataset
# You can change the model_name to any supported model
# Examples: "gpt-4o", "claude-3-5-sonnet-20241022", "gemini-2.0-flash-exp"
model_name = "gpt-4o"
dataset = "test"

print(f"\nRunning benchmark with {model_name} on {dataset} dataset...")
results = run_evaluation(model_name=model_name, dataset=dataset, resume=True)

print("\n=== Results ===")
print(results)
print(f"\nFull results saved to: outputs/{model_name}_{dataset}.csv")
