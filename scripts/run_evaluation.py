"""
Run full benchmark evaluation on a dataset with a specific model.

Usage:
    python run_evaluation.py --model gpt-4o --dataset test
    python run_evaluation.py --model claude-3-5-sonnet-20241022 --dataset zambia --max-tokens 50000

Examples:
    # Run GPT-4o on test dataset
    python run_evaluation.py --model gpt-4o --dataset test

    # Run Claude on your benchmark with custom token limit
    python run_evaluation.py --model claude-3-5-sonnet-20241022 --dataset your-dataset --max-tokens 50000

    # No resume - start fresh
    python run_evaluation.py --model gpt-4o --dataset test --no-resume
"""

import argparse

from dotenv import load_dotenv

from src.vrb_benchmark.load import setup_models
from src.vrb_benchmark.run import run_evaluation

# Load environment variables
load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Run VRB benchmark evaluation on a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        required=True,
        help="Model name to evaluate (e.g., gpt-4o, claude-3-5-sonnet-20241022)",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="test",
        help="Dataset name (default: test)",
    )
    parser.add_argument(
        "--max-tokens",
        "-t",
        type=int,
        default=33000,
        help="Maximum tokens for model response (default: 33000)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh instead of resuming from previous run",
    )

    args = parser.parse_args()

    # Setup custom models if custom_models.yaml exists
    try:
        setup_models()
        print("✓ Custom models loaded successfully")
    except Exception as e:
        print(f"Note: Could not load custom models: {e}")
        print("Using default models only")

    print(f"\n{'=' * 60}")
    print("Running benchmark evaluation:")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Resume: {not args.no_resume}")
    print(f"{'=' * 60}\n")

    # Run evaluation
    results = run_evaluation(
        model_name=args.model,
        dataset=args.dataset,
        resume=not args.no_resume,
        max_tokens=args.max_tokens,
    )

    print(f"\n{'=' * 60}")
    print("Evaluation complete!")
    print(f"Results saved to: outputs/{args.model}_{args.dataset}.csv")
    print(f"Total questions processed: {len(results)}")
    print(f"{'=' * 60}\n")

    # Show summary
    if "error" in results.columns:
        errors = results["error"].notna().sum()
        if errors > 0:
            print(f"⚠ {errors} questions had errors")

    if "model_answer" in results.columns:
        answered = results["model_answer"].notna().sum()
        print(f"✓ {answered}/{len(results)} questions answered successfully")


if __name__ == "__main__":
    main()
