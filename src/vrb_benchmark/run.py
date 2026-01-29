import os
import re

import pandas as pd
from tqdm import tqdm

from .load import DEFAULT_MAX_TOKENS, OUTPUTS_DIR, call_model, get_image_path


def extract_answer(response_text, options, option_id="alpha_upper"):
    """Extract answer based on the option_id format."""
    if not response_text:
        return None

    # Get valid answers for this format
    valid_answers = options.get(option_id, ["A", "B", "C", "D"])

    # Create dynamic regex pattern
    pattern_chars = "|".join(re.escape(ans) for ans in valid_answers)
    single_pattern = f"([{pattern_chars}])"

    # Check the last line
    last_line = response_text.strip().split("\n")[-1].strip()
    last_line = last_line.replace("*", "").replace("_", "").replace("$", "")

    if last_line in valid_answers:
        return last_line

    RE_BOX_TOKEN = re.compile(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", re.DOTALL)

    m = RE_BOX_TOKEN.search(last_line)
    if not m:
        m = RE_BOX_TOKEN.search(response_text)

    if m:
        candidate = m.group(1).strip()
        # Keep behavior consistent with option casing rules
        if option_id == "alpha_upper":
            candidate_cmp = candidate.upper()
        elif option_id == "alpha_lower":
            candidate_cmp = candidate.lower()
        else:
            candidate_cmp = candidate

        if candidate_cmp in valid_answers:
            return candidate_cmp

    # Look for LaTeX boxed answers
    for box_pattern in [
        rf"\\boxed\{{({pattern_chars})\}}",
        rf"\$\\boxed\{{({pattern_chars})\}}\$",
    ]:
        match = re.search(box_pattern, response_text)
        if match:
            return match.group(1)

    # Look for <answer> tags
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    match = answer_pattern.search(response_text)
    if match:
        answer_text = match.group(1).strip()
        if answer_text in valid_answers:
            return answer_text

    # Look for common answer patterns
    answer_patterns = [
        rf"(?:final\s+answer|answer|option)(?:\s*:|is\s*:?)\s*({pattern_chars})$",
        rf"(?:the\s+)?answer\s+is\s+({pattern_chars})",
    ]

    for pattern in answer_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            found = match.group(1)
            # return found.upper() if option_id == "alpha_upper" else found
            if option_id == "alpha_upper":
                return found.upper()
            elif option_id == "alpha_lower":
                return found.lower()
            else:
                return found

    # Search for specific answer patterns in text
    for answer in valid_answers:
        patterns = [
            rf"\bfinal\s+answer\s*(?:is|:)\s*{re.escape(answer)}\b",
            rf"\banswer\s*(?:is|:)\s*{re.escape(answer)}\b",
            rf"\bsolution\s*(?:is|:)\s*{re.escape(answer)}\b",
            rf"\bchoose\s*{re.escape(answer)}\b",
            rf"\bselect\s*{re.escape(answer)}\b",
            rf"\boption\s+{re.escape(answer)}\s+is\s+correct\b",
            rf"\boption\s+{re.escape(answer)}\b",
        ]

        for pattern in patterns:
            if re.search(pattern, response_text, re.IGNORECASE):
                return answer

    # Last resort
    clean_text = re.sub(r"\$[^$]*\$", "", response_text)
    all_matches = re.findall(single_pattern, clean_text)
    if all_matches:
        return all_matches[-1]

    return None


def run_question(
    question_data, model_name, dataset="zambia", max_tokens=DEFAULT_MAX_TOKENS
):
    """
    Process a single question with the model and return results.
    """
    from .load import DATA_DIR, load_options

    image_path = get_image_path(question_data["id"], dataset=dataset)

    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        return {
            "id": question_data["id"],
            "question": question_data["question"],
            "model_response": "",
            "model_answer": None,
            # "solution": question_data.get("solution"),
            "model": model_name,
            "dataset": dataset,
            "error": "Image not found",
        }

    try:
        options = load_options(DATA_DIR / "options.csv")
        # Call model using the function from load.py
        response = None
        for attempt in range(3):
            try:
                response = call_model(
                    question_data["question"],
                    image_path,
                    model_name,
                    option_id=question_data["option_id"],
                    max_tokens=max_tokens,
                )
                if response is not None:
                    break
            except Exception as e:
                if attempt == 2:
                    raise e

        if response is None:
            raise ValueError("Model returned None response after 3 attempts")

        model_answer = extract_answer(
            response.Message, options, option_id=question_data["option_id"]
        )

        result = {
            "id": question_data["id"],
            "question": question_data["question"],
            "model_response": response.Message,
            "model_answer": model_answer,
            # "solution": question_data.get("solution"),
            "model": model_name,
            "dataset": dataset,
            # "tokens_used": response.TokensUsed,
            # "token_used_completion": response.TokensUsedCompletion,
            # "token_used_reasoning": response.TokensUsedReasoning,
        }

        return result

    except Exception as e:
        print(f"ERROR processing question {question_data['id']}: {str(e)}")
        return {
            "id": question_data["id"],
            "question": question_data["question"],
            "model_response": "",
            "model_answer": None,
            # "solution": question_data.get("solution"),
            "model": model_name,
            "dataset": dataset,
            "error": str(e),
        }


def run_benchmark(
    df, model_name, dataset="zambia", resume=True, max_tokens=DEFAULT_MAX_TOKENS
):
    """
    Evaluate model on all questions in the dataframe.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUTS_DIR / f"{model_name}_{dataset}.csv"

    if resume and os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        processed_ids = set(existing_df["id"].tolist())
        results = existing_df.to_dict("records")
        print(f"Resuming: Found {len(processed_ids)} already processed questions")
    else:
        processed_ids = set()
        results = []

    for _, question_data in tqdm(
        df.iterrows(), total=len(df), desc=f"Evaluating {model_name} on {dataset}"
    ):
        if question_data["id"] in processed_ids:
            continue

        result = run_question(
            question_data, model_name, dataset=dataset, max_tokens=max_tokens
        )
        results.append(result)

        # save after each question (for crash recovery)
        pd.DataFrame(results).to_csv(output_file, index=False)

    final_df = pd.DataFrame(results)
    print(
        f"Evaluation complete for {model_name} on {dataset}. {len(final_df)} questions processed."
    )

    return final_df


def run_evaluation(
    model_name,
    dataset="zambia",
    resume=True,
    max_tokens=DEFAULT_MAX_TOKENS,
):
    """
    Complete evaluation pipeline - loads data, sets up models, runs evaluation.
    Note: You must call setup_models() before using this function to register custom models
    """
    from .load import load_questions

    # setup and run evaluation
    df = load_questions(dataset=dataset)
    results = run_benchmark(
        df, model_name, dataset=dataset, resume=resume, max_tokens=max_tokens
    )

    return results
