# %%

import pandas as pd

from vrb_benchmark import PROJECT_ROOT

# %%
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_DIR = PROJECT_ROOT / "data"
# Model name fixes
MODEL_NAME_FIXES = {
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "claude-3-5-haiku-latest": "claude-3-5-haiku-20241022",
}


def fix_model_name(name):
    return MODEL_NAME_FIXES.get(name, name)


def load_questions(which):
    return pd.read_csv(DATA_DIR / f"{which}-questions-gold.csv")


def load_all_results():
    """Load all result files and calculate overall accuracy"""
    files = list(OUTPUTS_DIR.glob("*.csv"))
    combined = pd.concat(
        [
            pd.read_csv(f).assign(
                dataset="zambia" if "zambia" in f.stem.lower() else "india",
                model=fix_model_name(f.stem.split("_")[0]),
            )
            for f in files
        ],
        ignore_index=True,
    )

    questions = pd.concat(
        [
            load_questions("zambia").assign(dataset="zambia"),
            load_questions("india").assign(dataset="india"),
        ],
        ignore_index=True,
    )

    merged = combined.merge(
        questions[["dataset", "id", "solution"]], on=["dataset", "id"]
    )
    merged["is_correct"] = merged["model_answer"] == merged["solution"]

    return (
        merged.groupby("model")["is_correct"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index()
        .rename(columns={"model": "model_id", "is_correct": "accuracy"})
        .sort_values("accuracy", ascending=False)
    )


def load_category_results():
    """Load results broken down by category"""
    files = list(OUTPUTS_DIR.glob("*.csv"))
    combined = pd.concat(
        [
            pd.read_csv(f).assign(
                dataset="zambia" if "zambia" in f.stem.lower() else "india",
                model=fix_model_name(f.stem.split("_")[0]),
            )
            for f in files
        ],
        ignore_index=True,
    )

    qz = pd.read_csv(DATA_DIR / "zambia-questions-categories-gold.csv").assign(
        dataset="zambia"
    )
    qi = pd.read_csv(DATA_DIR / "india-questions-categories-gold.csv").assign(
        dataset="india"
    )
    questions = pd.concat([qz, qi], ignore_index=True)

    merged = combined.merge(
        questions[["dataset", "id", "solution", "category"]],
        on=["dataset", "id"],
        how="left",
    )
    merged["is_correct"] = merged["model_answer"] == merged["solution"]

    return (
        merged.groupby(["model", "category"])["is_correct"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index()
        .rename(columns={"model": "model_id", "is_correct": "accuracy"})
    )


def load_individual_results():
    """Load results broken down by model"""
    files = list(OUTPUTS_DIR.glob("*.csv"))
    combined = pd.concat(
        [
            pd.read_csv(f).assign(
                dataset="zambia" if "zambia" in f.stem.lower() else "india",
                model=fix_model_name(f.stem.split("_")[0]),
            )
            for f in files
        ],
        ignore_index=True,
    )

    qz = pd.read_csv(DATA_DIR / "zambia-questions-categories.csv").assign(
        dataset="zambia"
    )
    qi = pd.read_csv(DATA_DIR / "india-questions-categories.csv").assign(
        dataset="india"
    )
    questions = pd.concat([qz, qi], ignore_index=True)

    merged = combined.merge(
        questions[["dataset", "id", "solution", "category"]],
        on=["dataset", "id"],
        how="left",
    )
    merged["is_correct"] = merged["model_answer"] == merged["solution"]

    return merged


def load_metadata():
    """Load model and provider metadata"""
    # First try loading from fab-benchmarks-configs submodule
    config_dir = PROJECT_ROOT / "fab-benchmarks-configs"
    models_path = config_dir / "models.csv"
    providers_path = config_dir / "providers.csv"

    # Fallback to local data directory if not found in submodule
    if not models_path.exists():
        models_path = DATA_DIR / "models.csv"
    if not providers_path.exists():
        providers_path = DATA_DIR / "providers.csv"

    try:
        models_df = pd.read_csv(models_path)
        providers_df = pd.read_csv(providers_path)
        return models_df, providers_df
    except FileNotFoundError:
        return None, None


def add_metadata(df):
    """Add metadata to results dataframe"""
    models_df, _ = load_metadata()

    if models_df is not None:
        df = df.merge(models_df, on="model_id", how="left")
        df["display_name"] = df["display_name"].fillna(df["model_id"])

        for col in ["provider", "open", "url", "input_cost", "output_cost"]:
            if col not in df.columns:
                df[col] = None
    else:
        for col in [
            "display_name",
            "provider",
            "open",
            "url",
            "input_cost",
            "output_cost",
        ]:
            df[col] = (
                df["model_id"]
                if col == "display_name"
                else ("Unknown" if col == "provider" else None)
            )

    return df
