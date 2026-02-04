# %%
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from plot_utils import (
    FONT_BOLD,
    FONT_REGULAR,
    bootstrap_model_accuracies,
    plot_benchmark_df,
)

from vrb_benchmark import DATA_DIR, PROJECT_ROOT

PLOTS_DIR = PROJECT_ROOT / "data" / "plots_gold"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# %%
# GENERATE RESULTS CSVs FROM RAW MODEL OUTPUTS
# This section loads model outputs and creates the CSV files needed for visualisation

ZAMBIA_PATH = DATA_DIR / "zambia-questions-categories-gold.csv"
INDIA_PATH = DATA_DIR / "india-questions-categories-gold.csv"

# Models to exclude from analysis
EXCLUDE_MODELS = {
    "gpt-5-mini-2025-08-07-high",
    "gpt-5-2025-08-07-high",
    "gpt-5-nano-2025-08-07-high",
    "gpt-5-nano-2025-08-07-low",
    "gpt-5-mini-2025-08-07-low",
    "gpt-5-2025-08-07-low",
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.0-flash-thinking-exp-1219",
    "lfm2-vl-1.6b",
    "claude-3-7-sonnet-20250219-thinking-low",
    "gpt-5.1-2025-11-13-high",
    "claude-opus-4-5-20251101-thinking-high",
}


def load_catalog():
    """Load question catalog with correct answers and categories"""
    catalogs = []
    for dataset, path in [("zambia", ZAMBIA_PATH), ("india", INDIA_PATH)]:
        df = pd.read_csv(path)
        df = df.rename(columns=str.strip)
        df = df[["id", "solution", "category"]]
        df = df.assign(dataset=dataset)
        catalogs.append(df)

    catalog = pd.concat(catalogs, ignore_index=True)
    for c in ["id", "solution", "category", "dataset"]:
        catalog[c] = catalog[c].astype(str).str.strip()

    return catalog


def load_model_responses():
    """Load all model responses from output files"""
    files = list(OUTPUTS_DIR.glob("*.csv"))
    all_data = []

    for f in files:
        stem = f.stem
        model = stem.split("_")[0]

        if model in EXCLUDE_MODELS:
            continue

        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]

        if not {"id", "model_answer"}.issubset(df.columns):
            continue

        dataset = "zambia" if "zambia" in stem.lower() else "india"
        df = df[["id", "model_answer"]].copy()
        df["model"] = model
        df["dataset"] = dataset
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


print("Loading catalog and model responses...")
catalog = load_catalog()
combined = load_model_responses()

# Merge and score
eval_df = combined.merge(catalog, on=["dataset", "id"], how="inner")
eval_df["correct"] = (eval_df["model_answer"] == eval_df["solution"]).astype(int)

print(f"Loaded {len(eval_df)} responses from {eval_df['model'].nunique()} models")
print(f"Questions: {eval_df['id'].nunique()}")
print(f"Categories: {eval_df['category'].nunique()}")

# Generate visual_reasoning_accuracy_results_gold.csv
print("\nGenerating visual_reasoning_accuracy_results_gold.csv...")

model_dataset_accuracy = (
    eval_df.groupby(["model", "dataset"])["correct"].mean().reset_index()
)
model_dataset_accuracy = model_dataset_accuracy.rename(columns={"correct": "accuracy"})
model_dataset_accuracy.to_csv(
    DATA_DIR / "visual_reasoning_accuracy_results_gold.csv", index=False
)
print(f"Saved: {DATA_DIR / 'visual_reasoning_accuracy_results_gold.csv'}\n")

# %%
# Load results dataset
visual_reasoning_results = pd.read_csv(
    DATA_DIR / "visual_reasoning_accuracy_results_gold.csv"
)
print(visual_reasoning_results.shape)
visual_reasoning_results.head()

# %%
# prepare data for plotting
accuracies = visual_reasoning_results.set_index(["model"])
accuracies = accuracies[~accuracies.index.isin(EXCLUDE_MODELS)]

# check that all datasets have the same models
print(f"Number of unique models:\nIn accuracy DataFrame: {accuracies.index.nunique()}")

accuracies.head()

# %%
# Verify models
print("Models in CSV before any processing:")
print(f"Total: {len(visual_reasoning_results)}")
print(sorted(visual_reasoning_results["model"].tolist()))

print(f"\nModels to exclude: {EXCLUDE_MODELS}")

models_in_csv = set(visual_reasoning_results["model"])
excluded_models_found = [m for m in EXCLUDE_MODELS if m in models_in_csv]
excluded_models_missing = [m for m in EXCLUDE_MODELS if m not in models_in_csv]

print(f"\nExcluded models found in CSV: {excluded_models_found}")
print(f"Excluded models NOT in CSV: {excluded_models_missing}")
print(f"Expected final count: {len(models_in_csv) - len(excluded_models_found)}")

# %%
# Merge India and Zambia results
india_questions = pd.read_csv(DATA_DIR / "india-questions-gold.csv")
zambia_questions = pd.read_csv(DATA_DIR / "zambia-questions-gold.csv")

N_qu_india = india_questions.shape[0]
N_qu_zambia = zambia_questions.shape[0]

acc_india = visual_reasoning_results[visual_reasoning_results["dataset"] == "india"]
acc_zambia = visual_reasoning_results[visual_reasoning_results["dataset"] == "zambia"]

accuracies_combined = pd.DataFrame(index=accuracies.index.unique())
accuracies_combined["accuracy_india"] = acc_india.set_index("model")["accuracy"]
accuracies_combined["accuracy_zambia"] = acc_zambia.set_index("model")["accuracy"]

accuracies_combined["accuracy"] = (
    accuracies_combined["accuracy_india"] * N_qu_india
    + accuracies_combined["accuracy_zambia"] * N_qu_zambia
) / (N_qu_india + N_qu_zambia)
accuracies_combined.head()

# %%
# Reset index and prepare the dataframe structure
accuracies_combined = accuracies_combined.sort_values(
    "accuracy", ascending=False
).reset_index(drop=False)
if accuracies_combined.columns[0] != "model":
    accuracies_combined = accuracies_combined.rename(
        columns={accuracies_combined.columns[0]: "model"}
    )

# Apply model renaming FIRST (before metadata lookup)
renamed_models_dict = {
    "gemini-2.0-flash-thinking-exp-1219": "gemini-2.0-flash-thinking-exp-01-21",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "claude-3-5-haiku-latest": "claude-3-5-haiku-20241022",
}

accuracies_combined["model"] = accuracies_combined["model"].replace(renamed_models_dict)

# Now add metadata about models (AFTER renaming so names match models.csv)
models_csv = pd.read_csv(DATA_DIR / "models.csv")

for var in [
    "provider",
    "open",
    "release_date",
    "size",
    "input_cost",
    "output_cost",
    "display_name",
]:
    accuracies_combined[var] = accuracies_combined["model"].apply(
        lambda x: models_csv[models_csv["model_id"] == x][var].values[0]
        if x in models_csv["model_id"].values
        else "Unknown"
    )

accuracies_vr = accuracies_combined.copy()
accuracies_by_model = accuracies_vr.set_index("model")

accuracies_combined.head()

# %%
# save csv for later use
accuracies_combined.to_csv(DATA_DIR / "accuracies_vr_metadata.csv", index=False)

# %%
# =============================================================================
# SECTION 1: ACCURACY ANALYSIS
# =============================================================================
# This section contains all accuracy-based visualizations:
#   - Plot 1A: Simple accuracy bar plot (all models, no CI)
#   - Plot 1B: Accuracy bar plot with bootstrap confidence intervals
#   - Plot 1C: Proprietary vs Open-source models comparison
#   - Plot 1D: Reasoning vs Non-reasoning models comparison

# %%
# Plot 1A: Simple accuracy bar plot (no confidence intervals)
print("\n" + "=" * 80)
print("PLOT 1A: Simple Accuracy Bar Plot")
print("=" * 80)

plot_benchmark_df(
    accuracies=accuracies_by_model,
    category="accuracy",
    save_fig=True,
    category_in_title=False,
    benchmark_name="Visual Reasoning Benchmark",
    folder_name="plots_gold",
    xticklabels_fontsize=18,
    width_figure=25,
    height_figure=9,
    position_legend=(1.10, 0.8),
    size_legend=18,
    y_limit=90,
)

# %%
# Plot 1B: Accuracy bar plot with bootstrap confidence intervals
print("\n" + "=" * 80)
print("PLOT 1B: Accuracy Bar Plot with Bootstrap Confidence Intervals")
print("=" * 80)

# Use eval_df that was already created earlier instead of loading CSV
df_scores_original = eval_df[["id", "model", "model_answer", "solution"]].copy()

# Apply the same model renaming as was done earlier
renamed_models_dict = {
    "gemini-2.0-flash-thinking-exp-1219": "gemini-2.0-flash-thinking-exp-01-21",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "claude-3-5-haiku-latest": "claude-3-5-haiku-20241022",
}
df_scores_original["model"] = df_scores_original["model"].replace(renamed_models_dict)

accuracies_bootstrap = accuracies_vr.copy()

# Make sure same models in both files
if not (
    df_scores_original["model"].unique().tolist()
    == accuracies_bootstrap["model"].unique().tolist()
):
    print("Warning: Models in df_scores and accuracies do not match!")
    print(
        f"   Models in df_scores but not in accuracies: {set(df_scores_original['model'].unique()) - set(accuracies_bootstrap['model'].unique())}"
    )
    print(
        f"   Models in accuracies but not in df_scores: {set(accuracies_bootstrap['model'].unique()) - set(df_scores_original['model'].unique())}"
    )

    remove_models = set(df_scores_original["model"].unique()) - set(
        accuracies_bootstrap["model"].unique()
    )
    df_scores_original = df_scores_original[
        df_scores_original["model"].isin(
            accuracies_bootstrap["model"].unique().tolist()
        )
    ].reset_index(drop=True)
    print(f"\nRemoved {len(remove_models)} models from df_scores to match accuracies.")
    print(
        f"After filtering, df_scores now has {df_scores_original['model'].nunique()} unique models."
    )

print(f"\nDataFrame shape: {df_scores_original.shape}")
print(f"Columns in DataFrame: {df_scores_original.columns.tolist()}")
print(f"Number of models in accuracies: {accuracies_bootstrap.shape[0]}")

# Reformat df_scores to have one column per model with predictions
df_scores = df_scores_original.pivot(
    index="id", columns="model", values="model_answer"
).reset_index()
df_scores.columns = ["id"] + [f"pred_{col}" for col in df_scores.columns if col != "id"]
df_scores["solution"] = (
    df_scores_original.drop_duplicates(subset=["id"])[["id", "solution"]]
    .set_index("id")
    .loc[df_scores["id"]]["solution"]
    .values
)
df_scores.insert(1, "solution", df_scores.pop("solution"))
print(f"Reformatted DataFrame shape: {df_scores.shape}")
print(
    f"Number of models: {len([col for col in df_scores.columns if col.startswith('pred_')])}"
)
print(f"Number of unique questions: {df_scores['id'].nunique()}")

correct_answer_column = "solution"
model_prediction_columns = [col for col in df_scores.columns if col.startswith("pred_")]
print(f"Number of models to analyze: {len(model_prediction_columns)}")

# Set the number of bootstrap replicates
num_replicates = 1000

# Run the bootstrap analysis
bootstrap_results_df, bootstrap_results_dict = bootstrap_model_accuracies(
    df_scores,
    correct_answer_column,
    model_prediction_columns,
    n_bootstraps=num_replicates,
    random_state=42,
)

print(f"Results DataFrame shape: {bootstrap_results_df.shape}")

# Add model metadata
for i, row in bootstrap_results_df.iterrows():
    model_id = row["model_id"]
    if model_id in models_csv["model_id"].values:
        model_info = models_csv[models_csv["model_id"] == model_id]
        bootstrap_results_df.at[i, "provider"] = model_info["provider"].values[0]
        bootstrap_results_df.at[i, "size"] = model_info["size"].values[0]
        bootstrap_results_df.at[i, "input_cost"] = model_info["input_cost"].values[0]
        bootstrap_results_df.at[i, "display_name"] = model_info["display_name"].values[
            0
        ]
    else:
        print(f"Model {model_id} not found in models_csv, skipping...")

providers_csv = pd.read_csv(DATA_DIR / "providers.csv")

# Creating color mapping
bootstrap_results_df["color"] = bootstrap_results_df["provider"].map(
    providers_csv.set_index("provider")["color"]
)

# Create a copy to avoid modifying the original DataFrame
df_plot_bootstrap = bootstrap_results_df.copy()
df_plot_bootstrap.sort_values(by="Original Accuracy", ascending=False, inplace=True)

# Convert accuracies and CIs to percentages
df_plot_bootstrap["Original Accuracy %"] = df_plot_bootstrap["Original Accuracy"] * 100
df_plot_bootstrap["ci_lower_%"] = df_plot_bootstrap["95% Confidence Interval"].apply(
    lambda x: x[0] * 100
    if isinstance(x, (tuple, list)) and len(x) == 2 and x[0] is not None
    else np.nan
)
df_plot_bootstrap["ci_upper_%"] = df_plot_bootstrap["95% Confidence Interval"].apply(
    lambda x: x[1] * 100
    if isinstance(x, (tuple, list)) and len(x) == 2 and x[1] is not None
    else np.nan
)

# Filter to models we want to keep (same as accuracies_bootstrap)
models_to_keep = {}
for provider in df_plot_bootstrap["provider"].unique():
    models_to_keep[provider] = df_plot_bootstrap[
        df_plot_bootstrap["provider"] == provider
    ]["model_id"].tolist()

models_to_keep_list = [model for models in models_to_keep.values() for model in models]
df_plot_bootstrap = df_plot_bootstrap[
    df_plot_bootstrap["model_id"].isin(models_to_keep_list)
].reset_index(drop=True)
accuracies_bootstrap = accuracies_bootstrap[
    accuracies_bootstrap["model"].isin(models_to_keep_list)
].reset_index(drop=True)
accuracies_bootstrap.set_index("model", inplace=True)

# remove from df_plot_bootstrap models that are not in accuracies_bootstrap
df_plot_bootstrap = df_plot_bootstrap[
    df_plot_bootstrap["model_id"].isin(accuracies_bootstrap.index)
].reset_index(drop=True)

# Check that same number of models in both dataframes
if df_plot_bootstrap.shape[0] != accuracies_bootstrap.shape[0]:
    print(
        f"Warning: Number of models in df_plot_bootstrap ({df_plot_bootstrap.shape[0]}) does not match accuracies ({accuracies_bootstrap.shape[0]})."
    )
else:
    print(
        f"Number of models in both dataframes matches: {df_plot_bootstrap.shape[0]} models."
    )

# Calculate error bar lengths for percentages
lower_errors_pct = np.maximum(
    0, df_plot_bootstrap["Original Accuracy %"] - df_plot_bootstrap["ci_lower_%"]
)
upper_errors_pct = np.maximum(
    0, df_plot_bootstrap["ci_upper_%"] - df_plot_bootstrap["Original Accuracy %"]
)
error_bars_pct = [lower_errors_pct.values, upper_errors_pct.values]

plot_benchmark_df(
    accuracies=accuracies_bootstrap,
    category="accuracy",
    save_fig=True,
    category_in_title=False,
    benchmark_name="Visual Reasoning Benchmark",
    folder_name="plots_gold",
    include_ci=True,
    error_bars_pct=error_bars_pct,
    add_accuracy_annotations=True,
    xticklabels_fontsize=20,
    filename_suffix="_bootstrap_CI95",
    width_figure=30,
    height_figure=9,
    position_legend=(1.06, 1.05),
    size_legend=16,
    y_limit=90,
)

# %%
# Plot 1C: Proprietary vs Open-source models
print("\n" + "=" * 80)
print("PLOT 1C: Proprietary vs Open-Source Models")
print("=" * 80)

# Create color mapping for open vs proprietary models
open_models = []
proprietary_models = []
for model in accuracies_vr["model"].to_list():
    model_info = models_csv[models_csv["model_id"] == model]
    if not model_info.empty:
        is_open = model_info["open"].values[0]
        if is_open:
            open_models.append(model)
        else:
            proprietary_models.append(model)

color_mapping_open = {
    "#67C667": open_models,
    "#C95555": proprietary_models,
}

plot_benchmark_df(
    accuracies=accuracies_by_model,
    category="accuracy",
    save_fig=True,
    category_in_title=False,
    benchmark_name="Visual Reasoning Benchmark",
    folder_name="plots_gold",
    add_accuracy_annotations=False,
    add_legend=True,
    xticklabels_fontsize=15,
    custom_legend_elements=[
        mlines.Line2D(
            [],
            [],
            color="#C95555",
            marker="o",
            linestyle="None",
            markersize=13,
            label="Proprietary models",
        ),
        mlines.Line2D(
            [],
            [],
            color="#67C667",
            marker="o",
            linestyle="None",
            markersize=13,
            label="Open Weights models",
        ),
    ],
    custom_color_mapping=color_mapping_open,
    filename_suffix="_open_models",
    width_figure=25,
    height_figure=9,
    position_legend=(0.95, 0.85),
    size_legend=25,
    y_limit=90,
)

# %%
# Plot 1D: Reasoning vs Non-reasoning models
print("\n" + "=" * 80)
print("PLOT 1D: Reasoning vs Non-Reasoning Models")
print("=" * 80)

reasoning_models_list = [
    "o1-preview-2024-09-12",
    "o1-mini-2024-09-12",
    "o3-mini-medium",
    "o1-2024-12-17",
    "o1-high",
    "o3-2025-04-16",
    "o4-mini-2025-04-16",
    "gpt-5-2025-08-07-medium",
    "gpt-5-mini-2025-08-07-medium",
    "gpt-5-2025-08-07-minimal",
    "gpt-5-mini-2025-08-07-minimal",
    "gpt-5-nano-2025-08-07-medium",
    "gpt-5-nano-2025-08-07-minimal",
    "claude-3-7-sonnet-20250219-thinking-low",
    "claude-3-7-sonnet-20250219-thinking-high",
    "claude-3-7-sonnet-20250219-thinking-medium",
    "claude-opus-4-20250514-low",
    "claude-sonnet-4-20250514-low",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.0-flash-thinking-exp-01-21",
    "gemini-2.5-pro-preview-05-06",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-pro-preview-06-05",
    "gemini-2.5-flash-lite-preview-06-17",
    "qwen-qvq-72b-preview",
    "qwen-qwq-32b",
    "fw-deepseek-r1",
    "fw-deepseek-r1-0528",
    "glm-4.1v-9b-thinking",
    "gpt-5.2-2025-12-11-medium",
]

color_mapping_reasoning = {
    "#08519c": reasoning_models_list,
    "#6baed6": [
        model
        for model in accuracies_vr["model"].to_list()
        if model not in reasoning_models_list
    ],
}

plot_benchmark_df(
    accuracies=accuracies_by_model,
    category="accuracy",
    save_fig=True,
    category_in_title=False,
    benchmark_name="Visual Reasoning Benchmark",
    folder_name="plots_gold",
    add_accuracy_annotations=False,
    add_legend=True,
    xticklabels_fontsize=15,
    custom_legend_elements=[
        mlines.Line2D(
            [],
            [],
            color="#08519c",
            marker="o",
            linestyle="None",
            markersize=13,
            label="Reasoning models",
        ),
        mlines.Line2D(
            [],
            [],
            color="#6baed6",
            marker="o",
            linestyle="None",
            markersize=13,
            label="Non-reasoning models",
        ),
    ],
    custom_color_mapping=color_mapping_reasoning,
    filename_suffix="_reasoning_models",
    width_figure=25,
    height_figure=9,
    position_legend=(0.95, 0.85),
    size_legend=25,
    y_limit=90,
)

# %%
# =============================================================================
# SECTION 2: SKILL VARIANCE ANALYSIS
# =============================================================================
# This section analyzes model performance variation across different skills.
# It computes per-skill accuracy for each model and visualizes the spread
# (min/max/average) across skills for top and bottom performers.

# Build a skills lookup per question
skill_catalogs = []
for dataset, path in [("zambia", ZAMBIA_PATH), ("india", INDIA_PATH)]:
    df_skill = pd.read_csv(path)
    df_skill = df_skill.rename(columns=str.strip)
    # Drop questions flagged with errors for cleaner skill analysis
    df_skill = df_skill[df_skill["error_severity_value"].isna()].copy()
    df_skill["skills_list"] = (
        df_skill["skills"]
        .fillna("")
        .apply(lambda x: [s.strip() for s in str(x).split(";") if s.strip()])
    )
    df_skill = df_skill.explode("skills_list")
    df_skill["skills_list"] = df_skill["skills_list"].str.strip()
    df_skill = df_skill[df_skill["skills_list"] != ""]
    df_skill = df_skill[~df_skill["skills_list"].str.lower().eq("other")]
    df_skill["dataset"] = dataset
    skill_catalogs.append(df_skill[["id", "dataset", "skills_list"]])

skills_map = pd.concat(skill_catalogs, ignore_index=True).rename(
    columns={"skills_list": "skill"}
)

# Align model naming with the rest of the script
renamed_models_dict = {
    "gemini-2.0-flash-thinking-exp-1219": "gemini-2.0-flash-thinking-exp-01-21",
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "claude-3-5-haiku-latest": "claude-3-5-haiku-20241022",
}

eval_skills = eval_df.copy()
eval_skills["model"] = eval_skills["model"].replace(renamed_models_dict)
skills_eval = eval_skills.merge(skills_map, on=["dataset", "id"], how="inner")

# Per-skill accuracy per model (percentage)
skill_accuracy = (
    skills_eval.groupby(["model", "skill"])["correct"].mean().mul(100).reset_index()
)
skill_pivot = skill_accuracy.pivot(index="skill", columns="model", values="correct")

# Prepare spread data (min/max/avg per model across skills)
avg_scores_series = accuracies_by_model["accuracy"]
if avg_scores_series.max() <= 1:
    avg_scores_series = avg_scores_series * 100

sorted_models_names = avg_scores_series.sort_values(ascending=False).index
plot_data_skill = []
for model_name in sorted_models_names:
    if model_name not in skill_pivot.columns:
        plot_data_skill.append(
            {
                "model": model_name,
                "avg": avg_scores_series.loc[model_name],
                "min_score": np.nan,
                "max_score": np.nan,
                "min_skill": "N/A",
                "max_skill": "N/A",
            }
        )
        continue
    model_skill_scores = skill_pivot[model_name].dropna()
    if model_skill_scores.empty:
        min_s, max_s, min_c, max_c = np.nan, np.nan, "N/A", "N/A"
    else:
        min_s = model_skill_scores.min()
        max_s = model_skill_scores.max()
        min_c = model_skill_scores.idxmin()
        max_c = model_skill_scores.idxmax()

    plot_data_skill.append(
        {
            "model": model_name,
            "avg": avg_scores_series.loc[model_name],
            "min_score": min_s,
            "max_score": max_s,
            "min_skill": min_c,
            "max_skill": max_c,
        }
    )

processed_skill_df = pd.DataFrame(plot_data_skill)

top_n_skills = 15
processed_skill_df = pd.concat(
    [processed_skill_df.head(top_n_skills), processed_skill_df.tail(top_n_skills)]
).reset_index(drop=True)
processed_skill_df = processed_skill_df.sort_values(
    by="avg", ascending=False
).reset_index(drop=True)

models_csv = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "models.csv")
processed_skill_df["display_name"] = processed_skill_df["model"].apply(
    lambda x: models_csv[models_csv["model_id"] == x]["display_name"].values[0]
    if x in models_csv["model_id"].values
    else x
)

# Colors per skill
all_skills_names = skill_pivot.index.tolist()
if len(all_skills_names) <= 10:
    cmap_skills = plt.cm.get_cmap("tab10", len(all_skills_names))
elif len(all_skills_names) <= 20:
    cmap_skills = plt.cm.get_cmap("tab20", len(all_skills_names))
else:
    cmap_skills = plt.cm.get_cmap("viridis", len(all_skills_names))

skill_color_map = {skill: cmap_skills(i) for i, skill in enumerate(all_skills_names)}
skill_color_map["N/A"] = "lightgrey"

# Plot spread across skills (similar to category variance)
num_models_skill = len(processed_skill_df)
gap_size = 2.5
top_y = np.arange(top_n_skills)
bottom_y = np.arange(top_n_skills) + top_n_skills + gap_size
y_positions_skill = np.concatenate([top_y, bottom_y])
bottom_group_start_index = top_n_skills

fig_height_skill = max(7, num_models_skill * 0.28)
fig, ax = plt.subplots(figsize=(12, fig_height_skill))

gap_center_y = top_n_skills - 0.5 + gap_size / 2
ax.text(
    0.05,
    gap_center_y,
    "· · ·",
    ha="center",
    va="center",
    fontsize=20,
    color="gray",
    transform=ax.get_yaxis_transform(),
)

ax.axhspan(
    y_positions_skill[bottom_group_start_index] - 0.5,
    y_positions_skill[-1] + 0.5,
    facecolor="whitesmoke",
    alpha=1.0,
    zorder=0,
)

ax.text(
    0.04,
    (top_n_skills - 1) / 2.0,
    f"Top {top_n_skills} Models",
    ha="right",
    va="center",
    rotation=90,
    fontsize=13,
    fontweight="bold",
    color="gray",
    transform=ax.get_yaxis_transform(),
)

ax.text(
    0.04,
    np.mean(bottom_y),
    f"Bottom {top_n_skills} Models",
    ha="right",
    va="center",
    rotation=90,
    fontsize=13,
    fontweight="bold",
    color="dimgray",
    transform=ax.get_yaxis_transform(),
)

for i, row in processed_skill_df.iterrows():
    y_pos = y_positions_skill[i]
    if pd.notna(row["min_score"]) and pd.notna(row["max_score"]):
        ax.plot(
            [row["min_score"], row["max_score"]],
            [y_pos, y_pos],
            color="silver",
            linewidth=3,
            zorder=1,
            solid_capstyle="round",
        )
        ax.scatter(
            row["min_score"],
            y_pos,
            color=skill_color_map[row["min_skill"]],
            s=100,
            zorder=2,
            edgecolors="grey",
            linewidth=0.5,
        )
        ax.scatter(
            row["max_score"],
            y_pos,
            color=skill_color_map[row["max_skill"]],
            s=300,
            marker="*",
            zorder=3,
            edgecolors="black",
            linewidth=0.7,
        )
    if pd.notna(row["avg"]):
        ax.scatter(
            row["avg"],
            y_pos,
            color="black",
            marker="D",
            s=50,
            zorder=4,
            edgecolors="white",
            linewidth=0.5,
        )

ax.set_yticks(y_positions_skill)
ax.set_yticklabels(processed_skill_df["display_name"], fontsize=13)
ax.invert_yaxis()

ax.set_xlabel("Accuracy (%)", font=FONT_BOLD, fontsize=16)
ax.tick_params(axis="x", labelsize=16)
ax.set_title(
    f"Visual Reasoning Performance Spread Across Skills\nTop & Bottom {top_n_skills} Models",
    font=FONT_BOLD,
    fontsize=20,
)
ax.grid(axis="x", linestyle=":", color="gray", alpha=0.7)
ax.tick_params(axis="y", length=0)

legend_elements_skills = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=skill,
        markerfacecolor=col,
        markersize=10,
        markeredgecolor="grey",
    )
    for skill, col in skill_color_map.items()
    if skill != "N/A"
]
legend_elements_skills.append(Line2D([], [], color="none", label=""))
legend_elements_skills.append(Line2D([], [], color="none", label="Markers"))
legend_elements_skills.append(
    Line2D(
        [0],
        [0],
        marker="*",
        color="w",
        label="Best Skill",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=15,
    )
)
legend_elements_skills.append(
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Worst Skill",
        markerfacecolor="white",
        markersize=10,
        markeredgecolor="black",
    )
)
legend_elements_skills.append(
    Line2D(
        [0],
        [0],
        marker="D",
        color="w",
        label="Overall Score",
        markerfacecolor="black",
        markersize=8,
        markeredgecolor="white",
    )
)
legend_elements_skills.append(
    Line2D([0], [0], color="silver", lw=4, label="Performance Range")
)

legend = ax.legend(
    handles=legend_elements_skills,
    title="Skill",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0.0,
)
legend.get_title().set_fontweight("bold")
for text in legend.get_texts():
    if text.get_text() == "Markers":
        text.set_weight("bold")

plt.tight_layout()

# %%
# Plot top 3 and bottom 3 models by skill in a 2x3 grid

# Create long-form data for skills
df_skill_long = skill_pivot.reset_index().melt(
    id_vars="skill",
    var_name="model",
    value_name="accuracy",
)

# Dynamically select top 3 and bottom 3 models based on average accuracy
top_3_models_skill = processed_skill_df.head(3)["model"].tolist()
bottom_3_models_skill = processed_skill_df.tail(3)["model"].tolist()
models_plot_skill = top_3_models_skill + bottom_3_models_skill

# Append display names to the models
df_skill_long["display_name"] = df_skill_long["model"].apply(
    lambda x: models_csv[models_csv["model_id"] == x]["display_name"].values[0]
    if x in models_csv["model_id"].values
    else x
)

fig, axs = plt.subplots(2, 3, figsize=(22, 12), sharey=True, sharex=True)

for model in models_plot_skill:
    model_data = df_skill_long[df_skill_long["model"] == model]
    if model_data.empty:
        print(f"No data for model {model}, skipping...")
        continue
    row_index = models_plot_skill.index(model) // 3
    col_index = models_plot_skill.index(model) % 3
    ax = axs[row_index, col_index]
    sns.barplot(
        x="skill",
        y="accuracy",
        hue="skill",
        data=model_data,
        palette=[cmap_skills(i) for i in range(len(all_skills_names))],
        ax=ax,
    )
    ax.set_title(model_data["display_name"].values[0], font=FONT_BOLD, fontsize=23)
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy (%)", font=FONT_BOLD, fontsize=20)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=45, labelsize=15)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.tick_params(axis="y", labelsize=15)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.axhline(y=100, color="white", linewidth=2)

legend_elements_skill = [
    Line2D(
        [0], [0], marker="s", color="w", label=skill, markerfacecolor=col, markersize=12
    )
    for skill, col in skill_color_map.items()
    if skill != "N/A"
]
legend = ax.legend(
    handles=legend_elements_skill,
    title="Skill",
    bbox_to_anchor=(1.6, 1.3),
    loc="upper center",
    fontsize=14,
    title_fontsize=16,
)
legend.get_title().set_fontweight("bold")

plt.tight_layout()

# %%
# =============================================================================
# SECTION 3: CATEGORY VARIANCE ANALYSIS
# =============================================================================
# This section analyzes model performance variation across different categories.
# It includes:
#   - Category variance plot (min/max/average spread across categories)
#   - Top/bottom models breakdown by category (2x3 grid)
#   - Bayesian category difficulty analysis (overall and by dataset)

# Compute category-level results from eval_df
eval_df_filtered = eval_df[~eval_df["model"].isin(EXCLUDE_MODELS)].copy()

category_df = (
    eval_df_filtered.groupby(["category", "model"])["correct"].mean().unstack("model")
) * 100

# Align model naming
category_df.columns = category_df.columns.str.strip()
category_df = category_df.rename(columns=renamed_models_dict)

# %%
# Plot category variance

# Separate average scores and category scores
avg_scores_series = accuracies_by_model["accuracy"]

# Get sorted model names based on average score (descending)
sorted_models_names = avg_scores_series.sort_values(ascending=False).index

# Prepare data for plotting
plot_data_list = []
for model_name in sorted_models_names:
    model_avg = avg_scores_series.loc[model_name]
    if model_name not in category_df.columns:
        plot_data_list.append(
            {
                "model": model_name,
                "avg": model_avg,
                "min_score": np.nan,
                "max_score": np.nan,
                "min_category": "N/A",
                "max_category": "N/A",
            }
        )
        continue
    model_category_scores = category_df[model_name].dropna()

    if model_category_scores.empty:
        min_s, max_s, min_c, max_c = np.nan, np.nan, "N/A", "N/A"
    else:
        min_s = model_category_scores.min()
        max_s = model_category_scores.max()
        min_c = model_category_scores.idxmin()
        max_c = model_category_scores.idxmax()

    plot_data_list.append(
        {
            "model": model_name,
            "avg": model_avg,
            "min_score": min_s,
            "max_score": max_s,
            "min_category": min_c,
            "max_category": max_c,
        }
    )

processed_df = pd.DataFrame(plot_data_list)

processed_df["avg"] = (
    processed_df["avg"] * 100 if any(processed_df["avg"] <= 1) else processed_df["avg"]
)
processed_df["min_score"] = (
    processed_df["min_score"] * 100
    if any(processed_df["min_score"] <= 1)
    else processed_df["min_score"]
)
processed_df["max_score"] = (
    processed_df["max_score"] * 100
    if any(processed_df["max_score"] <= 1)
    else processed_df["max_score"]
)

models_csv = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "models.csv")

top_n = 15
processed_df = pd.concat(
    [processed_df.head(top_n), processed_df.tail(top_n)]
).reset_index(drop=True)
processed_df = processed_df.sort_values(by="avg", ascending=False).reset_index(
    drop=True
)

processed_df["display_name"] = processed_df["model"].apply(
    lambda x: models_csv[models_csv["model_id"] == x]["display_name"].values[0]
    if x in models_csv["model_id"].values
    else x
)
# rename some models for better readability
processed_df["display_name"] = processed_df["display_name"].replace(
    {
        "Claude Opus 4 Thinking (low)": "Claude Opus 4 Thinking",
        "Claude Sonnet 4 Thinking (low)": "Claude Sonnet 4 Thinking",
        "Deepseek R1 (May '25)": "DeepSeek R1",
        "Claude-3.7 Sonnet Thinking (medium)": "Claude-3.7 Sonnet Thinking",
        "o1-Medium": "o1",
    }
)

# Color Mapping for Categories
all_categories_names = category_df.index.tolist()
if len(all_categories_names) <= 10:
    cmap = plt.cm.get_cmap("tab10", len(all_categories_names))
elif len(all_categories_names) <= 20:
    cmap = plt.cm.get_cmap("tab20", len(all_categories_names))
else:
    cmap = plt.cm.get_cmap("viridis", len(all_categories_names))

category_color_map = {
    category: cmap(i) for i, category in enumerate(all_categories_names)
}
category_color_map["N/A"] = "lightgrey"

# Plotting
num_models = len(processed_df)
bottom_group_start_index = top_n

fig_height = max(7, num_models * 0.28)
fig, ax = plt.subplots(figsize=(12, fig_height))

gap_size = 2.5
top_y = np.arange(top_n)
bottom_y = np.arange(top_n) + top_n + gap_size

y_positions = np.concatenate([top_y, bottom_y])

gap_center_y = top_n - 0.5 + gap_size / 2
ax.text(
    0.05,
    gap_center_y,
    "· · ·",
    ha="center",
    va="center",
    fontsize=20,
    color="gray",
    transform=ax.get_yaxis_transform(),
)

ax.axhspan(
    y_positions[bottom_group_start_index] - 0.5,
    y_positions[-1] + 0.5,
    facecolor="whitesmoke",
    alpha=1.0,
    zorder=0,
)

ax.text(
    0.04,
    (top_n - 1) / 2.0,
    f"Top {top_n} Models",
    ha="right",
    va="center",
    rotation=90,
    fontsize=13,
    fontweight="bold",
    color="gray",
    transform=ax.get_yaxis_transform(),
)

ax.text(
    0.04,
    np.mean(bottom_y),
    f"Bottom {top_n} Models",
    ha="right",
    va="center",
    rotation=90,
    fontsize=13,
    fontweight="bold",
    color="dimgray",
    transform=ax.get_yaxis_transform(),
)

for i, row in processed_df.iterrows():
    y_pos = y_positions[i]

    if pd.notna(row["min_score"]) and pd.notna(row["max_score"]):
        ax.plot(
            [row["min_score"], row["max_score"]],
            [y_pos, y_pos],
            color="silver",
            linewidth=3,
            zorder=1,
            solid_capstyle="round",
        )

        ax.scatter(
            row["min_score"],
            y_pos,
            color=category_color_map[row["min_category"]],
            s=100,
            zorder=2,
            edgecolors="grey",
            linewidth=0.5,
        )

        ax.scatter(
            row["max_score"],
            y_pos,
            color=category_color_map[row["max_category"]],
            s=300,
            marker="*",
            zorder=3,
            edgecolors="black",
            linewidth=0.7,
        )

    if pd.notna(row["avg"]):
        ax.scatter(
            row["avg"],
            y_pos,
            color="black",
            marker="D",
            s=50,
            zorder=4,
            edgecolors="white",
            linewidth=0.5,
        )

ax.set_yticks(y_positions)
ax.set_yticklabels(processed_df["display_name"], fontsize=13)
ax.invert_yaxis()

ax.set_xlabel("Accuracy (%)", font=FONT_BOLD, fontsize=16)
ax.tick_params(axis="x", labelsize=16)
ax.set_title(
    f"Visual Reasoning Performance Spread Across Tasks\nTop & Bottom {top_n} Models",
    font=FONT_BOLD,
    fontsize=20,
)
ax.grid(axis="x", linestyle=":", color="gray", alpha=0.7)
ax.tick_params(axis="y", length=0)

legend_elements = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label=cat,
        markerfacecolor=col,
        markersize=10,
        markeredgecolor="grey",
    )
    for cat, col in category_color_map.items()
    if cat != "N/A"
]
legend_elements.append(Line2D([], [], color="none", label=""))
legend_elements.append(Line2D([], [], color="none", label="Markers"))

legend_elements.append(
    Line2D(
        [0],
        [0],
        marker="*",
        color="w",
        label="Best Category",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=15,
    )
)
legend_elements.append(
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Worst Category",
        markerfacecolor="white",
        markersize=10,
        markeredgecolor="black",
    )
)
legend_elements.append(
    Line2D(
        [0],
        [0],
        marker="D",
        color="w",
        label="Overall Score",
        markerfacecolor="black",
        markersize=8,
        markeredgecolor="white",
    )
)
legend_elements.append(
    Line2D([0], [0], color="silver", lw=4, label="Performance Range")
)

legend = ax.legend(
    handles=legend_elements,
    title="Task",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    borderaxespad=0.0,
)
legend.get_title().set_fontweight("bold")
for text in legend.get_texts():
    if text.get_text() == "Markers":
        text.set_weight("bold")

plt.tight_layout()

# %%
# Plot spread for top 3 and bottom 3 in bar plot in a 2x3 grid

df_long = category_df.reset_index().melt(
    id_vars="category",
    var_name="model",
    value_name="accuracy",
)

df_long["accuracy"] = (
    df_long["accuracy"] * 100 if all(df_long["accuracy"] <= 1) else df_long["accuracy"]
)

# Dynamically select top 3 and bottom 3 models based on average accuracy
top_3_models = processed_df.head(3)["model"].tolist()
bottom_3_models = processed_df.tail(3)["model"].tolist()
models_plot = top_3_models + bottom_3_models

# append display names to the models_plot
df_long["display_name"] = df_long["model"].apply(
    lambda x: models_csv[models_csv["model_id"] == x]["display_name"].values[0]
    if x in models_csv["model_id"].values
    else x
)

fig, axs = plt.subplots(2, 3, figsize=(22, 12), sharey=True, sharex=True)
cmap = plt.cm.get_cmap("tab10", len(all_categories_names))

for model in models_plot:
    model_data = df_long[df_long["model"] == model]
    if model_data.empty:
        print(f"No data for model {model}, skipping...")
        continue
    row_index = models_plot.index(model) // 3
    col_index = models_plot.index(model) % 3
    ax = axs[row_index, col_index]
    sns.barplot(
        x="category",
        y="accuracy",
        hue="category",
        data=model_data,
        palette=[cmap(i) for i in range(len(all_categories_names))],
        ax=ax,
    )
    ax.set_title(model_data["display_name"].values[0], font=FONT_BOLD, fontsize=23)
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy (%)", font=FONT_BOLD, fontsize=20)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=45, labelsize=15)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.tick_params(axis="y", labelsize=15)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.axhline(y=100, color="white", linewidth=2)

legend_elements = [
    Line2D(
        [0], [0], marker="s", color="w", label=cat, markerfacecolor=col, markersize=12
    )
    for cat, col in category_color_map.items()
    if cat != "N/A"
]
legend = ax.legend(
    handles=legend_elements,
    title="Task",
    bbox_to_anchor=(1.6, 1.3),
    loc="upper center",
    fontsize=14,
    title_fontsize=16,
)
legend.get_title().set_fontweight("bold")

plt.tight_layout()

# %%
# =============================================================================
# SECTION 4: FRONTIER ANALYSIS & PLOTS
# =============================================================================

# %%
# frontier over time

# Use the already-computed accuracies_vr instead of reading from CSV
df = accuracies_vr.copy()
df = df.rename(columns={"model": "model_id"})
df["accuracy"] = df["accuracy"] * 100
if "url" not in df.columns:
    df["url"] = df["model_id"].apply(
        lambda x: models_csv[models_csv["model_id"] == x]["url"].values[0]
        if x in models_csv["model_id"].values and "url" in models_csv.columns
        else ""
    )

# Parse release date and enforce numeric types
df["release_date"] = pd.to_datetime(df["release_date"], format="%m/%Y", errors="coerce")
df["input_cost"] = pd.to_numeric(df["input_cost"], errors="coerce")
df["output_cost"] = pd.to_numeric(df["output_cost"], errors="coerce")
df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce")


def compute_frontier(data, acc_col="accuracy", sorting_col="input_cost", smooth=False):
    data = data.sort_values(sorting_col, ascending=True)
    frontier, best = [], -1e9
    for _, row in data.iterrows():
        if row[acc_col] > best:
            frontier.append(row)
            best = row[acc_col]
    df_return = pd.DataFrame(frontier)
    df_return = df_return.sort_values(sorting_col).drop_duplicates(
        sorting_col, keep="last"
    )

    if smooth:
        df_return["angle"] = np.arctan2(
            df_return["accuracy"].diff().fillna(0),
            df_return[sorting_col].diff().fillna(1),
        )
        df_return["angle"] = np.degrees(df_return["angle"])

    df_return = df_return.reset_index(drop=True)
    return df_return


def plot_frontier_on_ax(
    ax, df, frontier_var, target_dates, colors, smooth=False, title=None
):
    """
    Plots the efficiency/value frontier on a given Matplotlib Axes object.

    Args:
        ax (matplotlib.axes.Axes): The subplot to draw on.
        df (pd.DataFrame): The dataframe containing all model data.
        frontier_var (str): The column name for the x-axis (e.g., 'size' or 'input_cost').
        target_dates (dict): Dictionary of labels and date strings for frontiers.
        colors (list): A list of colors for the frontier lines.
        title (str, optional): Custom title for the plot. If None, uses default title.
    """
    ax.scatter(
        df[frontier_var],
        df["accuracy"],
        color="grey",
        alpha=0.6,
        s=50,
        label="All Models",
    )

    if not target_dates:
        frontier = compute_frontier(df, sorting_col=frontier_var, smooth=smooth)
        if not frontier.empty:
            ax.plot(
                frontier[frontier_var],
                frontier["accuracy"],
                linewidth=2,
                label="Value Frontier",
                color="grey",
            )
            ax.scatter(
                frontier[frontier_var],
                frontier["accuracy"],
                edgecolors="white",
                color="grey",
                s=120,
                zorder=5,
            )
    else:
        for (label, date_str), color in zip(target_dates.items(), colors):
            subset = df[df["release_date"] <= pd.to_datetime(date_str)]
            if subset.empty:
                continue

            frontier = compute_frontier(subset, sorting_col=frontier_var, smooth=smooth)
            if frontier.empty:
                continue

            ax.plot(
                frontier[frontier_var],
                frontier["accuracy"],
                linewidth=2,
                label=f"In {label}",
                color=color,
            )
            ax.scatter(
                frontier[frontier_var],
                frontier["accuracy"],
                edgecolors="white",
                color=color,
                s=120,
                zorder=5,
            )

    ax.set_xscale("log")
    y_min, y_max = 20, 90
    ax.set_ylim(y_min, y_max)

    if "cost" in frontier_var:
        ax.set_xticks([0.01, 0.1, 1, 10, 100])
        ax.set_xticklabels(
            ["0.01", "0.1", "1", "10", "100"], font=FONT_REGULAR, fontsize=13
        )
        ax.set_xlabel("Cost ($/1M tokens)", font=FONT_BOLD, fontsize=18)
        ax.set_title(title if title else "Value Frontier", font=FONT_BOLD, fontsize=18)
        ax.legend(
            title="Value Frontier",
            loc="upper right",
            fontsize=13,
            title_fontsize=14,
            frameon=True,
        )
    elif frontier_var == "size":
        ax.set_xticks([1e9, 1e10, 1e11, 1e12])
        ax.set_xticklabels(["1B", "10B", "100B", "1T"], font=FONT_REGULAR, fontsize=13)
        ax.set_xlabel("Model size (# param.)", font=FONT_BOLD, fontsize=18)
        ax.set_title(
            title if title else "Efficiency Frontier over time",
            font=FONT_BOLD,
            fontsize=18,
        )
        ax.legend(
            title="Efficiency Frontier",
            loc="upper right",
            fontsize=13,
            title_fontsize=14,
            frameon=True,
        )

    ax.set_yticks(np.arange(y_min, y_max + 1, 10))
    ax.set_yticklabels(np.arange(y_min, y_max + 1, 10), font=FONT_REGULAR, fontsize=13)
    ax.set_ylabel("Accuracy (%)", font=FONT_BOLD, fontsize=18)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


df_frontier = compute_frontier(df, sorting_col="input_cost", smooth=True)
print(f"Frontier has {df_frontier.shape[0]} models.")
df_frontier.head()

# %%
# select type of frontier
frontier_var = "input_cost"

# Define target dates and colors
target_dates = {
    "November 2024": "2024-11-30",
    "April 2025": "2025-04-30",
    "December 2025": "2025-12-31",
}

colors = ["firebrick", "goldenrod", "steelblue"]

fig, ax = plt.subplots(figsize=(9, 6))

plot_frontier_on_ax(
    ax=ax,
    df=df,
    frontier_var=frontier_var,
    target_dates=target_dates,
    colors=colors,
    smooth=True,
)
plt.tight_layout()

# %%
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Frontier overall
plot_frontier_on_ax(
    ax=axs[0],
    df=df,
    frontier_var="input_cost",
    target_dates=None,
    colors=colors,
    title="Value Frontier",
)

# Plot 2: Frontier with time
plot_frontier_on_ax(
    ax=axs[1],
    df=df,
    frontier_var="input_cost",
    target_dates=target_dates,
    colors=colors,
    title="Value Frontier over time",
)

plt.tight_layout()

# %%
# Experiment with ratios
cost_ratios = {
    "Exp 1": (1.0, 0.0),
    "Exp 2": (0.5, 0.5),
    "Exp 3": (0.8, 0.2),
}

fig, axs = plt.subplots(1, len(cost_ratios), figsize=(25, 7))

for ax, (exp, (w1, w2)) in zip(axs, cost_ratios.items()):
    print(f"Running {exp} with weights: {w1}, {w2}")

    df["composite_cost"] = w1 * df["input_cost"] + w2 * df["output_cost"]

    plot_frontier_on_ax(
        ax=ax,
        df=df,
        frontier_var="composite_cost",
        target_dates=target_dates,
        colors=colors,
        smooth=True,
    )
    ax.set_title(f"I/O costs: {w1} / {w2}", font=FONT_BOLD, fontsize=16)
plt.tight_layout()

# %%
