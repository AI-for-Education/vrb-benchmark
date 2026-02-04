# %%
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis_utils import tokenize_skills
from matplotlib.patches import Patch

from vrb_benchmark import DATA_DIR, PROJECT_ROOT

# Configuration
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ZAMBIA_PATH = DATA_DIR / "zambia-questions-categories-gold.csv"
INDIA_PATH = DATA_DIR / "india-questions-categories-gold.csv"

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
}

print(f"Excluding {len(EXCLUDE_MODELS)} models")

# %%
# DATA LOADING AND PREPARATION


def load_catalog():
    """Load question catalog with categories and skills"""
    catalogs = []
    for dataset, path in [("zambia", ZAMBIA_PATH), ("india", INDIA_PATH)]:
        df = pd.read_csv(path)
        df = df.rename(columns=str.strip)
        df = df[
            [
                "id",
                "solution",
                "category",
                "skills",
                "error_type_value",
                "error_severity_value",
            ]
        ]
        df["error_type_value"] = df["error_type_value"].fillna("none").str.strip()
        df["error_severity_value"] = (
            df["error_severity_value"].fillna("none").str.strip()
        )
        df["error_flag"] = (df["error_severity_value"] != "none").astype(int)
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


def prepare_skill_indicators(catalog):
    """Create binary indicators for each skill (for multi-membership)"""
    all_skills = set()
    for skills_str in catalog["skills"].dropna():
        all_skills.update(tokenize_skills(skills_str))

    return catalog, sorted(all_skills)


# Reload with correct parsing
catalog = load_catalog()
catalog, all_skills = prepare_skill_indicators(catalog)

print(f"Total INDIVIDUAL unique skills: {len(all_skills)}")
print("\nAll individual skills:")
for i, skill in enumerate(sorted(all_skills), 1):
    print(f"{i}. {skill}")

# Load data
combined = load_model_responses()

# Merge and score
eval_df = combined.merge(catalog, on=["dataset", "id"], how="inner")
eval_df["correct"] = (eval_df["model_answer"] == eval_df["solution"]).astype(int)

print(f"Loaded {len(eval_df)} responses")
print(f"Models: {eval_df['model'].nunique()}")
print(f"Questions: {eval_df['id'].nunique()}")
print(f"Categories: {eval_df['category'].nunique()}")
print(f"Skills: {len(all_skills)}")

# %%
# Count categories in the catalog
category_counts = catalog["category"].value_counts().sort_values()

# Plot
plt.figure(figsize=(10, 6))
bars = plt.barh(
    range(len(category_counts)),
    category_counts.values,
    color="steelblue",
    alpha=0.8,
)

plt.yticks(range(len(category_counts)), category_counts.index)
plt.xlabel("Number of Questions")
plt.ylabel("Category")
plt.title("Distribution of Questions by Category (Combined Dataset)")

for i, v in enumerate(category_counts.values):
    plt.text(
        v + max(category_counts.values) * 0.01,
        i,
        str(v),
        va="center",
        fontsize=10,
    )

plt.tight_layout()
plt.show()

print(f"Total questions: {len(catalog)}")
print(f"Number of categories: {len(category_counts)}")
print("\nCategory breakdown:")
print(category_counts.sort_values(ascending=False))

print("\n" + "=" * 50)
print("Category distribution by dataset:")
print("=" * 50)
dataset_category = catalog.groupby(["dataset", "category"]).size().unstack(fill_value=0)
print(dataset_category)

# %%
# AGGREGATION FOR CATEGORY-LEVEL ANALYSIS

agg_category = eval_df.groupby(["model", "dataset", "category"], as_index=False).agg(
    y_all=("correct", "size"),
    y_correct=("correct", "sum"),
)
agg_category["prop"] = agg_category["y_correct"] / agg_category["y_all"]

# %%
# AGGREGATION FOR SKILL-LEVEL ANALYSIS

# Need to expand data for multi-membership: one row per question-skill pair
skill_rows = []
for _, row in eval_df.iterrows():
    skills = tokenize_skills(row["skills"])
    if not skills:
        continue
    n_skills = len(skills)
    for skill in skills:
        skill_row = row.copy()
        skill_row["skill"] = skill
        skill_row["skill_weight"] = 1.0 / n_skills
        skill_rows.append(skill_row)

eval_skills = pd.DataFrame(skill_rows)
if eval_skills.empty:
    raise ValueError(
        "No skill annotations found after expansion, check the skills column format."
    )

print(f"Expanded to {len(eval_skills)} skill-level rows")
print(f"Unique skills: {eval_skills['skill'].nunique()}")

# Create weighted_success column (needed for Q1b and Q2 aggregations)
eval_skills["weighted_success"] = eval_skills["correct"] * eval_skills["skill_weight"]
print(f"Created weighted_success column. Columns: {list(eval_skills.columns)}")

# %%


def plot_estimates_with_hdi(
    df,
    figsize=(10, 6),
    point_size=100,
    error_bar_width=0.3,
    capsize=5,
    plot_column="category",
    color="steelblue",
    alpha=0.8,
):
    """
    Create a point plot with highest density intervals.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns [plot_column, 'estimate', 'lower_3.0%', 'upper_97.0%']
    figsize : tuple
        Figure size (width, height)
    point_size : int
        Size of the point markers
    error_bar_width : float
        Width of error bar lines
    capsize : int
        Size of the error bar caps
    color : str
        Color for points and error bars
    alpha : float
        Transparency level for points
    """
    df_sorted = df.sort_values("estimate", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=figsize)

    lower_errors = df_sorted["estimate"] - df_sorted["lower_3.0%"]
    upper_errors = df_sorted["upper_97.0%"] - df_sorted["estimate"]

    y_positions = np.arange(len(df_sorted))

    ax.errorbar(
        df_sorted["estimate"],
        y_positions,
        xerr=[lower_errors, upper_errors],
        fmt="none",
        ecolor=color,
        elinewidth=error_bar_width,
        capsize=capsize,
        alpha=alpha * 0.8,
        capthick=2,
    )

    ax.scatter(
        df_sorted["estimate"],
        y_positions,
        s=point_size,
        color=color,
        alpha=alpha,
        zorder=5,
        edgecolors="white",
        linewidth=0.5,
    )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(df_sorted[plot_column])

    ax.set_xlabel("Estimate", fontsize=12)
    ax.set_ylabel(plot_column, fontsize=12)
    ax.set_title("Estimates with 94% Highest Density Intervals", fontsize=14, pad=20)

    ax.grid(True, axis="x", alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    if df_sorted["lower_3.0%"].min() < 0 < df_sorted["upper_97.0%"].max():
        ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)

    plt.tight_layout()

    return fig, ax


# %%
# QUESTION 1: INHERENT CATEGORY DIFFICULTY
# which categories are inherently most difficult?
print("\n" + "=" * 70)
print("Q1: Which categories are inherently most difficult?")
print("=" * 70)

model_q1 = bmb.Model(
    "p(y_correct, y_all) ~ 0 + category + (1|dataset) + (1|model)",
    agg_category,
    family="binomial",
)
fit_q1 = model_q1.fit(tune=2000, draws=5000, target_accept=0.95, cores=4)

# Conditional predictions (equal weighting)
pred_q1_conditional = bmb.interpret.predictions(
    model_q1,
    fit_q1,
    conditional=["category", "dataset", "model"],
    average_by=["category"],
)

# Empirical predictions (weighted by actual distribution)
pred_q1_empirical = bmb.interpret.predictions(
    model_q1,
    fit_q1,
    average_by=["category"],
)

print("\nCategory difficulty (conditional - equal weighting):")
print(pred_q1_conditional.sort_values("estimate", ascending=False))

print("\nCategory difficulty (empirical - actual distribution):")
print(pred_q1_empirical.sort_values("estimate", ascending=False))

# %%
# Plot conditional
fig, ax = plot_estimates_with_hdi(
    pred_q1_conditional,
    alpha=1.0,
    error_bar_width=2,
    plot_column="category",
)
ax.set_xlabel("Accuracy")
ax.set_ylabel("Category")
ax.set_title("Accuracy per category (avg. across models and datasets")
plt.tight_layout()
plt.savefig(
    OUTPUTS_DIR / "q1_category_difficulty_conditional.png", dpi=300, bbox_inches="tight"
)
plt.show()

# Plot empirical
fig, ax = plot_estimates_with_hdi(
    pred_q1_empirical,
    alpha=1.0,
    error_bar_width=2,
    plot_column="category",
)
ax.set_xlabel("Accuracy")
ax.set_ylabel("Category")
ax.set_title("Category Difficulty (Empirical)\nWeighted by actual dataset distribution")
plt.tight_layout()
plt.savefig(
    OUTPUTS_DIR / "q1_category_difficulty_empirical.png", dpi=300, bbox_inches="tight"
)
plt.show()

# %%
# DATASET-SPECIFIC PREDICTIONS (by-dataset visualizations)
print("\n" + "=" * 70)
print("CREATING DATASET-SPECIFIC PREDICTIONS")
print("=" * 70)

# Q1 predictions - by dataset (need this for visualization)
pred_q1_by_dataset = bmb.interpret.predictions(
    model_q1,
    fit_q1,
    conditional=["category", "dataset", "model"],
    average_by=["category", "dataset"],
)

# Keep only dataset/category combinations that actually exist in the evaluation data.
valid_pairs = (
    agg_category[["dataset", "category"]].drop_duplicates().reset_index(drop=True)
)
pred_q1_by_dataset = pred_q1_by_dataset.merge(
    valid_pairs, on=["dataset", "category"], how="inner"
)

print("\nCategory difficulty by dataset:")
print(pred_q1_by_dataset.sort_values(["dataset", "estimate"], ascending=[True, False]))

# Save this prediction
pred_q1_by_dataset.to_csv(
    DATA_DIR / "bayesian_category_difficulty_by_dataset.csv", index=False
)
print(f"saved: {DATA_DIR / 'bayesian_category_difficulty_by_dataset.csv'}")

# %%
# QUESTION 1B: INDIVIDUAL SKILL DIFFICULTY (RAW)
print("\n" + "=" * 70)
print("Q1b: Which skills are most difficult?")
print("=" * 70)

# Aggregate WITHOUT category (parallel to Q1)
agg_skills = eval_skills.groupby(
    ["model", "dataset", "skill"],
    as_index=False,
).agg(
    y_all=("skill_weight", "sum"),
    y_correct=("weighted_success", "sum"),
)

# Scale to integers for binomial likelihood
agg_skills["y_all"] = np.round(agg_skills["y_all"] * 100).astype(int)
agg_skills["y_correct"] = np.round(agg_skills["y_correct"] * 100).astype(int)
agg_skills["y_correct"] = agg_skills[["y_correct", "y_all"]].min(axis=1)

print(f"Aggregated data shape: {agg_skills.shape}")
print(f"Models: {agg_skills['model'].nunique()}")
print(f"Skills: {agg_skills['skill'].nunique()}")

# Fit model - NOT controlling for category (parallel to Q1)
model_q1b = bmb.Model(
    "p(y_correct, y_all) ~ 0 + skill + (1|dataset) + (1|model)",
    agg_skills,
    family="binomial",
)
fit_q1b = model_q1b.fit(
    tune=2000, draws=5000, target_accept=0.95, cores=4, max_treedepth=12
)
print("\nModel fitted successfully!")

# %%
# Get predictions - skill difficulty independent of category
pred_skills_conditional = bmb.interpret.predictions(
    model_q1b,
    fit_q1b,
    average_by=["skill"],
)

# By-dataset predictions
pred_skills_by_dataset = bmb.interpret.predictions(
    model_q1b,
    fit_q1b,
    average_by=["skill", "dataset"],
)

# Filter to valid combinations
valid_pairs = eval_skills[["dataset", "skill"]].drop_duplicates().reset_index(drop=True)
pred_skills_by_dataset = pred_skills_by_dataset.merge(
    valid_pairs, on=["dataset", "skill"], how="inner"
)

print("\nSkill difficulty (controlling for category):")
print(pred_skills_conditional.sort_values("estimate", ascending=False))

print("\nSkill difficulty by dataset (controlling for category):")
print(
    pred_skills_by_dataset.sort_values(["dataset", "estimate"], ascending=[True, False])
)

# Save
pred_skills_conditional.to_csv(
    DATA_DIR / "bayesian_skill_difficulty_controlled.csv", index=False
)
pred_skills_by_dataset.to_csv(
    DATA_DIR / "bayesian_skill_difficulty_by_dataset_controlled.csv", index=False
)
print(f" Saved: {DATA_DIR / 'bayesian_skill_difficulty_controlled.csv'}")
print(f" Saved: {DATA_DIR / 'bayesian_skill_difficulty_by_dataset_controlled.csv'}")

# %%
# Visualize
fig, ax = plot_estimates_with_hdi(
    pred_skills_conditional,
    alpha=1.0,
    error_bar_width=2,
    plot_column="skill",
    color="darkorange",
)
ax.set_xlabel("Accuracy", fontsize=12)
ax.set_ylabel("Skill", fontsize=12)
ax.set_title(
    "Individual Skill Difficulty\n(controlling for dataset, model)", fontsize=14, pad=15
)
plt.tight_layout()
plt.savefig(
    OUTPUTS_DIR / "q1b_skill_difficulty_controlled.png", dpi=300, bbox_inches="tight"
)
plt.show()

# %%
# QUESTION 2: SKILL CONTRIBUTION TO CATEGORY DIFFICULTY
# Aggregate WITH both category and skill
agg_q2 = eval_skills.groupby(
    ["model", "dataset", "category", "skill"],
    as_index=False,
).agg(
    y_all=("skill_weight", "sum"),
    y_correct=("weighted_success", "sum"),
)

# Scale to integers
agg_q2["y_all"] = np.round(agg_q2["y_all"] * 100).astype(int)
agg_q2["y_correct"] = np.round(agg_q2["y_correct"] * 100).astype(int)
agg_q2["y_correct"] = agg_q2[["y_correct", "y_all"]].min(axis=1)

# Model with binomial
model_q2 = bmb.Model(
    "p(y_correct, y_all) ~ 0 + category + skill + (1|dataset) + (1|model)",
    agg_q2,
    family="binomial",
)
fit_q2 = model_q2.fit(tune=2000, draws=5000, target_accept=0.95, cores=4)

# %%
# PREDICTIONS: Category difficulty after controlling for skills
print("\n" + "=" * 70)
print("Q2 PREDICTIONS: Category difficulty (controlling for skills)")
print("=" * 70)

pred_q2_conditional = bmb.interpret.predictions(
    model_q2,
    fit_q2,
    conditional={
        "category": agg_q2["category"].unique(),
        "skill": agg_q2["skill"].unique(),
        "dataset": agg_q2["dataset"].unique(),
        "model": agg_q2["model"].unique(),
    },
    average_by=["category"],
)

print("\nCategory difficulty (controlling for skills):")
print(pred_q2_conditional.sort_values("estimate", ascending=False))

print(f"\n Saved: {DATA_DIR / 'bayesian_category_difficulty_controlling_skills.csv'}")

# %%
# SKILL COMPOSITION ANALYSIS: How much does skill composition explain category difficulty?
print("\n" + "=" * 70)
print("SKILL COMPOSITION ANALYSIS")
print("=" * 70)

# Merge Q1 (without skills) and Q2 (with skills controlled)
comparison = pred_q1_conditional.merge(
    pred_q2_conditional,
    on="category",
    suffixes=("_without_skills", "_with_skills"),
)

# Calculate the change when skills are added to the model
comparison["change_when_controlling_skills"] = (
    comparison["estimate_with_skills"] - comparison["estimate_without_skills"]
)

# Sort by change (largest positive = most explained by skills)
comparison_sorted = comparison.sort_values(
    "change_when_controlling_skills", ascending=False
)

print("\nCategories where difficulty is MOST explained by skill composition:")
print(
    comparison_sorted[
        [
            "category",
            "estimate_without_skills",
            "estimate_with_skills",
            "change_when_controlling_skills",
        ]
    ].head()
)

print("\nCategories with difficulty BEYOND skill composition (inherently hard):")
print(
    comparison_sorted[
        [
            "category",
            "estimate_without_skills",
            "estimate_with_skills",
            "change_when_controlling_skills",
        ]
    ].tail()
)

print(f"\nSaved: {DATA_DIR / 'bayesian_skill_composition_analysis.csv'}")

# %%
# VISUALIZATION 1: Side-by-side comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Category difficulty without controlling for skills (Q1)
ax = axes[0]
df_sorted = pred_q1_conditional.sort_values("estimate")
y_pos = np.arange(len(df_sorted))

ax.errorbar(
    df_sorted["estimate"],
    y_pos,
    xerr=[
        df_sorted["estimate"] - df_sorted["lower_3.0%"],
        df_sorted["upper_97.0%"] - df_sorted["estimate"],
    ],
    fmt="o",
    color="steelblue",
    markersize=8,
    capsize=5,
    capthick=2,
    elinewidth=2,
    markeredgewidth=0,
)

ax.set_yticks(y_pos)
ax.set_yticklabels(df_sorted["category"])
ax.set_xlabel("Accuracy", fontsize=12)
ax.set_ylabel("Category", fontsize=12)
ax.set_title(
    "Category Difficulty\n(Without controlling for skills)", fontsize=14, pad=15
)
ax.grid(True, axis="x", alpha=0.3, linestyle="--", linewidth=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Plot 2: Category difficulty with skills controlled (Q2)
ax = axes[1]
df_sorted = pred_q2_conditional.sort_values("estimate")
y_pos = np.arange(len(df_sorted))

ax.errorbar(
    df_sorted["estimate"],
    y_pos,
    xerr=[
        df_sorted["estimate"] - df_sorted["lower_3.0%"],
        df_sorted["upper_97.0%"] - df_sorted["estimate"],
    ],
    fmt="o",
    color="steelblue",
    markersize=8,
    capsize=5,
    capthick=2,
    elinewidth=2,
    markeredgewidth=0,
)

ax.set_yticks(y_pos)
ax.set_yticklabels(df_sorted["category"])
ax.set_xlabel("Accuracy", fontsize=12)
ax.set_ylabel("Category", fontsize=12)
ax.set_title(
    "Category Difficulty\n(Controlling for skill composition)", fontsize=14, pad=15
)
ax.grid(True, axis="x", alpha=0.3, linestyle="--", linewidth=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(
    OUTPUTS_DIR / "q2_skill_composition_comparison.png", dpi=300, bbox_inches="tight"
)
plt.show()

# %%
# VISUALIZATION 2: Change in difficulty estimates
fig, ax = plt.subplots(figsize=(10, 6))

comparison_plot = comparison_sorted.sort_values("change_when_controlling_skills")
y_pos = np.arange(len(comparison_plot))

colors = np.where(
    comparison_plot["change_when_controlling_skills"] > 0.02, "steelblue", "lightcoral"
)

bars = ax.barh(
    y_pos,
    comparison_plot["change_when_controlling_skills"] * 100,
    color=colors,
    alpha=0.7,
    edgecolor="black",
    linewidth=0.5,
)

ax.set_yticks(y_pos)
ax.set_yticklabels(comparison_plot["category"])
ax.set_xlabel(
    "Change in Accuracy Estimate (percentage points)", fontsize=12, fontweight="bold"
)
ax.set_ylabel("Category", fontsize=12, fontweight="bold")
ax.set_title(
    "Change When Controlling for Skill Composition",
    fontsize=13,
    fontweight="bold",
    pad=15,
)
ax.axvline(x=0, color="black", linestyle="-", linewidth=1.5)
ax.grid(True, axis="x", alpha=0.3)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

legend_elements = [
    Patch(
        facecolor="steelblue",
        alpha=0.7,
        label="Difficulty explained by skill composition",
    ),
    Patch(
        facecolor="lightcoral", alpha=0.7, label="Difficulty beyond skill composition"
    ),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=11)

plt.tight_layout()
plt.savefig(
    OUTPUTS_DIR / "q2_skill_composition_change.png", dpi=300, bbox_inches="tight"
)
plt.show()

print("\n" + "=" * 70)
print("SKILL COMPOSITION ANALYSIS COMPLETE")
print("=" * 70)
print("\nInterpretation:")
print(
    "- Positive change: Category appears hard mainly because it tests difficult skills"
)
print("- Near-zero change: Category has difficulty beyond its skill composition")

# %%
# EXPORT RESULTS TO CSV (for visualisations)
print("\n" + "=" * 70)
print("EXPORTING RESULTS")
print("=" * 70)

# Overall accuracy by model and dataset (using Q1 predictions)
model_dataset_accuracy = (
    eval_df.groupby(["model", "dataset"])["correct"].mean().reset_index()
)
model_dataset_accuracy = model_dataset_accuracy.rename(columns={"correct": "accuracy"})
model_dataset_accuracy.to_csv(
    DATA_DIR / "visual_reasoning_accuracy_results_gold.csv", index=False
)
print(f" Saved: {DATA_DIR / 'visual_reasoning_accuracy_results_gold.csv'}")

# Category performance by model
model_cat = eval_df.groupby(["model", "category"])["correct"].mean().reset_index()
category_pivot = model_cat.pivot(index="category", columns="model", values="correct")
category_pivot.to_csv(DATA_DIR / "model_reasoning_category_results_gold.csv")
print(f" Saved: {DATA_DIR / 'model_reasoning_category_results_gold.csv'}")

# Category performance by model AND dataset
model_cat_dataset = (
    eval_df.groupby(["model", "category", "dataset"])["correct"].mean().reset_index()
)
model_cat_dataset = model_cat_dataset.rename(columns={"correct": "accuracy"})
model_cat_dataset.to_csv(
    DATA_DIR / "model_reasoning_category_results_by_dataset_gold.csv", index=False
)
print(f" Saved: {DATA_DIR / 'model_reasoning_category_results_by_dataset_gold.csv'}")

# BONUS: Export Bayesian estimates too
pred_q1_conditional.to_csv(
    DATA_DIR / "bayesian_category_difficulty_conditional.csv", index=False
)
print(f" Saved: {DATA_DIR / 'bayesian_category_difficulty_conditional.csv'}")

print("\nAll results exported successfully!")

# %%
