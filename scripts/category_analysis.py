# %%
import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vrb_benchmark import DATA_DIR, PROJECT_ROOT

# Configuration
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = PROJECT_ROOT / "data" / "plots_gold"
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
    "gpt-5.1-2025-11-13-high",
    "claude-opus-4-5-20251101-thinking-high",
}

PLOTS_DIR.mkdir(exist_ok=True, parents=True)

# %%
# HELPER FUNCTIONS


def load_catalog_all():
    """Load question catalog with all questions (including those with errors)"""
    catalogs = []
    for dataset, path in [("zambia", ZAMBIA_PATH), ("india", INDIA_PATH)]:
        df = pd.read_csv(path)
        df = df.rename(columns=str.strip)
        # Keep relevant columns
        cols = [
            "id",
            "solution",
            "category",
            "skills",
            "error_severity_value",
            "error_type_value",
        ]
        df = df[cols].copy()
        df["dataset"] = dataset
        catalogs.append(df)

    return pd.concat(catalogs, ignore_index=True)


def filter_catalog_for_analysis(catalog_all):
    """Filter out questions with errors for Bayesian analysis"""
    # Filter out questions with any error severity
    n_before = len(catalog_all)
    catalog_clean = catalog_all[catalog_all["error_severity_value"].isna()].copy()
    n_after = len(catalog_clean)

    print(f"\n{'=' * 70}")
    print("FILTERING QUESTIONS WITH ERRORS FOR BAYESIAN ANALYSIS")
    print(f"{'=' * 70}")
    print(f"Questions before filtering: {n_before}")
    print(f"Questions with errors removed: {n_before - n_after}")
    print(f"Questions after filtering: {n_after}")

    catalog_clean = catalog_clean.drop(
        columns=["error_severity_value", "error_type_value"]
    )

    return catalog_clean


def load_model_responses():
    """Load all model responses"""
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


print("=" * 70)
print("STEP 1: LOADING DATA")
print("=" * 70)

# Load all questions (including those with errors) for distribution plot
catalog_all = load_catalog_all()

print(f"\nTotal questions (including errors): {len(catalog_all)}")
print(f"Questions with errors: {catalog_all['error_severity_value'].notna().sum()}")

# %%
# STEP 2: Visualize category distribution
# Show the distribution of questions across all task categories before filtering

print("\n" + "=" * 70)
print("STEP 2: CATEGORY DISTRIBUTION (ALL QUESTIONS)")
print("=" * 70)

# Count how many questions for each category (all questions)
category_counts_all = (
    catalog_all["category"].value_counts().sort_values(ascending=False)
)
print(category_counts_all.to_string())

fig, ax = plt.subplots(figsize=(10, 6))

# Sort by count (descending)
plot_data = category_counts_all.sort_values(ascending=True)  # True for horizontal bars

# Create horizontal bar chart
bars = ax.barh(range(len(plot_data)), plot_data.values, color="steelblue", alpha=0.8)

# Add value labels on bars
for i, (cat, count) in enumerate(plot_data.items()):
    ax.text(count + 5, i, str(count), va="center", fontsize=10)

# Customize plot
ax.set_yticks(range(len(plot_data)))
ax.set_yticklabels(plot_data.index)
ax.set_xlabel("Number of Questions", fontsize=11)
ax.set_ylabel("Category", fontsize=11)
ax.set_title(
    "Distribution of Questions by Category (Combined Dataset)",
    fontsize=12,
    fontweight="bold",
)

ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.5)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "category_distribution.png", dpi=300, bbox_inches="tight")
plt.savefig(PLOTS_DIR / "category_distribution.svg", format="svg", bbox_inches="tight")
plt.show()

print("\nSaved category distribution plot")

# %%
# STEP 3: Filter data and load model responses
# Remove questions with errors and merge with model responses for analysis

catalog = filter_catalog_for_analysis(catalog_all)
responses = load_model_responses()

eval_df = responses.merge(catalog, on=["dataset", "id"], how="inner")
eval_df["correct"] = (eval_df["model_answer"] == eval_df["solution"]).astype(int)

print(f"\nLoaded {len(eval_df)} responses from {eval_df['model'].nunique()} models")
print(f"Total questions (after filtering): {eval_df['id'].nunique()}")

# Load skill indicators created by skill_difficulty_analysis
skill_indicators_path = DATA_DIR / "question_skill_indicators.csv"
if skill_indicators_path.exists():
    skill_indicators = pd.read_csv(skill_indicators_path)
    eval_df = eval_df.merge(skill_indicators, on=["id", "dataset"], how="left")
    skill_cols = [col for col in skill_indicators.columns if col.startswith("has_")]
    print(f"\nLoaded {len(skill_cols)} skill indicators from skill_difficulty_analysis")
else:
    print(f"\nWarning: Skill indicators file not found at {skill_indicators_path}")
    print("Run skill_difficulty_analysis.py first to generate skill indicators")
    skill_cols = []

# %%
# STEP 4: Explore task types and accuracy
# Examine the filtered dataset to understand task type distribution and overall accuracy by category

print("\n" + "=" * 70)
print("STEP 4: EXPLORING TASK TYPES (FILTERED DATA)")
print("=" * 70)

# Get all unique categories
unique_categories = sorted(catalog["category"].unique())

print(f"\nTotal unique task types: {len(unique_categories)}")
print("\nAll task types:")
for i, cat in enumerate(unique_categories, 1):
    print(f"  {i:2d}. {cat}")

# Count how many questions for each category
category_counts = catalog["category"].value_counts().sort_values(ascending=False)

print("\n" + "=" * 70)
print("TASK TYPE FREQUENCY (Number of questions per task type)")
print("=" * 70)
print(category_counts.to_string())

# Calculate accuracy by category
print("\n" + "=" * 70)
print("OVERALL ACCURACY BY TASK TYPE (across all models)")
print("=" * 70)

category_accuracy = eval_df.groupby("category")["correct"].agg(["mean", "count"])
category_accuracy.columns = ["accuracy", "n_responses"]
category_accuracy = category_accuracy.sort_values("accuracy", ascending=False)
category_accuracy["accuracy_pct"] = (category_accuracy["accuracy"] * 100).round(1)

print(category_accuracy[["accuracy_pct", "n_responses"]].to_string())


# %%
# STEP 5: Fit Bayesian hierarchical model
# Use hierarchical logistic regression to estimate how each task type affects question difficulty
# The model controls for model, dataset and skill differences

print("\n" + "=" * 70)
print("STEP 5: FITTING BAYESIAN HIERARCHICAL MODEL")
print("=" * 70)

# Use category directly as a categorical predictor, controlling for skills
# Bambi will automatically handle the reference category
if skill_cols:
    skill_formula = " + ".join(skill_cols)
    formula = f"correct ~ category + {skill_formula} + (1|model) + (1|dataset)"
    model_cols = ["correct", "category", "model", "dataset"] + skill_cols
    print("\nFitting Bayesian hierarchical model with task type as predictor")
    print("Model: correct ~ category + [skills] + (1|model) + (1|dataset)")
    print(
        "\nThis estimates the effect of each task type on question difficulty while controlling for:"
    )
    print("  - Cognitive skills required (independent skill effects)")
    print("  - Model variation (some models are better/worse overall)")
    print("  - Dataset effects (India vs Zambia)")
else:
    formula = "correct ~ category + (1|model) + (1|dataset)"
    model_cols = ["correct", "category", "model", "dataset"]
    print("\nFitting Bayesian hierarchical model with task type as predictor")
    print(f"Model: {formula}")
    print(
        "\nThis estimates the effect of each task type on question difficulty while controlling for:"
    )
    print("  - Model variation (some models are better/worse overall)")
    print("  - Dataset effects (India vs Zambia)")

model_data = eval_df[model_cols].copy()

print(f"\nData shape: {model_data.shape}")
print(f"Unique models: {model_data['model'].nunique()}")
print(f"Unique datasets: {model_data['dataset'].nunique()}")
print(f"Unique task types: {model_data['category'].nunique()}")

model_all_categories = bmb.Model(formula, model_data, family="bernoulli")

print("\nModel structure:")
print(model_all_categories)

# Fit the model
print("\nSampling from posterior")
fit_all_categories = model_all_categories.fit(
    tune=2000, draws=5000, target_accept=0.99, cores=4, max_treedepth=12, random_seed=42
)
print("Model fitted successfully!")

# %%
# STEP 6: Check model diagnostics
# Verify that the MCMC sampling worked correctly by checking for convergence issues

print("\n" + "=" * 70)
print("STEP 6: CHECKING MODEL DIAGNOSTICS")
print("=" * 70)

# Check divergences
n_divergences = fit_all_categories.sample_stats.diverging.sum().item()
print(f"\nNumber of divergences: {n_divergences}")

# Get full summary to check R-hat and ESS
full_summary = az.summary(fit_all_categories)
print("\nParameters with R-hat > 1.01:")
problematic_rhat = full_summary[full_summary["r_hat"] > 1.01]
if len(problematic_rhat) > 0:
    print(problematic_rhat[["mean", "r_hat", "ess_bulk"]])
else:
    print("None - all parameters converged")

print("\nParameters with ESS < 400:")
low_ess = full_summary[full_summary["ess_bulk"] < 400]
if len(low_ess) > 0:
    print(low_ess[["mean", "r_hat", "ess_bulk"]])
else:
    print("None - all parameters have sufficient samples")

# %%
# STEP 7: Extract and interpret task type effects
# Convert the model coefficients from log-odds to interpretable metrics like odds ratios
# and percentage point changes in accuracy relative to the reference category

print("\n" + "=" * 70)
print("STEP 7: EXTRACTING TASK TYPE EFFECT ESTIMATES")
print("=" * 70)

# Get summary of fixed effects (category coefficients)
# Bambi creates coefficients like "category[T.odd one out]" for non-reference levels
summary_all_categories = az.summary(fit_all_categories, var_names=["category"])

# Clean up category names for display
# Remove both possible formats: 'category[T.xxx]' and 'category[xxx'
summary_all_categories["task_type"] = (
    summary_all_categories.index.str.replace("category[T.", "", regex=False)
    .str.replace("category[", "", regex=False)
    .str.replace("]", "", regex=False)
)

print("\nTask type effects (sorted by mean coefficient - negative = harder):")
print(
    summary_all_categories[
        ["task_type", "mean", "sd", "hdi_3%", "hdi_97%"]
    ].sort_values("mean")
)

# Identify reference category
reference_category = [
    cat
    for cat in unique_categories
    if not any(cat in idx for idx in summary_all_categories.index)
][0]
print(f"\nReference task type (baseline): {reference_category}")

# Convert log-odds to probability effects for interpretation
print("\n" + "=" * 70)
print("INTERPRETABLE EFFECTS (converted from log-odds)")
print("=" * 70)

# Create interpretable summary
interp_summary = summary_all_categories[
    ["task_type", "mean", "hdi_3%", "hdi_97%"]
].copy()

# Convert to odds ratios
interp_summary["odds_ratio"] = np.exp(interp_summary["mean"])
interp_summary["or_lower"] = np.exp(interp_summary["hdi_3%"])
interp_summary["or_upper"] = np.exp(interp_summary["hdi_97%"])

# Compute predictions for each category using unit-level (empirical) marginalization
print("\n" + "=" * 70)
print("COMPUTING PREDICTIONS (Unit-level/Empirical Marginalization)")
print("=" * 70)
print("\nUsing Bambi's interpret.predictions() to compute category effects...")
print("This computes predicted accuracy for each category,")
print("accounting for the full context (skills, model, dataset)")
print("and weighted by the empirical distribution of observations.")

# Use predictions to get predicted accuracy for each category
# This uses unit-level (empirical) marginalization - averages over actual observations
# Use subset of posterior samples for speed (use every 5th sample)
summary_empirical = bmb.interpret.predictions(
    model_all_categories,
    fit_all_categories.sel(draw=slice(None, None, 5)),
    average_by=["category"],
)

print("\nPredicted accuracy by category:")
print(
    summary_empirical[["category", "estimate", "lower_3.0%", "upper_97.0%"]].to_string(
        index=False
    )
)

# Create a new summary from the predictions (which includes ALL categories)
# This replaces interp_summary which was missing the reference category
interp_summary_new = []

for _, row in summary_empirical.iterrows():
    cat = row["category"]

    interp_summary_new.append(
        {
            "task_type": cat,
            "predicted_accuracy": row["estimate"] * 100,
            "pred_lower": row["lower_3.0%"] * 100,
            "pred_upper": row["upper_97.0%"] * 100,
        }
    )

# Convert to DataFrame
interp_summary = pd.DataFrame(interp_summary_new)

# Add the log-odds info from original summary where available (for non-reference categories)
# The reference category won't have these values, which is fine
for idx, row in interp_summary.iterrows():
    cat = row["task_type"]
    orig_data = summary_all_categories[
        summary_all_categories.index.str.contains(cat, regex=False)
    ]

    if len(orig_data) > 0:
        interp_summary.at[idx, "log_odds_mean"] = orig_data["mean"].values[0]
        interp_summary.at[idx, "log_odds_lower"] = orig_data["hdi_3%"].values[0]
        interp_summary.at[idx, "log_odds_upper"] = orig_data["hdi_97%"].values[0]
        interp_summary.at[idx, "odds_ratio"] = np.exp(orig_data["mean"].values[0])
    else:
        # This is the reference category - set to NaN or 0
        interp_summary.at[idx, "log_odds_mean"] = 0.0
        interp_summary.at[idx, "log_odds_lower"] = 0.0
        interp_summary.at[idx, "log_odds_upper"] = 0.0
        interp_summary.at[idx, "odds_ratio"] = 1.0

# For compatibility with existing code, keep prob_change columns
# But now they represent predicted accuracy rather than change from reference
interp_summary["prob_change"] = interp_summary["predicted_accuracy"]
interp_summary["prob_change_lower"] = interp_summary["pred_lower"]
interp_summary["prob_change_upper"] = interp_summary["pred_upper"]

# Sort by predicted accuracy (low to high = hardest to easiest)
interp_summary_sorted = interp_summary.sort_values("predicted_accuracy")

print("\n" + "=" * 70)
print("CATEGORY DIFFICULTY RANKING")
print("=" * 70)

print("\nCategories ranked by difficulty (hardest to easiest):")
print("\n{:<35} {:>15} {:>25}".format("Task Type", "Pred. Accuracy", "94% HDI"))
print("-" * 80)
for idx, row in interp_summary_sorted.iterrows():
    print(
        "{:<35} {:>14.1f}% [{:>6.1f}% to {:>6.1f}%]".format(
            row["task_type"],
            row["predicted_accuracy"],
            row["pred_lower"],
            row["pred_upper"],
        )
    )

print("\n" + "=" * 70)
print("INTERPRETATION GUIDE:")
print("=" * 70)
print("• Predicted Accuracy: Expected accuracy for each category")
print("  - Computed using unit-level (empirical) marginalization")
print("  - Averages predictions across all actual observations")
print("  - Accounts for context (skills, model, dataset)")
print("  - Weighted by empirical distribution of observations")
print("• Lower predicted accuracy = harder category")
print("• Higher predicted accuracy = easier category")
print("• HDI: 94% Highest Density Interval (Bayesian credible interval)")
print(
    "• Example: 45% means models correctly answer questions in this category 45% of the time"
)

# Check for categories with high uncertainty
print("\n" + "=" * 70)
print("UNCERTAINTY ANALYSIS:")
print("=" * 70)
interp_summary_sorted["hdi_width"] = (
    interp_summary_sorted["prob_change_upper"]
    - interp_summary_sorted["prob_change_lower"]
)
high_uncertainty = interp_summary_sorted.nlargest(3, "hdi_width")
print("\nCategories with highest uncertainty (widest credible intervals):")
for idx, row in high_uncertainty.iterrows():
    cat_name = row["task_type"]
    n_questions = category_counts.get(cat_name, "NOT FOUND")
    # Get accuracy stats for this category
    if cat_name in category_accuracy.index:
        cat_acc = category_accuracy.loc[cat_name, "accuracy_pct"]
        cat_n_resp = category_accuracy.loc[cat_name, "n_responses"]
    else:
        cat_acc = "N/A"
        cat_n_resp = "N/A"

    print(f"\n  {cat_name}:")
    print(f"    - HDI width: {row['hdi_width']:.1f} percentage points")
    print(f"    - Number of questions: {n_questions}")
    print(f"    - Total responses (all models): {cat_n_resp}")
    print(f"    - Average accuracy: {cat_acc}%")
    print(
        f"    - Effect estimate: {row['prob_change']:.1f}pp [{row['prob_change_lower']:.1f} to {row['prob_change_upper']:.1f}]"
    )

# Save results
interp_summary_sorted.to_csv(DATA_DIR / "bayesian_task_difficulty_by_type.csv")
print(f"\nSaved: {DATA_DIR / 'bayesian_task_difficulty_by_type.csv'}")

# %%
# STEP 8: Visualize task type effects
# Create a forest plot showing which task types are harder or easier than the reference category

fig, ax = plt.subplots(figsize=(12, 8))

# Sort by predicted accuracy (low to high = hard to easy)
plot_data = interp_summary_sorted.copy()
y_pos = np.arange(len(plot_data))

# Calculate error bars for predicted accuracy
lower_errors = plot_data["prob_change"] - plot_data["prob_change_lower"]
upper_errors = plot_data["prob_change_upper"] - plot_data["prob_change"]

# Use single blue color for all points
colors = ["#2171b5"] * len(plot_data)  # Professional blue

# Plot error bars
for i, (idx, row) in enumerate(plot_data.iterrows()):
    ax.errorbar(
        row["prob_change"],
        i,
        xerr=[[lower_errors.iloc[i]], [upper_errors.iloc[i]]],
        fmt="o",
        color=colors[i],
        markersize=10,
        capsize=6,
        capthick=2.5,
        elinewidth=3,
        markeredgewidth=0,
        alpha=0.85,
        zorder=3,
    )

# Styling
ax.set_yticks(y_pos)
ax.set_yticklabels(plot_data["task_type"], fontsize=12)
ax.set_xlabel("Predicted Accuracy (%)", fontsize=14, fontweight="bold")
ax.set_ylabel("Task", fontsize=14, fontweight="bold")
ax.set_title(
    "Predicted Accuracy by Task",
    fontsize=16,
    fontweight="bold",
    pad=20,
)

ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.7)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(
    PLOTS_DIR / "category_difficulty_effects.svg",
    format="svg",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(PLOTS_DIR / "category_difficulty_effects.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved category difficulty visualization (SVG and PNG)")

# %%
# Red = Harder than reference  Blue = Easier than reference

# %%
