# %%
import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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


def load_catalog():
    """Load question catalog with error annotations"""
    catalogs = []
    for dataset, path in [("zambia", ZAMBIA_PATH), ("india", INDIA_PATH)]:
        df = pd.read_csv(path)
        df = df.rename(columns=str.strip)
        df = df[
            ["id", "solution", "category", "error_type_value", "error_severity_value"]
        ]
        df["error_type_value"] = df["error_type_value"].fillna("none").str.strip()
        df["error_severity_value"] = (
            df["error_severity_value"].fillna("none").str.strip()
        )
        df["error_flag"] = (df["error_severity_value"] != "none").astype(int)
        df = df.assign(dataset=dataset)
        catalogs.append(df)

    return pd.concat(catalogs, ignore_index=True)


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


# %%
# LOAD AND MERGE DATA

catalog = load_catalog()
responses = load_model_responses()
eval_df = responses.merge(catalog, on=["dataset", "id"], how="inner")
eval_df["correct"] = (eval_df["model_answer"] == eval_df["solution"]).astype(int)

print(f"Loaded {len(eval_df)} responses from {eval_df['model'].nunique()} models")

# %%
# STEP 1: Prepare the data for analysis
# Create a clean dataset with three severity levels: no_error (clean questions),
# minor errors and medium errors. Fatal errors are filtered out since they're too severe
# for meaningful comparison.

print("\n" + "=" * 70)
print("STEP 1: PREPARING DATA WITH ALL SEVERITY LEVELS")
print("=" * 70)

severity_all_df = eval_df.copy()
severity_all_df["severity_level"] = severity_all_df["error_severity_value"].replace(
    "none", "no_error"
)
severity_all_df = severity_all_df[severity_all_df["severity_level"] != "fatal"].copy()

severity_order = ["no_error", "minor", "medium"]
severity_all_df["severity_level"] = pd.Categorical(
    severity_all_df["severity_level"], categories=severity_order, ordered=True
)

print(f"\nTotal observations: {len(severity_all_df)}")
print(f"Models: {severity_all_df['model'].nunique()}")
print(f"Datasets: {severity_all_df['dataset'].nunique()}")

print("\nDistribution across severity levels:")
severity_counts = severity_all_df["severity_level"].value_counts().sort_index()
print(severity_counts)
print("\nPercentages:")
print((severity_counts / len(severity_all_df) * 100).round(1))

print("\nAccuracy by severity level:")
severity_stats = severity_all_df.groupby("severity_level", observed=False)[
    "correct"
].agg(["mean", "count"])
print(severity_stats)

# %%
# STEP 2: Fit Bayesian hierarchical model
# Estimates how error severity affects question difficulty while controlling
# for random variation across models and datasets. The baseline is 'no_error' questions
# and coefficients show how much harder minor and medium errors make questions.

print("\n" + "=" * 70)
print("STEP 2: FITTING BAYESIAN MODEL")
print("=" * 70)

model_data = severity_all_df[
    ["correct", "severity_level", "category", "model", "dataset"]
].copy()
model_data["severity_level"] = model_data["severity_level"].astype(str)
model_data["category"] = model_data["category"].astype(str)

print(
    "\nModel formula: correct ~ C(severity_level, Treatment('no_error')) + (1|category) + (1|model) + (1|dataset)"
)
print("This estimates how each severity level affects accuracy")
print(
    "relative to 'no_error' baseline, controlling for category, model, and dataset effects"
)

print(f"\nData shape: {model_data.shape}")

severity_model = bmb.Model(
    "correct ~ C(severity_level, Treatment('no_error')) + (1|category) + (1|model) + (1|dataset)",
    model_data,
    family="bernoulli",
)

print("\nModel structure:")
print(severity_model)

print("\nSampling from posterior...")
severity_fit = severity_model.fit(
    tune=2000, draws=5000, target_accept=0.95, cores=4, max_treedepth=12
)
print("Model fitted successfully!")

# %%
# Extract and interpret severity effects using marginal effects
# Use Bambi's interpret.predictions() to get predicted accuracy for each severity level.
# This uses unit-level (empirical) marginalization - averaging over actual observations
# while accounting for task category, model, and dataset context.

print("\n" + "=" * 70)
print("COMPUTING PREDICTIONS (Unit-level/Empirical Marginalization)")
print("=" * 70)

print("\nUsing Bambi's interpret.predictions() to compute severity effects...")
print("This computes predicted accuracy for each severity level,")
print("accounting for the full context (category, model, dataset)")
print("and weighted by the empirical distribution of observations.")

# Use predictions to get predicted accuracy for each severity level
# This uses unit-level (empirical) marginalization - averages over actual observations
summary_empirical = bmb.interpret.predictions(
    severity_model, severity_fit, average_by=["severity_level"]
)

print("\nPredicted accuracy by severity level:")
print(
    summary_empirical[
        ["severity_level", "estimate", "lower_3.0%", "upper_97.0%"]
    ].to_string(index=False)
)

# Also get the log-odds coefficients for reference
all_params = az.summary(severity_fit)
print("\nAll available parameters:")
print(all_params.index.tolist())

severity_params = [p for p in all_params.index if "severity_level" in p.lower()]
print(f"\nSeverity parameters found: {severity_params}")

if len(severity_params) > 0:
    severity_coeffs = all_params.loc[severity_params]
    severity_coeffs["severity"] = severity_coeffs.index.str.replace(
        r"C\(severity_level, Treatment\('no_error'\)\)\[T\.", "", regex=True
    ).str.replace("]", "")
else:
    severity_coeffs = pd.DataFrame()

# Create interpretable summary from predictions
results_list = []

for _, row in summary_empirical.iterrows():
    severity = row["severity_level"]

    result_dict = {
        "severity": severity,
        "predicted_accuracy": row["estimate"] * 100,
        "pred_lower": row["lower_3.0%"] * 100,
        "pred_upper": row["upper_97.0%"] * 100,
    }

    # Add log-odds coefficients where available (for non-reference categories)
    if len(severity_coeffs) > 0:
        coeff_data = severity_coeffs[severity_coeffs["severity"] == severity]
        if len(coeff_data) > 0:
            result_dict["log_odds_mean"] = coeff_data["mean"].values[0]
            result_dict["log_odds_lower"] = coeff_data["hdi_3%"].values[0]
            result_dict["log_odds_upper"] = coeff_data["hdi_97%"].values[0]
            result_dict["odds_ratio"] = np.exp(coeff_data["mean"].values[0])
        else:
            # This is the reference category (no_error)
            result_dict["log_odds_mean"] = 0.0
            result_dict["log_odds_lower"] = 0.0
            result_dict["log_odds_upper"] = 0.0
            result_dict["odds_ratio"] = 1.0

    results_list.append(result_dict)

results_table = pd.DataFrame(results_list)

# Sort by predicted accuracy (high to low = easy to hard)
results_table = results_table.sort_values("predicted_accuracy", ascending=False)

print("\n" + "=" * 70)
print("SEVERITY LEVEL DIFFICULTY RANKING")
print("=" * 70)

print("\nSeverity levels ranked by predicted accuracy (easiest to hardest):")
print("\n{:<15} {:>18} {:>25}".format("Severity", "Pred. Accuracy", "94% HDI"))
print("-" * 65)
for idx, row in results_table.iterrows():
    print(
        "{:<15} {:>17.1f}% [{:>6.1f}% to {:>6.1f}%]".format(
            row["severity"],
            row["predicted_accuracy"],
            row["pred_lower"],
            row["pred_upper"],
        )
    )

print("\n" + "=" * 70)
print("INTERPRETATION GUIDE:")
print("=" * 70)
print("• Predicted Accuracy: Expected accuracy for each severity level")
print("  - Computed using unit-level (empirical) marginalization")
print("  - Averages predictions across all actual observations")
print("  - Accounts for context (category, model, dataset)")
print("  - Weighted by empirical distribution of observations")
print("• Lower predicted accuracy = harder (more impactful errors)")
print("• Higher predicted accuracy = easier (less impactful errors)")
print("• HDI: 94% Highest Density Interval (Bayesian credible interval)")
print(
    "• Example: 45% means models correctly answer 45% of questions with this severity level"
)

# Save results
results_table.to_csv(DATA_DIR / "bayesian_severity_effects.csv", index=False)
print(f"\nSaved: {DATA_DIR / 'bayesian_severity_effects.csv'}")

# %%
# STEP 3: Create model performance tiers
# Divide models into three groups based on overall accuracy: Top tier (best performers),
# Middle tier (mid-range), and Low tier (weakest). This lets us compare how different
# performance levels handle errors.

print("\n" + "=" * 70)
print("STEP 3: CREATING MODEL BASKETS")
print("=" * 70)

model_overall_acc = (
    severity_all_df.groupby("model")["correct"].mean().sort_values(ascending=False)
)

print(f"\nTotal models: {len(model_overall_acc)}")
print(
    f"\nOverall accuracy range: {model_overall_acc.min():.3f} - {model_overall_acc.max():.3f}"
)

# Split into 3 equal baskets
n_models = len(model_overall_acc)
top_n = n_models // 3
middle_n = n_models // 3
low_n = n_models - top_n - middle_n  # Remaining models go to low basket

top_models = set(model_overall_acc.iloc[:top_n].index)
middle_models = set(model_overall_acc.iloc[top_n : top_n + middle_n].index)
low_models = set(model_overall_acc.iloc[top_n + middle_n :].index)

print(
    f"\nTop models ({len(top_models)}): accuracy {model_overall_acc.iloc[:top_n].min():.3f} - {model_overall_acc.iloc[:top_n].max():.3f}"
)
print(
    f"Middle models ({len(middle_models)}): accuracy {model_overall_acc.iloc[top_n : top_n + middle_n].min():.3f} - {model_overall_acc.iloc[top_n : top_n + middle_n].max():.3f}"
)
print(
    f"Low models ({len(low_models)}): accuracy {model_overall_acc.iloc[top_n + middle_n :].min():.3f} - {model_overall_acc.iloc[top_n + middle_n :].max():.3f}"
)


# Add basket column to dataframe
def assign_basket(model):
    if model in top_models:
        return "Top"
    elif model in middle_models:
        return "Middle"
    else:
        return "Low"


severity_all_df["model_basket"] = severity_all_df["model"].apply(assign_basket)

print("\nTop models:")
print(list(top_models))
print("\nMiddle models:")
print(list(middle_models))
print("\nLow models:")
print(list(low_models))

# %%
# STEP 4: Heatmap visualization
# Shows accuracy for each model across severity levels. Models are sorted by overall
# performance with the best at top. Color coding: Green = higher accuracy, Red = lower.

print("\n" + "=" * 70)
print("STEP 4: CREATING HEATMAP VISUALIZATION")
print("=" * 70)

heatmap_data = (
    severity_all_df.groupby(["model", "severity_level"], observed=False)["correct"]
    .mean()
    .reset_index()
    .pivot(index="model", columns="severity_level", values="correct")
)

# Reorder columns
heatmap_data = heatmap_data[severity_order]

# Sort rows by overall accuracy (descending)
heatmap_data = heatmap_data.reindex(model_overall_acc.index)

print(f"\nHeatmap shape: {heatmap_data.shape}")

# Create heatmap
fig, ax = plt.subplots(figsize=(10, 14))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    vmin=0,
    vmax=1,
    cbar_kws={"label": "Accuracy"},
    ax=ax,
    linewidths=0.5,
)

ax.set_xlabel("Error Severity Level", fontsize=12, fontweight="bold")
ax.set_ylabel("Model", fontsize=12, fontweight="bold")
ax.set_title(
    "Model Performance by Error Severity (3 Levels)\nSorted by overall accuracy",
    fontsize=14,
    fontweight="bold",
    pad=15,
)

plt.xticks(rotation=0)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
plt.savefig(PLOTS_DIR / "severity_3levels_heatmap.png", dpi=300, bbox_inches="tight")
plt.savefig(
    PLOTS_DIR / "severity_3levels_heatmap.svg", format="svg", bbox_inches="tight"
)
plt.show()

print("Heatmap saved")

# %%
# STEP 5: Line plot by model tier
# Shows how each tier of models (Top/Middle/Low) performs across the three severity
# levels. We can see if better models are more robust to errors.

print("\n" + "=" * 70)
print("STEP 5: CREATING LINE PLOT BY MODEL BASKET")
print("=" * 70)

basket_severity = (
    severity_all_df.groupby(["model_basket", "severity_level"], observed=False)[
        "correct"
    ]
    .agg(["mean", "count"])
    .reset_index()
)

print("\nAccuracy by basket and severity:")
print(basket_severity)

# Create line plot
fig, ax = plt.subplots(figsize=(10, 6))

# Define colors and markers for each basket
basket_styles = {
    "Top": {"color": "#2ecc71", "marker": "o", "linewidth": 2.5},
    "Middle": {"color": "#f39c12", "marker": "s", "linewidth": 2.5},
    "Low": {"color": "#e74c3c", "marker": "^", "linewidth": 2.5},
}

# Plot each basket
for basket in ["Top", "Middle", "Low"]:
    basket_data = basket_severity[basket_severity["model_basket"] == basket]

    ax.plot(
        range(len(severity_order)),
        basket_data["mean"].values,
        label=f"{basket} Models",
        marker=basket_styles[basket]["marker"],
        color=basket_styles[basket]["color"],
        linewidth=basket_styles[basket]["linewidth"],
        markersize=10,
        markeredgewidth=2,
        markeredgecolor="white",
    )

ax.set_xticks(range(len(severity_order)))
ax.set_xticklabels(["No Error", "Minor", "Medium"], fontsize=11)
ax.set_xlabel("Error Severity Level", fontsize=12, fontweight="bold")
ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
ax.set_title(
    "Model Performance Across Error Severity Levels\nBy Model Tier (Top/Middle/Low)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)

ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
ax.set_axisbelow(True)
ax.legend(fontsize=11, loc="best", framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(
    PLOTS_DIR / "severity_3levels_lineplot_by_basket.png", dpi=300, bbox_inches="tight"
)
plt.savefig(
    PLOTS_DIR / "severity_3levels_lineplot_by_basket.svg",
    format="svg",
    bbox_inches="tight",
)
plt.show()

print("Line plot saved")

# %%
# STEP 6: Spaghetti plot of all models
# Each light blue line represents one model's performance trajectory across severity levels.
# The thick red line shows the average trend. This visualizes the overall pattern and
# individual variation.

print("\n" + "=" * 70)
print("STEP 6: CREATING SPAGHETTI PLOT (ALL MODELS)")
print("=" * 70)

model_severity = (
    severity_all_df.groupby(["model", "severity_level"], observed=False)["correct"]
    .mean()
    .reset_index()
    .pivot(index="model", columns="severity_level", values="correct")
)

# Reorder columns
model_severity = model_severity[severity_order]

print(f"\nData shape: {model_severity.shape}")
print(f"Number of models: {len(model_severity)}")

# Calculate overall mean across all models
mean_accuracy = model_severity.mean(axis=0)

print("\nOverall mean accuracy by severity level:")
for level, acc in zip(severity_order, mean_accuracy):
    print(f"  {level}: {acc:.3f}")

# Create spaghetti plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each model as a semi-transparent line
x_pos = np.arange(len(severity_order))
for model_name, row in model_severity.iterrows():
    ax.plot(x_pos, row.values, color="steelblue", alpha=0.15, linewidth=1.5, zorder=1)

# Plot mean line on top
ax.plot(
    x_pos,
    mean_accuracy.values,
    color="#e74c3c",
    linewidth=4,
    label="Mean Across All Models",
    marker="o",
    markersize=12,
    markeredgewidth=2,
    markeredgecolor="white",
    zorder=10,
)

ax.set_xticks(x_pos)
ax.set_xticklabels(["No Error", "Minor", "Medium"], fontsize=11)
ax.set_xlabel("Error Severity Level", fontsize=12, fontweight="bold")
ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
ax.set_title(
    "Model Performance Across Error Severity Levels",
    fontsize=14,
    fontweight="bold",
    pad=15,
)

ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
ax.set_axisbelow(True)
ax.legend(fontsize=11, loc="best", framealpha=0.9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(
    PLOTS_DIR / "severity_3levels_spaghetti_plot.png", dpi=300, bbox_inches="tight"
)
plt.savefig(
    PLOTS_DIR / "severity_3levels_spaghetti_plot.svg", format="svg", bbox_inches="tight"
)
plt.show()

print("Spaghetti plot saved")

# %%
