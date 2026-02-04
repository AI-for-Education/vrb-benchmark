# %%
import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis_utils import tokenize_skills

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
    """Load question catalog with skills"""
    catalogs = []
    for dataset, path in [("zambia", ZAMBIA_PATH), ("india", INDIA_PATH)]:
        df = pd.read_csv(path)
        df = df.rename(columns=str.strip)
        cols = ["id", "solution", "category", "skills", "error_severity_value"]
        df = df[cols].copy()
        df["dataset"] = dataset
        catalogs.append(df)

    catalog_all = pd.concat(catalogs, ignore_index=True)

    # Filter out questions with any errors
    n_before = len(catalog_all)
    catalog_clean = catalog_all[catalog_all["error_severity_value"].isna()].copy()
    n_after = len(catalog_clean)

    print(f"\n{'=' * 70}")
    print("FILTERING QUESTIONS WITH ERRORS")
    print(f"{'=' * 70}")
    print(f"Questions before filtering: {n_before}")
    print(f"Questions with errors removed: {n_before - n_after}")
    print(f"Questions after filtering: {n_after}")

    catalog_clean = catalog_clean.drop(columns=["error_severity_value"])

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


# %%
# STEP 1: Load and merge data

print("=" * 70)
print("STEP 1: LOADING DATA")
print("=" * 70)

catalog = load_catalog()
responses = load_model_responses()

eval_df = responses.merge(catalog, on=["dataset", "id"], how="inner")
eval_df["correct"] = (eval_df["model_answer"] == eval_df["solution"]).astype(int)

print(f"Loaded {len(eval_df)} responses from {eval_df['model'].nunique()} models")
print(f"Total questions: {eval_df['id'].nunique()}")

# %%
# STEP 2: Explore skill types and frequency

print("\n" + "=" * 70)
print("STEP 2: EXPLORING SKILL TYPES")
print("=" * 70)

# Get all unique skills
unique_skills = set()
for skills_str in catalog["skills"].dropna():
    unique_skills.update(tokenize_skills(skills_str))

unique_skills = sorted(unique_skills)

print(f"\nTotal unique skills: {len(unique_skills)}")
print("\nAll skills:")
for i, skill in enumerate(unique_skills, 1):
    print(f"  {i:2d}. {skill}")

# Count how many questions use each skill
skill_counts = {}
for skill in unique_skills:
    count = catalog["skills"].apply(lambda x: skill in tokenize_skills(x)).sum()
    skill_counts[skill] = count

skill_counts_df = pd.DataFrame(
    [{"skill": skill, "n_questions": count} for skill, count in skill_counts.items()]
).sort_values("n_questions", ascending=False)

print("\n" + "=" * 70)
print("SKILL FREQUENCY (Number of questions using each skill)")
print("=" * 70)
print(skill_counts_df.to_string(index=False))

# %%
# STEP 3: Create binary skill indicators
# Transform the skills data into binary indicators (0/1) for each skill
# This allows us to use them as predictors in the regression model

print("\n" + "=" * 70)
print("STEP 3: CREATING BINARY SKILL INDICATORS")
print("=" * 70)

# Start with question-level data
question_skill_df = eval_df.copy()

# Create binary indicator for each skill
for skill in unique_skills:
    col_name = f"has_{skill.lower().replace(' ', '_').replace('&', 'and')}"

    question_skill_df[col_name] = question_skill_df["skills"].apply(
        lambda x: 1 if skill in tokenize_skills(x) else 0
    )

# skill indicator excluding other
skill_cols = [
    f"has_{skill.lower().replace(' ', '_').replace('&', 'and')}"
    for skill in unique_skills
    if skill.lower() != "other"
]

print(f"\nCreated {len(skill_cols)} binary skill indicator columns")
print(f"Question-level dataframe shape: {question_skill_df.shape}")
print("\nSkill indicator columns:")
for col in skill_cols:
    n_questions = question_skill_df[col].sum()
    print(f"  {col}: {n_questions} questions")

print("\nPreview of question-level data with skill indicators:")
print(
    question_skill_df[
        ["id", "model", "dataset", "category", "correct"] + skill_cols[:3]
    ].head(10)
)

# Export skill indicators for use in other analyses
skill_indicators_export = question_skill_df[
    ["id", "dataset"] + skill_cols
].drop_duplicates(subset=["id", "dataset"])
skill_indicators_export.to_csv(DATA_DIR / "question_skill_indicators.csv", index=False)
print(f"\nSaved skill indicators: {DATA_DIR / 'question_skill_indicators.csv'}")

# %%
# STEP 4: Fit Bayesian hierarchical model
# Use hierarchical logistic regression to estimate how each skill affects question difficulty
# The model controls for model, dataset differences and tasks (categories)

print("\n" + "=" * 70)
print("STEP 4: FITTING BAYESIAN HIERARCHICAL MODEL")
print("=" * 70)

# Create formula with all skill indicators
skill_formula = " + ".join(skill_cols)
formula_all_skills = f"correct ~ {skill_formula} + (1|model) + (1|dataset) + category"

print("\nFitting Bayesian hierarchical model with all skills as predictors")
print(f"Model: {formula_all_skills}")
print(
    "This estimates the effect of each skill on question difficulty while controlling for:"
)
print("  - Model variation (some models are better/worse overall)")
print("  - Dataset effects (India vs Zambia)")
print("  - Category effects (different question types)")

model_data_all_skills = question_skill_df[
    ["correct", "model", "dataset", "category"] + skill_cols
].copy()

print(f"\nData shape: {model_data_all_skills.shape}")
print(f"Unique models: {model_data_all_skills['model'].nunique()}")
print(f"Unique datasets: {model_data_all_skills['dataset'].nunique()}")
print(f"Unique categories: {model_data_all_skills['category'].nunique()}")

model_all_skills = bmb.Model(
    formula_all_skills,
    model_data_all_skills,
    family="bernoulli",
)

print("\nModel structure:")
print(model_all_skills)

# Fit the model
print("\nSampling from posterior")
fit_all_skills = model_all_skills.fit(
    tune=2000,
    draws=5000,
    target_accept=0.99,
    cores=4,
    max_treedepth=12,
    random_seed=42,
)
print("Model fitted successfully!")

# %%
# STEP 5: Check model diagnostics
# Checking for convergence issues (rHAT and ESS) to ensure reliable estimates

print("\n" + "=" * 70)
print("STEP 5: CHECKING MODEL DIAGNOSTICS")
print("=" * 70)

# Check divergences
n_divergences = fit_all_skills.sample_stats.diverging.sum().item()
print(f"\nNumber of divergences: {n_divergences}")

# Get full summary to check R-hat and ESS
full_summary = az.summary(fit_all_skills)
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
# STEP 6: Extract and interpret skill effects
# Convert the model coefficients from log-odds to interpretable metrics
# like odds ratios and percentage point changes in accuracy

print("\n" + "=" * 70)
print("STEP 6: EXTRACTING SKILL EFFECT ESTIMATES")
print("=" * 70)

# Get summary of fixed effects (skill coefficients)
summary_all_skills = az.summary(fit_all_skills, var_names=[col for col in skill_cols])

# Clean up skill names for display
summary_all_skills["skill_name"] = (
    summary_all_skills.index.str.replace("has_", "").str.replace("_", " ").str.title()
)

print("\nSkill effects (sorted by mean coefficient - negative = harder):")
print(
    summary_all_skills[["skill_name", "mean", "sd", "hdi_3%", "hdi_97%"]].sort_values(
        "mean"
    )
)

# Convert log-odds to probability effects using marginal effects
print("\n" + "=" * 70)
print("COMPUTING MARGINAL EFFECTS (Unit-level/Empirical Marginalization)")
print("=" * 70)

print(
    "\nUsing Bambi's interpret.comparisons() to compute Average Marginal Effects (AMEs)..."
)
print("This approach:")
print("  - Computes predictions for each actual observation in the data")
print("  - Compares predictions with vs. without each skill (contrast: 0 vs 1)")
print("  - Averages across observations, weighted by empirical distribution")
print("  - Accounts for full context (other skills, category, model, dataset)")

# Store results for each skill
marginal_effects_results = []

for skill_col in skill_cols:
    skill_name = (
        skill_col.replace("has_", "").replace("_", " ").replace("and", "&").title()
    )
    print(f"\n  Computing marginal effect for: {skill_name}...", end="", flush=True)

    comparison = bmb.interpret.comparisons(
        model_all_skills,
        fit_all_skills,
        contrast={skill_col: [0, 1]},
        average_by=True,
    )

    ame_mean = comparison["estimate"].values[0] * 100
    ame_lower = comparison["lower_3.0%"].values[0] * 100
    ame_upper = comparison["upper_97.0%"].values[0] * 100

    marginal_effects_results.append(
        {
            "skill_col": skill_col,
            "skill_name": skill_name,
            "ame_mean": ame_mean,
            "ame_lower": ame_lower,
            "ame_upper": ame_upper,
        }
    )

    print(f" AME = {ame_mean:+.2f}pp [94% HDI: {ame_lower:+.2f} to {ame_upper:+.2f}]")

# Convert to df
interp_summary = pd.DataFrame(marginal_effects_results)
interp_summary = interp_summary.set_index("skill_col")

# Add the original log-odds coefficients for reference
interp_summary = interp_summary.join(
    summary_all_skills[["mean", "hdi_3%", "hdi_97%"]], how="left"
)
interp_summary = interp_summary.rename(
    columns={
        "mean": "log_odds_mean",
        "hdi_3%": "log_odds_lower",
        "hdi_97%": "log_odds_upper",
    }
)

# Add odds ratios for reference
interp_summary["odds_ratio"] = np.exp(interp_summary["log_odds_mean"])
interp_summary["or_lower"] = np.exp(interp_summary["log_odds_lower"])
interp_summary["or_upper"] = np.exp(interp_summary["log_odds_upper"])

# Use AME as the primary interpretation metric
interp_summary["prob_change"] = interp_summary["ame_mean"]
interp_summary["prob_change_lower"] = interp_summary["ame_lower"]
interp_summary["prob_change_upper"] = interp_summary["ame_upper"]

print("\n" + "=" * 70)
print("MARGINAL EFFECTS COMPUTED")
print("=" * 70)
print(f"\nComputed Average Marginal Effects for {len(skill_cols)} skills")
print(
    f"Each AME represents the average change in probability across {len(model_data_all_skills)} observations"
)

# Sort by effect size (using AME)
interp_summary_sorted = interp_summary.sort_values("ame_mean")

print("\nSkills making questions HARDER (negative effect):")
harder_skills = interp_summary_sorted[interp_summary_sorted["ame_mean"] < 0]
print(
    "\n{:<25} {:>8} {:>12} {:>25}".format(
        "Skill", "Odds Ratio", "Prob Change", "94% HDI"
    )
)
print("-" * 75)
for idx, row in harder_skills.iterrows():
    print(
        "{:<25} {:>8.2f} {:>11.1f}pp [{:>6.1f}pp to {:>6.1f}pp]".format(
            row["skill_name"],
            row["odds_ratio"],
            row["prob_change"],
            row["prob_change_lower"],
            row["prob_change_upper"],
        )
    )

print("\nSkills making questions EASIER (positive effect):")
easier_skills = interp_summary_sorted[interp_summary_sorted["ame_mean"] > 0]
print(
    "\n{:<25} {:>8} {:>12} {:>25}".format(
        "Skill", "Odds Ratio", "Prob Change", "94% HDI"
    )
)
print("-" * 75)
for idx, row in easier_skills.iterrows():
    print(
        "{:<25} {:>8.2f} {:>11.1f}pp [{:>6.1f}pp to {:>6.1f}pp]".format(
            row["skill_name"],
            row["odds_ratio"],
            row["prob_change"],
            row["prob_change_lower"],
            row["prob_change_upper"],
        )
    )

print("\n" + "=" * 70)
print("INTERPRETATION GUIDE:")
print("=" * 70)
print(
    "  AME (Average Marginal Effect): Average change in probability when skill is present"
)
print("  - Computed across all actual observations in the data")
print("  - Accounts for context (other skills, category, model, dataset)")
print("  - Weighted by empirical distribution")
print("  Negative AME: Skill makes questions harder (reduces accuracy)")
print("  Positive AME: Skill makes questions easier (increases accuracy)")
print(
    "  Example: -5pp means having this skill reduces accuracy by 5 percentage points on average"
)
print("  HDI: 94% Highest Density Interval (Bayesian credible interval)")
print("  Odds Ratio: Included for reference (exp of log-odds coefficient)")

# Save results
interp_summary_sorted.to_csv(
    DATA_DIR / "bayesian_skill_effects_interpretable.csv", index=False
)
print(f"\nSaved: {DATA_DIR / 'bayesian_skill_effects_interpretable.csv'}")

# %%
# STEP 7: Visualize skill effects
# Create a forest plot showing which skills make questions harder or easier

fig, ax = plt.subplots(figsize=(10, 8))

# Sort by probability change (makes it more interpretable)
plot_data = interp_summary_sorted.copy()
y_pos = np.arange(len(plot_data))

# Calculate error bars for probability change
lower_errors = plot_data["prob_change"] - plot_data["prob_change_lower"]
upper_errors = plot_data["prob_change_upper"] - plot_data["prob_change"]

# Color code: negative (harder) = red, positive (easier) = green
colors = ["#e74c3c" if x < 0 else "#2ecc71" for x in plot_data["prob_change"]]

# Plot error bars
for i, (idx, row) in enumerate(plot_data.iterrows()):
    ax.errorbar(
        row["prob_change"],
        i,
        xerr=[[lower_errors.iloc[i]], [upper_errors.iloc[i]]],
        fmt="o",
        color=colors[i],
        markersize=9,
        capsize=5,
        capthick=2,
        elinewidth=2.5,
        markeredgewidth=0,
        alpha=0.8,
    )

# Add vertical line at 0
ax.axvline(
    x=0, color="black", linestyle="--", linewidth=1.5, alpha=0.7, label="No effect"
)

ax.set_yticks(y_pos)
ax.set_yticklabels(plot_data["skill_name"])
ax.set_xlabel(
    "Average Marginal Effect (percentage points)", fontsize=12, fontweight="bold"
)
ax.set_ylabel("Skill", fontsize=12, fontweight="bold")
ax.set_title(
    "Skill Effects on Question Difficulty (Average Marginal Effects)\n"
    "Accounting for other skills, category, model, and dataset",
    fontsize=13,
    fontweight="bold",
    pad=15,
)

ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.5)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "skill_effects_interpretable.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved skills visualisation")

# %%
# STEP 8: Check for multicollinearity
# Verify that skills are relatively independent by checking correlations
# High correlations would make it difficult to estimate independent effects

print("\n" + "=" * 70)
print("STEP 8: CHECKING SKILL CORRELATIONS")
print("=" * 70)

# Calculate correlation matrix for skill indicators on UNIQUE QUESTIONS only
unique_questions = question_skill_df.drop_duplicates(subset=["id", "dataset"])
print(f"Calculating correlations on {len(unique_questions)} unique questions")
skill_corr = unique_questions[skill_cols].corr()

# Show summary statistics of correlations
off_diagonal_corrs = []
for i, col1 in enumerate(skill_cols):
    for j, col2 in enumerate(skill_cols):
        if i < j:
            off_diagonal_corrs.append(abs(skill_corr.loc[col1, col2]))

if off_diagonal_corrs:
    print("\nCorrelation summary (absolute values, off-diagonal only):")
    print(f"  Maximum: {max(off_diagonal_corrs):.3f}")
    print(f"  Mean: {sum(off_diagonal_corrs) / len(off_diagonal_corrs):.3f}")
    print(f"  Median: {sorted(off_diagonal_corrs)[len(off_diagonal_corrs) // 2]:.3f}")

# Find pairs of skills with high correlations
print("\nSearching for highly correlated skill pairs...")
high_corr_pairs = []
for i, col1 in enumerate(skill_cols):
    for j, col2 in enumerate(skill_cols):
        if i < j:
            corr = skill_corr.loc[col1, col2]
            if abs(corr) > 0.5:
                skill1 = (
                    col1.replace("has_", "")
                    .replace("_", " ")
                    .replace("and", "&")
                    .title()
                )
                skill2 = (
                    col2.replace("has_", "")
                    .replace("_", " ")
                    .replace("and", "&")
                    .title()
                )
                high_corr_pairs.append((skill1, skill2, corr))

if high_corr_pairs:
    print(f"\nFound {len(high_corr_pairs)} skill pair(s) with |correlation| > 0.5:")
    print(f"\n{'Skill 1':<30} {'Skill 2':<30} {'Correlation':>12}")
    print("-" * 75)
    for skill1, skill2, corr in sorted(
        high_corr_pairs, key=lambda x: abs(x[2]), reverse=True
    ):
        print(f"{skill1:<30} {skill2:<30} {corr:>12.3f}")
    print("\nNote: High correlations suggest these skills frequently co-occur,")
    print("which may make it difficult to estimate their independent effects.")
else:
    print("\nNo strong correlations detected (all |r| <= 0.5)")
    print("Skills appear to be relatively independent.")

# %%
