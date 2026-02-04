"""Plotting utilities and shared mappings for benchmark visualisations."""

# %%
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyfonts import load_font
from sklearn.utils import resample
from tqdm import tqdm

from vrb_benchmark import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Build provider/model mappings from CSV data
# ---------------------------------------------------------------------------
_provider_df = pd.read_csv(PROJECT_ROOT / "data" / "providers.csv")
_models_df = pd.read_csv(PROJECT_ROOT / "data" / "models.csv")

PROVIDER_MODEL_MAPPING: dict[str, list[str]] = {}
for _, _row in _models_df.iterrows():
    PROVIDER_MODEL_MAPPING.setdefault(_row["provider"], []).append(_row["model_id"])

PROVIDER_COLOR_MAPPING: dict[str, str] = {
    row["provider"]: row["color"] for _, row in _provider_df.iterrows()
}

MODEL_FULLNAME_MAPPING: dict[str, str] = {
    row["model_id"]: row["display_name"] for _, row in _models_df.iterrows()
}

# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
FONT_REGULAR = load_font(
    font_url="https://github.com/google/fonts/blob/main/ofl/poppins/Poppins-Regular.ttf?raw=true"
)
FONT_BOLD = load_font(
    font_url="https://github.com/google/fonts/blob/main/ofl/poppins/Poppins-Bold.ttf?raw=true"
)


# %%
def plot_benchmark_df(
    accuracies,
    category,
    benchmark_name,
    folder_name,
    category_in_title=True,
    save_fig=False,
    include_ci=False,
    error_bars_pct=None,
    add_accuracy_annotations=True,
    xticklabels_fontsize=13,
    filename_suffix="",
    add_legend=True,
    custom_legend_elements=None,
    custom_color_mapping=None,
    width_figure=30,
    height_figure=9,
    position_legend=(1.10, 1.08),
    size_legend=18,
    y_limit=None,
):
    # check that we have colors for all models
    for model in accuracies.index:
        if model not in MODEL_FULLNAME_MAPPING:
            print(f"Model {model} not found in MODEL_FULLNAME_MAPPING")

    # Set colors
    if custom_color_mapping is not None:
        colormapping = custom_color_mapping
    else:
        color_mapping = {}
        for provider, models in PROVIDER_MODEL_MAPPING.items():
            if PROVIDER_COLOR_MAPPING[provider] in color_mapping:
                color_mapping[PROVIDER_COLOR_MAPPING[provider]].extend(models)
            else:
                color_mapping[PROVIDER_COLOR_MAPPING[provider]] = models

        colormapping = color_mapping

    company_to_color_dict = PROVIDER_COLOR_MAPPING

    fig, ax = plt.subplots(figsize=(width_figure, height_figure))
    fig.tight_layout(pad=1.8)

    # Get the accuracy and bad format values, sorted by accuracy
    accuracy_values = accuracies[category].sort_values(ascending=False) * 100
    model_names_ordered = accuracy_values.index.tolist()

    # Plot the accuracy bars
    model_colors = {}
    for color, models in colormapping.items():
        for model in models:
            model_colors[model] = color

    bar_colors = [model_colors.get(model, "grey") for model in model_names_ordered]

    if include_ci:
        bar_plot = ax.bar(
            model_names_ordered,
            accuracy_values,
            label="Accuracy",
            color=bar_colors,
            yerr=error_bars_pct,
            capsize=3,
        )
        raise_score = 3
    else:
        bar_plot = ax.bar(
            model_names_ordered,
            accuracy_values,
            label="Accuracy",
            color=bar_colors,
        )
        raise_score = 0

    # Add accuracy value on top of bars
    if add_accuracy_annotations:
        for p in bar_plot:
            ax.annotate(
                f"{p.get_height():.0f}",
                (
                    p.get_x() + p.get_width() / 2.0,
                    p.get_height() + 3 + raise_score,
                ),
                ha="center",
                va="center",
                fontsize=22,
                color="black",
            )

    model_names_short = [
        MODEL_FULLNAME_MAPPING.get(model, model) for model in model_names_ordered
    ]

    # Set labels and title
    ax.yaxis.grid(True, alpha=0.5)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.axhline(y=100, color="white", linewidth=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Accuracy (%)", fontsize=28, font=FONT_BOLD)
    if category_in_title:
        title = f"{benchmark_name} Performance - {category}"
    else:
        title = f"{benchmark_name} Performance"

    ax.set_title(
        title,
        font=FONT_BOLD,
        fontsize=35,
    )
    if y_limit is None:
        y_limit = 100

    ax.set_ylim(0, y_limit)

    ax.set_yticks(range(0, y_limit + 1, 20))
    ax.set_yticklabels(
        np.arange(0, y_limit + 1, 20),
        font=FONT_REGULAR,
        fontsize=18,
    )
    ax.set_xticks(range(len(accuracy_values.index)))
    ax.set_xticklabels(
        model_names_short,
        rotation=35,
        font=FONT_REGULAR,
        fontsize=xticklabels_fontsize,
        ha="right",
    )

    if add_legend:
        if custom_legend_elements is None:
            legend_elements = [
                mlines.Line2D(
                    [],
                    [],
                    color=color,
                    marker="o",
                    linestyle="None",
                    markersize=13,
                    label=company,
                )
                for company, color in company_to_color_dict.items()
                if color in set(bar_colors)
            ]
        else:
            legend_elements = custom_legend_elements
        ax.legend(
            handles=legend_elements,
            title="",
            loc="upper right",
            bbox_to_anchor=position_legend,
            fontsize=size_legend,
            frameon=True,
        )
    plt.tight_layout()
    plt.show()

    if save_fig:
        outfile = (
            PROJECT_ROOT
            / "data"
            / "plots"
            / folder_name
            / f"{category}_benchmark_finale{filename_suffix}.svg"
        )
        print(outfile)
        outfile.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(outfile, format="svg", dpi=1200)

    return ax


# %%
def plot_scatter_x_vs_y(
    df_cdpk_send_scatter,
    x_axis="CDPK",
    y_axis="SEND",
    save_fig=False,
    add_regression_line=False,
    filename="CDPK_vs_SEND_performance_scatter.svg",
):
    unique_companies = df_cdpk_send_scatter["provider"].unique()
    colors = [
        PROVIDER_COLOR_MAPPING.get(company, "gray") for company in unique_companies
    ]

    color2company_dict = dict(zip(unique_companies, colors))

    fig = plt.figure(figsize=(11, 6))
    ax = fig.gca()

    sns.scatterplot(
        data=df_cdpk_send_scatter,
        x=f"Acc {x_axis}",
        y=f"Acc {y_axis}",
        hue="provider",
        palette=color2company_dict,
        ax=ax,
    )
    if add_regression_line:
        print(
            f"Correlation between {x_axis} and {y_axis}: "
            f"{df_cdpk_send_scatter[f'Acc {x_axis}'].corr(df_cdpk_send_scatter[f'Acc {y_axis}']):.3f}"
        )
        sns.regplot(
            data=df_cdpk_send_scatter,
            x=f"Acc {x_axis}",
            y=f"Acc {y_axis}",
            scatter=False,
            color="black",
            ax=ax,
            line_kws={"linestyle": "-", "linewidth": 1.5},
        )

    ax.set_title(f"{y_axis} vs {x_axis} Performance", font=FONT_BOLD, fontsize=20)
    ax.set_xlabel(f"{x_axis} Accuracy (%)", font=FONT_BOLD, fontsize=18)
    ax.set_ylabel(f"{y_axis} Accuracy (%)", font=FONT_BOLD, fontsize=18)

    x_min = 20
    x_max = 90
    y_min = x_min
    y_max = x_max
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.legend(title="", fontsize=14, loc="lower right", bbox_to_anchor=(1.3, 0.0))

    xb = ax.get_xbound()
    yb = ax.get_ybound()
    mn = min([xb[0], yb[0]])
    mx = max([xb[1], yb[1]])
    ax.plot([mn, mx], [mn, mx], linestyle="--", color="gray")

    ax.grid(True, which="both", linestyle="--", linewidth=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_fig:
        outfile = PROJECT_ROOT / "data" / "plots" / filename
        print(f"Saving figure to: {outfile}")
        outfile.parent.mkdir(exist_ok=True, parents=True)
        fig.savefig(outfile, format="svg", dpi=1200, bbox_inches="tight")

    plt.show()

    return fig


def prepare_table_for_latex(df, benchmark_name="CDPK"):
    print(f"Preparing table for latex with {df.shape[0]} models")
    df_table_latex = df.copy()
    df_table_latex = df_table_latex[
        ["display_name", f"Acc {benchmark_name}", "provider"]
    ]
    df_table_latex = df_table_latex.rename(
        columns={
            "display_name": "Model",
            f"Acc {benchmark_name}": "Accuracy",
            "provider": "Company",
        }
    )
    df_table_latex["Accuracy"] = df_table_latex["Accuracy"].apply(lambda x: f"{x:.2f}")

    row_cut = (
        df_table_latex.shape[0] // 2
        if df_table_latex.shape[0] % 2 == 0
        else (df_table_latex.shape[0] // 2) + 1
    )
    df_table_latex = pd.concat(
        [
            df_table_latex.iloc[:row_cut].reset_index(drop=True),
            df_table_latex.iloc[row_cut:].reset_index(drop=True),
        ],
        axis=1,
    )

    return df_table_latex


# Bootstrapping


def calculate_accuracy(df_sample, correct_answer_col, model_pred_col):
    """Calculates accuracy for a single model on a given data sample."""
    if df_sample.empty:
        return 0.0
    else:
        accuracy = (
            df_sample[df_sample[correct_answer_col] == df_sample[model_pred_col]].shape[
                0
            ]
            / df_sample.shape[0]
        )
        return accuracy


def bootstrap_model_accuracies(
    df, correct_answer_col, model_pred_cols, n_bootstraps=1000, random_state=None
):
    """
    Performs bootstrap analysis to estimate uncertainty in model accuracies.

    Args:
        df (pd.DataFrame): DataFrame with 'Correct answer' and model prediction columns.
        correct_answer_col (str): Name of the column with correct answers.
        model_pred_cols (list): List of names of model prediction columns.
        n_bootstraps (int): Number of bootstrap replicates. Default is 1000.
        random_state (int, optional): Seed for random number generator for reproducibility.

    Returns:
        dict: A dictionary where keys are model names and values are dictionaries
              containing 'original_accuracy', 'bootstrap_accuracies' (if not deleted),
              'mean_bootstrap_accuracy', 'std_error', and 'confidence_interval_95'.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_observations = len(df)
    bootstrap_results = {}

    # --- Calculate original accuracies ---
    for model_col in tqdm(model_pred_cols, desc="Original Accuracies"):
        original_acc = calculate_accuracy(df, correct_answer_col, model_col)
        bootstrap_results[model_col] = {
            "original_accuracy": original_acc,
            "bootstrap_accuracies": [],
        }

    # --- Perform bootstrap replicates ---
    for i in tqdm(range(n_bootstraps), desc="Bootstrapping"):
        df_bootstrap_sample = resample(
            df,
            replace=True,
            n_samples=n_observations,
            random_state=(
                np.random.randint(0, 100000)
                if random_state is None
                else random_state + i
            ),
        )

        for model_col in model_pred_cols:
            acc_bootstrap = calculate_accuracy(
                df_bootstrap_sample, correct_answer_col, model_col
            )
            bootstrap_results[model_col]["bootstrap_accuracies"].append(acc_bootstrap)

    # --- Calculate summary statistics from bootstrap results ---
    for model_col in tqdm(model_pred_cols, desc="Calculating Stats"):
        accuracies = np.array(bootstrap_results[model_col]["bootstrap_accuracies"])
        bootstrap_results[model_col]["mean_bootstrap_accuracy"] = np.mean(accuracies)

        lower_bound = np.percentile(accuracies, 2.5)
        upper_bound = np.percentile(accuracies, 97.5)
        bootstrap_results[model_col]["confidence_interval_95"] = (
            lower_bound,
            upper_bound,
        )

    # transform results into a DataFrame for easier handling
    bootstrap_results_df = pd.DataFrame.from_dict(
        {
            model_name: {
                "Original Accuracy": model_stats["original_accuracy"],
                f"Mean Bootstrap Accuracy ({n_bootstraps} bootstraps)": model_stats[
                    "mean_bootstrap_accuracy"
                ],
                "95% Confidence Interval": model_stats["confidence_interval_95"],
            }
            for model_name, model_stats in bootstrap_results.items()
        },
        orient="index",
    )
    bootstrap_results_df.reset_index(inplace=True)
    bootstrap_results_df.rename(columns={"index": "model_id"}, inplace=True)
    bootstrap_results_df["model_id"] = bootstrap_results_df["model_id"].apply(
        lambda x: x.replace("pred_", "")
    )

    return bootstrap_results_df, bootstrap_results
