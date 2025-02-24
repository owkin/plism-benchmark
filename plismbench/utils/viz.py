"""Visualization of robustness results across different extractors."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_style("darkgrid")
pd.set_option("future.no_silent_downcasting", True)

# Please leave those 2 dictionnaries as is.
EXTRACTOR_LABELS_DICT = {
    "conch": "CONCH",
    "gpfm": "GPFM",
    "hibou_vit_base": "Hibou Base",
    "hibou_vit_large": "Hibou Large",
    "h0_mini": "H0-Mini",
    "hoptimus0": "H-Optimus-0",
    "kaiko_vit_base_8": "Kaiko ViT-B/8",
    "kaiko_vit_large_14": "Kaiko ViT-L/14",
    "phikon": "Phikon",
    "phikon_v2": "Phikon v2",
    "plip": "PLIP",
    "provgigapath": "Prov-GigaPath",
    "uni": "UNI",
    "uni2h": "UNI2-h",
    "virchow": "Virchow",
    "virchow2": "Virchow2",
}
EXTRACTOR_PARAMETERS_DICT = {
    "conch": 86_000_000,
    "gpfm": 307_000_000,
    "hibou_vit_base": 86_000_000,
    "hibou_vit_large": 307_000_000,
    "h0_mini": 86_000_000,
    "hoptimus0": 1_100_000_000,
    "kaiko_vit_base_8": 86_000_000,
    "kaiko_vit_large_14": 307_000_000,
    "phikon": 86_000_000,
    "phikon_v2": 307_000_000,
    "plip": 86_000_000,
    "provgigapath": 1_100_000_000,
    "uni": 307_000_000,
    "uni2h": 681_000_000,
    "virchow": 632_000_000,
    "virchow2": 632_000_000,
}


def expand_columns(raw_results: pd.DataFrame) -> pd.DataFrame:
    """Expand columns so as to have one column per metric and robustness type."""
    output = []
    # Robustness types are "all", "inter-scanner", "inter-staining",
    # "inter-scanner, inter-staining"
    for robustness_type in raw_results["robustness_type"].unique():
        subset = raw_results[
            raw_results["robustness_type"] == robustness_type
        ].sort_values("extractor")
        subset = subset.set_index("extractor").iloc[:, 1:]
        subset.columns = [f"{c}__{robustness_type}" for c in subset.columns]
        output.append(subset)
    output_df = pd.concat(output, axis=1)
    output_df.insert(
        0, "extractor", output_df.index.to_series().replace(EXTRACTOR_LABELS_DICT)
    )
    output_df.insert(
        1, "Parameters", output_df.index.to_series().replace(EXTRACTOR_PARAMETERS_DICT)
    )
    return output_df


def display_plism_metrics(
    raw_results: pd.DataFrame,
    metric_x: str = "cosine_similarity_median",
    metric_y: str = "top_1_accuracy_median",
    robustness_x: str = "all",
    robustness_y: str = "all",
    label_x: str = "Median Cosine Similarity",
    label_y: str = "Median Top-1 Accuracy",
    fig_save_path: str | Path | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    palette: Any | None = None,
):
    """Display PLISM robustness metrics.

    Parameters
    ----------
    raw_results: pd.DataFrame
        Raw results as computed by ``plismbench.utils.metrics.format_results``.
    metric_x: str = "cosine_similarity_median"
        Metric to display for x-axis. Should be of type 'metric_aggregation'.
        Supported metrics depends on the columns of ``raw_results`` but are
        by "cosine_similarity", "top_1_accuracy", "top_3_accuracy",
        "top_5_accuracy" and "top_10_accuracy". Supported aggregation types
        are either "mean" or "median".
    metric_y: str = "top_1_accuracy_median"
        Metric to display for y-axis.
    robustness_x: str = "all"
        Type of robustness for ``metric_x``.
        Supported types are "all", "inter-scanner", "inter-staining",
        "inter-scanner" and "inter-staining".
    robustness_y: str = "all"
        Type of robustness for ``metric_y``.
    label_x: str = "Median Cosine Similarity"
        Label for x-axis (can be anything).
    label_y: str = "Median Top-1 Accuracy"
        Label for y-axis (can be anything).
    xlim: tuple[float, float] | None = None
        Limits for x-axis.
    ylim: tuple[float, float] | None = None
        Limits for y-axis.
    palette = None
        Color palette.
    fig_save_path: str | Path | None = None
        Figure save path.
    """
    # Set figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)

    # Default limits for axes
    xlim = (0, 1) if xlim is None else xlim
    ylim = (0, 1) if xlim is None else ylim

    # Define x and y column in ``results_df``
    col_x = f"{metric_x}__{robustness_x}"
    col_y = f"{metric_y}__{robustness_y}"
    results_df = expand_columns(raw_results)
    results_df = results_df[["extractor", "Parameters", col_x, col_y]]
    results_df[[col_x, col_y]] = results_df[[col_x, col_y]].astype(float)

    # Default color palette
    if palette is None:
        palette = sns.color_palette("tab20")[: results_df.shape[0]]

    # Display metrics for each extractor
    sns.scatterplot(
        data=results_df,
        x=col_x,
        y=col_y,
        hue="extractor",
        size="Parameters",
        sizes=(50, 2000),
        palette=palette,
        edgecolor="black",
        alpha=0.9,
        legend=True,
        ax=ax,
    )

    # Display number of parameters for each extractor
    sns.scatterplot(
        data=results_df,
        x=col_x,
        y=col_y,
        s=5,
        color="black",
        marker="+",
        facecolor="black",
        alpha=0.7,
        legend=True,
        ax=ax,
    )

    # Set labels and limits
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Please leave these values as-is
    for _, row in results_df.iterrows():
        plt.text(
            row[col_x] + 0.001,
            row[col_y] + 0.001,
            row["extractor"],
            fontsize=10,
            ha="left",
            va="bottom",
            fontweight="bold",
        )

    # Please leave these values as-is: adding circles in the legend
    # proportionnal to the model size.
    scatter_handles = [
        plt.scatter(
            [], [], s=50, edgecolor="black", color="gray", alpha=0.5, label="     22M"
        ),
        plt.scatter(
            [], [], s=155, edgecolor="black", color="gray", alpha=0.5, label="     86M"
        ),
        plt.scatter(
            [], [], s=555, edgecolor="black", color="gray", alpha=0.5, label="     307M"
        ),
        plt.scatter(
            [],
            [],
            s=1143,
            edgecolor="black",
            color="gray",
            alpha=0.5,
            label="     632M",
        ),
        plt.scatter(
            [],
            [],
            s=2000,
            edgecolor="black",
            color="gray",
            alpha=0.5,
            label="     1,100M",
        ),
    ]
    ax.legend(
        handles=scatter_handles,
        title="No. parameters",
        loc="center left",
        bbox_to_anchor=(1, 0.6),
        fontsize=12,
        labelspacing=0.2,
        handleheight=2,
    )
    # Export figure
    if fig_save_path is not None:
        fig.savefig(fig_save_path, dpi=300, bbox_inches="tight")
    plt.show()
