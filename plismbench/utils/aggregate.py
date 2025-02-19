"""Get aggregated metrics."""

import numpy as np
import pandas as pd


def iqr(x: pd.Series) -> float:
    """Get interquartile range."""
    return np.quantile(x, 0.75) - np.quantile(x, 0.25)


def aggregate_metrics(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics accross all possible pairs."""
    agg_metrics = (
        dataframe.apply(lambda x: (np.mean(x), np.std(x), np.median(x), iqr(x)))
        .set_index(np.array(["mean", "std", "median", "iqr"]))
        .round(3)
    )
    return agg_metrics


def pad(x: pd.Series | pd.DataFrame) -> pd.Series:
    """Pad values to third digit."""
    return x.astype(str).str.pad(5, side="right", fillchar="0")


def format_results(results: pd.DataFrame) -> dict[str, str]:
    """Format metrics."""
    _mean_std = pad(results.loc["mean", :]) + " (" + pad(results.loc["std", :]) + ")"
    _median_iqr = (
        pad(results.loc["median", :]) + " (" + pad(results.loc["iqr", :]) + ")"
    )

    mean_std = _mean_std.to_dict()
    median_iqr = _median_iqr.to_dict()
    output = {}
    for key in mean_std.keys():
        output[key] = mean_std[key] + " | " + median_iqr[key]
    return output


def get_results(metrics: pd.DataFrame, top_k: list[int]) -> pd.DataFrame:
    """Get aggregated robustness results."""
    metric_names = ["cosine_similarity"] + [f"top_{k}_accuracy" for k in top_k]
    all_results = aggregate_metrics(metrics[metric_names])
    inter_scanner_results = aggregate_metrics(
        metrics.loc[metrics["staining_a"] == metrics["staining_b"], metric_names]
    )
    inter_staining_results = aggregate_metrics(
        metrics.loc[metrics["scanner_a"] == metrics["scanner_b"], metric_names]
    )
    inter_scanner_inter_staining_results = aggregate_metrics(
        metrics.loc[
            (metrics["scanner_a"] != metrics["scanner_b"])
            & (metrics["staining_a"] != metrics["staining_b"]),
            metric_names,
        ]
    )
    output_dict = {}
    for robustness_type, results in zip(
        ["inter-scanner", "inter-staining", "inter-scanner, inter-staining", "all"],
        [
            inter_scanner_results,
            inter_staining_results,
            inter_scanner_inter_staining_results,
            all_results,
        ],
    ):
        _output = format_results(results)
        output_dict[robustness_type] = _output
    return pd.DataFrame(output_dict).T
