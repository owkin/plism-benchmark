"""Aggregation of robustness metrics across different extractors."""

from pathlib import Path

import pandas as pd


pd.set_option("future.no_silent_downcasting", True)


def get_extractor_results(results_path: Path) -> pd.DataFrame:
    """Get robustness results for a given extractor."""
    extractor = results_path.parent.name
    results = pd.read_csv(results_path, index_col=0)
    results.insert(0, "extractor", extractor)
    results.insert(1, "robustness_type", results.index.values)
    return results


def get_results(metrics_root_dir: Path, n_tiles: int = 8139) -> pd.DataFrame:
    """Get robustness results for all extractors and a given number of tiles."""
    results_paths = list((metrics_root_dir / f"{n_tiles}_tiles").glob("*/results.csv"))
    results = pd.concat(
        [get_extractor_results(results_path) for results_path in results_paths]
    ).reset_index(drop=True)
    return results


def format_results(
    metrics_root_dir: Path,
    agg_type: str = "median",
    n_tiles: int = 8139,
    top_k: list[int] | None = None,
) -> pd.DataFrame:
    """Store metrics according to an aggregation type ("mean" or "median")."""
    if top_k is None:
        top_k = [1, 3, 5, 10]
    metric_names = ["cosine_similarity"] + [f"top_{k}_accuracy" for k in top_k]
    results = get_results(metrics_root_dir, n_tiles=n_tiles)
    metric_idx = 0 if agg_type == "mean" else 1
    agg_cols = ["_mean", "_std"] if agg_type == "mean" else ["_median", "_iqr"]
    output_results = results.map(
        lambda x: x.split(" ; ")[metric_idx] if ";" in x else x
    )
    for metric_name in metric_names:
        metric_agg_cols = [f"{metric_name}{agg_col}" for agg_col in agg_cols]
        output_results[metric_agg_cols] = output_results[metric_name].str.extract(
            r"([0-9.]+)\s?\(([^)]+)\)"
        )
    return output_results


def rank_results(
    results: pd.DataFrame,
    robustness_type: str = "all",
    metric_name: str = "top_1_accuracy_median",
) -> pd.DataFrame:
    """Rank results according to a robustness type and metric name."""
    output = pd.pivot(
        results, columns="robustness_type", index="extractor", values=metric_name
    )
    return output.sort_values(robustness_type, ascending=False)


def show_aggregate_results(
    metrics_root_dir: Path,
    n_tiles: int = 8139,
    metric_name: str = "top_1_accuracy",
    robustness_type: str = "all",
    agg_type: str = "median",
    top_k: list[int] | None = None,
):
    """Retrieve results from .csv and rank by a given metric."""
    if top_k is None:
        top_k = [1, 3, 5, 10]
    supported_metric_names = ["cosine_similarity"] + [
        f"top_{k}_accuracy" for k in top_k
    ]
    if metric_name not in supported_metric_names:
        raise ValueError(
            f"{metric_name} metric not supported. Supported: {supported_metric_names}."
        )
    if agg_type not in (supported_agg_types := ["mean", "median"]):
        raise ValueError(
            f"{agg_type} aggregation not supported. Supported: {supported_agg_types}."
        )
    if robustness_type not in (
        supported_robustness_types := [
            "all",
            "inter-scanner",
            "inter-scanner, inter-staining",
            "inter-staining",
        ]
    ):
        raise ValueError(
            f"{robustness_type} robustness type not supported. Supported: {supported_robustness_types}."
        )
    results = format_results(
        metrics_root_dir, n_tiles=n_tiles, agg_type=agg_type, top_k=top_k
    )
    ranked_results = rank_results(
        results,
        metric_name=f"{metric_name}_{agg_type}",
        robustness_type=robustness_type,
    )
    ranked_results.insert(0, "extractor", ranked_results.index.values)
    return ranked_results
