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
    """Add float columns with parsed metrics wrt an aggregation type ("mean" or "median")."""
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


def get_aggregated_results(
    results: pd.DataFrame,
    metric_name: str = "top_1_accuracy",
    robustness_type: str = "all",
    agg_type: str = "median",
    top_k: list[int] | None = None,
) -> pd.DataFrame:
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
    ranked_results = rank_results(
        results,
        metric_name=f"{metric_name}_{agg_type}",
        robustness_type=robustness_type,
    )
    ranked_results.insert(0, "extractor", ranked_results.index.values)
    return ranked_results


def get_leaderboard_results(
    metrics_root_dir: Path,
) -> pd.DataFrame:
    """Generate leaderboard results."""
    # Get all results
    raw_results = format_results(
        metrics_root_dir=metrics_root_dir, agg_type="median", n_tiles=8139, top_k=None
    )
    # Get aggregated results for each type of robustness for cosine similarity and top 10 accuracy
    cosine_sim_results = get_aggregated_results(
        results=raw_results, metric_name="cosine_similarity", agg_type="median"
    )
    top_10_acc_results = get_aggregated_results(
        results=raw_results, metric_name="top_10_accuracy", agg_type="median"
    )
    # Merge the 2 dataframes into one
    leaderboard_cols = [
        "all_cosine_similarity",
        "inter-scanner_top_10_accuracy",
        "inter-staining_top_10_accuracy",
        "inter-scanner, inter-staining_top_10_accuracy",
    ]
    leaderboard_cols_labels = [
        "Cosine similarity (all)",
        "Top-10 accuracy (cross-scanner)",
        "Top-10 accuracy (cross-staining)",
        "Top-10 accuracy (cross-scanner, cross-staining)",
    ]
    leaderboard_results = (
        cosine_sim_results.sort_index()
        .merge(
            top_10_acc_results.iloc[:, 1:],
            left_index=True,
            right_index=True,
            suffixes=("_cosine_similarity", "_top_10_accuracy"),
        )[leaderboard_cols]
        .astype(float)
    )
    leaderboard_results.columns = leaderboard_cols_labels  # type: ignore
    leaderboard_results.insert(
        4, "Leaderboard metric", leaderboard_results.mean(axis=1).round(3)
    )
    leaderboard_results = leaderboard_results.sort_values(
        "Leaderboard metric", ascending=False
    )
    leaderboard_results["Rank"] = [
        f"#{i}" for i in range(1, leaderboard_results.shape[0] + 1)
    ]
    return leaderboard_results
