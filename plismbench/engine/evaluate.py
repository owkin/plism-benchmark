"""Compute robustness metrics: cosine similarity and top-k accuracies."""

import sys
from functools import partial
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
from loguru import logger
from p_tqdm import p_map
from rich import print as rprint
from tqdm import tqdm

from plismbench.utils.aggregate import get_results
from plismbench.utils.core import load_pickle, write_pickle
from plismbench.utils.evaluate import (
    get_tiles_subset_idx,
    load_features,
    prepare_pairs_dataframe,
)


# Leave those two variables as-is
STAININGS: list[str] = [
    "GIV",
    "GIVH",
    "GM",
    "GMH",
    "GV",
    "GVH",
    "HR",
    "HRH",
    "KR",
    "KRH",
    "LM",
    "LMH",
    "MY",
]

SCANNERS: list[str] = ["AT2", "GT450", "P", "S210", "S360", "S60", "SQ"]
NUM_SLIDES: int = 91
NUM_TILES_PER_SLIDE: int = 16_278
DEFAULT_NUM_TILES_PER_SLIDE_METRICS: int = NUM_TILES_PER_SLIDE // 2  # 8_139


def core_compute_metrics(
    matrix_ab: np.ndarray, k_list: list[int], device: str
) -> tuple[float, list[float]]:
    """Compute cosine similarity and top-k accuracies.

    Parameters
    ----------
    matrix_ab: np.ndarray
        Concatenated matrix of features.
    k_list: list[int]
        Values of k for top-k accuracies computation.
    device: str
        Either 'cpu' or 'gpu' to use ``numpy`` or ``cupy``, respectively.

    Returns
    -------
    metrics: tuple[float, list[float]]
       Cosine similarity and top-k accuracies.
    """
    ncp = cp if device == "gpu" else np
    if device == "gpu":
        mempool = ncp.get_default_memory_pool()
        pinned_mempool = ncp.get_default_pinned_memory_pool()

    # Compute cosine simlilarity for each pair of tiles between features
    # matrix a and b.
    n_tiles = matrix_ab.shape[0] // 2
    matrix_ab = ncp.asarray(matrix_ab)  # put concatenated matrix on the gpu
    # ``dot_product_ab`` is a block matrix of shape (2*n_tiles, 2*n_tiles)
    # [
    #   [<matrix_a, matrix_a>, <matrix_a, matrix_b>],
    #   [<matrix_b, matrix_a>, <matrix_b, matrix_b>]
    # ]
    dot_product_ab = ncp.matmul(matrix_ab, matrix_ab.T)  # shape (2*n_tiles, 2*n_tiles)
    norm_ab = ncp.linalg.norm(matrix_ab, axis=1, keepdims=True)  # shape (2*n_tiles, )
    cosine_ab = dot_product_ab / (norm_ab * norm_ab.T)  # shape (2*n_tiles, 2*n_tiles)
    _mean_cosine_ab = ncp.diag(cosine_ab[:n_tiles, n_tiles:]).mean()
    mean_cosine_ab = (
        float(_mean_cosine_ab.get()) if device == "gpu" else float(_mean_cosine_ab)
    )

    # Compute top-k indices for each row of cosine_ab using argpartition.
    # We use argpartition to efficiently find the top-k elements (excluding self-matches)
    kmax = max(k_list)
    # ``top_kmax_indices_ab`` has shape (2*n_tiles, kmax), for instance
    # ``top_kmax_indices_ab[i, 0]`` represents the closest tile index ``ci`` accross
    # slide a and slide b to the tile at index ``i`` (row index), hence ``ci``
    # is spanning between 0 and 2*n_tiles but excludes the index ``i`` of the tile
    # itself
    top_kmax_indices_ab = ncp.argpartition(-cosine_ab, range(1, kmax + 1), axis=1)[
        :, 1 : kmax + 1
    ]

    # Compute top-k accuracies by iterating over k values
    top_k_accuracies = []
    for k in k_list:
        top_k_indices_ab = top_kmax_indices_ab[:, :k]  # shape (2*n_tiles, kmax)
        top_k_indices_a = top_k_indices_ab[:n_tiles]  # shape (n_tiles, kmax)
        top_k_indices_b = top_k_indices_ab[n_tiles:]  # shape (n_tiles, kmax)

        top_k_accs = []
        for i, top_k_indices in enumerate([top_k_indices_a, top_k_indices_b]):
            # If ``i==0``, we look at the closest tiles of each tile of matrix a that
            # are present in matrix b, hence ``(n_tiles, 2 * n_tiles)``. See matrix
            # block decomposition above.
            other_slide_indices = (
                ncp.arange(n_tiles, 2 * n_tiles) if i == 0 else ncp.arange(0, n_tiles)
            )
            # We now count the number of times one of the top-k closest tiles to
            # tile ``i`` for slide a (resp. b) is the same tile but in slide b (resp. a)
            correct_matches = ncp.sum(
                ncp.any(top_k_indices == other_slide_indices[:, None], axis=1)
            )
            _top_k_acc = correct_matches / n_tiles
            top_k_acc = (
                float(_top_k_acc.get()) if device == "gpu" else float(_top_k_acc)
            )
            top_k_accs.append(top_k_acc)

        # Average over the two directions
        top_k_accuracies.append(sum(top_k_accs) / 2)

    if device == "gpu":
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    metrics = (mean_cosine_ab, top_k_accuracies)
    return metrics


def compute_metrics_ab(
    fp_a: Path,
    fp_b: Path,
    tiles_subset_idx: np.ndarray,
    top_k: list[int],
    device: str,
    pickles_save_dir: Path,
    overwrite: bool,
) -> list[float]:
    """Compute metrics between float16 features from slide a and slide b."""
    # Check if a pickle has already been dumped to disk to avoid computing
    # the metrics twice for a given slides pair.
    pickle_key = "---".join([fp_a.parent.name, fp_b.parent.name])
    if (pickle_path := pickles_save_dir / f"{pickle_key}.pkl").exists():
        if overwrite:
            pass
        else:
            try:
                return load_pickle(pickle_path)
            except Exception as exc:  # type: ignore
                logger.info(f"{str(pickle_path)} seems to be corrupted:\n{exc}.")

    matrix_a, matrix_b = (
        load_features(fp_a),
        load_features(fp_b),
    )
    # Coordinates should be equal for tiles location matching
    np.testing.assert_allclose(matrix_a[:, :3], matrix_b[:, :3])
    # Concanenate features from slide a and b to compute
    # top-k accuracies. Note: top-k accuracy is computed
    # over a subset of tiles.
    # Warning: convert matrix to float16 !
    matrix_ab = np.concatenate(
        [matrix_a[tiles_subset_idx, 3:], matrix_b[tiles_subset_idx, 3:]], axis=0
    ).astype(np.float16)
    cosine_similarity, top_k_accuracies = core_compute_metrics(
        matrix_ab, k_list=top_k, device=device
    )
    metrics_ab = [cosine_similarity, *top_k_accuracies]
    write_pickle(metrics_ab, pickle_path)
    return metrics_ab


def compute_metrics(
    features_root_dir: Path,
    metrics_save_dir: Path,
    extractor: str,
    top_k: list[int] | None = None,
    n_tiles: int | None = None,
    device: str = "gpu",
    workers: int = 4,
    overwrite: bool = False,
):
    """Compute robustness metrics and save it to disk.

    Parameters
    ----------
    features_root_dir: Path
        The root folder where features will be stored.
        The final export directory is ``features_root_dir / extractor``
    metrics_save_dir: Path
        Folder containing the output metrics.
        The final export directory is ``metrics_save_dir / extractor``.
    extractor: str
        The name of the feature extractor as defined in ``plismbench.models.__init__.py``
    top_k: list[int] | None = None
        Values of k for top-k accuracy computation.
    n_tiles: int | None = None
        Number of tiles per slide for metrics computation.
    device: str = "gpu"
        Device on which matrix operations will be performed.
    workers: int = 4
        Number of workers for cpu parallel computations if ``device='cpu'``.
    overwrite: bool = False
        Whether to overwrite existing metrics.
    """
    # Supported number of tiles correspond to
    # None: DEFAULT_NUM_TILES_PER_SLIDE_METRICS = 8_139
    # 460: corresponds to 10 tiles per TMA - meant for debugging purposes
    # 2_713: NUM_TILES_PER_SLIDE / 6
    # 5_426: NUM_TILES_PER_SLIDE / 3
    # 8_139: NUM_TILES_PER_SLIDE / 2
    # 16_278: NUM_TILES_PER_SLIDE

    if n_tiles not in (supported_n_tiles := [None, 460, 2_713, 5_426, 8_139, 16_278]):
        raise ValueError(
            f"n_tiles should take values in {supported_n_tiles}. Got {n_tiles}."
        )
    n_tiles = DEFAULT_NUM_TILES_PER_SLIDE_METRICS if n_tiles is None else n_tiles
    top_k = [1, 3, 5, 10] if top_k is None else top_k

    metrics_save_dir = metrics_save_dir / f"{n_tiles}_tiles" / extractor
    pickles_save_dir = metrics_save_dir / "pickles"
    metrics_export_path: Path = metrics_save_dir / "metrics.csv"
    if metrics_export_path.exists():
        if overwrite:
            logger.info("Metrics already exist. Overwriting...")
        else:
            logger.info("Metrics already exist. Skipping...")
            sys.exit()
    pickles_save_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Metrics will be saved at {str(metrics_export_path)}.")
    logger.info(f"Slide pairs pickles will be saved at {str(pickles_save_dir)}.")

    slide_pairs = prepare_pairs_dataframe(
        features_dir=features_root_dir, extractor=extractor
    )
    n_pairs = slide_pairs.shape[0]
    features_paths_pairs = slide_pairs[["features_path_a", "features_path_b"]].values
    tiles_subset_idx = get_tiles_subset_idx(n_tiles=n_tiles)
    logger.warning(
        f"Will compute metrics on {n_tiles} tiles per slide. "
        f"Top-k accuracies will be computed for k in {top_k}."
    )

    if device not in (supported_device := ["cpu", "gpu"]):
        raise ValueError(
            f"Device {device} not supported. Please choose among {supported_device}."
        )
    logger.warning(f"Metrics will be computed on {device}.")

    if device == "gpu":
        logger.info("Running on gpu: sequential computation over pairs.")
        pairs_metrics = []
        for fp_a, fp_b in tqdm(features_paths_pairs, total=n_pairs):
            metrics_ab = compute_metrics_ab(
                fp_a=fp_a,
                fp_b=fp_b,
                tiles_subset_idx=tiles_subset_idx,
                top_k=top_k,
                device=device,
                pickles_save_dir=pickles_save_dir,
                overwrite=overwrite,
            )
            pairs_metrics.append((fp_a, fp_b, *metrics_ab))
    else:
        logger.info("Running on cpu: parallel computation over pairs.")
        logger.warning(
            f"Number of workers: {workers}. Try reducing it if you have RAM issues."
        )
        _compute_metrics_ab = partial(
            compute_metrics_ab,
            tiles_subset_idx=tiles_subset_idx,
            top_k=top_k,
            device=device,
            pickles_save_dir=pickles_save_dir,
            overwrite=overwrite,
        )
        metrics = p_map(
            _compute_metrics_ab,
            features_paths_pairs[:, 0],
            features_paths_pairs[:, 1],
            num_cpus=workers,
        )
        pairs_metrics = [
            (fp_a, fp_b, *m) for ((fp_a, fp_b), m) in zip(features_paths_pairs, metrics)
        ]

    metrics = pd.DataFrame(
        pairs_metrics,
        columns=[
            "features_path_a",
            "features_path_b",
            "cosine_similarity",
        ]
        + [f"top_{k}_accuracy" for k in top_k],
    )
    output = slide_pairs.merge(
        metrics,
        how="inner",
        on=["features_path_a", "features_path_b"],
    )
    assert (n_rows := output.shape[0]) == n_pairs, (
        f"Output dataframe with metrics have n_rows: {n_rows} < {n_pairs}."
    )
    # Export metrics for all pairs
    output.to_csv(metrics_export_path, index=None)  # type: ignore
    robustness_results = get_results(metrics=output, top_k=top_k)
    # Get and export aggregated results
    results_export_path = metrics_save_dir / "results.csv"
    robustness_results.to_csv(results_export_path, index=True)  # type: ignore
    # Only display median (IQR)
    logger.info("Robustness results [median (iqr)]:")
    rprint(robustness_results.map(lambda x: x.split(" ; ")[1]))
    logger.success("Successfully computed and stored metrics.")
