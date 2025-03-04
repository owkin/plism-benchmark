"""Module for retrieval metrics."""

import numpy as np

from plismbench.metrics.base import BasePlismMetric


class TopkAccuracy(BasePlismMetric):
    """Top-k accuracy."""

    def __init__(
        self,
        device: str,
        use_mixed_precision: bool = True,
        k: list[int] | None = None,
    ):
        super().__init__(device, use_mixed_precision)
        self.k = [1, 3, 5, 10] if k is None else k

    def compute_metric(self, matrix_a, matrix_b):
        """Compute top-k accuracy metric."""
        if matrix_a.shape[0] != matrix_b.shape[0]:
            raise ValueError(
                f"Number of tiles must match. Got {matrix_a.shape[0]} and {matrix_b.shape[0]}."
            )

        matrix_ab = np.concatenate([matrix_a, matrix_b], axis=0)

        n_tiles = matrix_ab.shape[0] // 2

        if self.use_mixed_precision:
            matrix_ab = matrix_ab.astype(np.float16)

        matrix_ab = self.ncp.asarray(matrix_ab)  # put concatenated matrix on the gpu
        # ``dot_product_ab`` is a block matrix of shape (2*n_tiles, 2*n_tiles)
        # [
        #   [<matrix_a, matrix_a>, <matrix_a, matrix_b>],
        #   [<matrix_b, matrix_a>, <matrix_b, matrix_b>]
        # ]
        dot_product_ab = self.ncp.matmul(
            matrix_ab, matrix_ab.T
        )  # shape (2*n_tiles, 2*n_tiles)
        norm_ab = self.ncp.linalg.norm(
            matrix_ab, axis=1, keepdims=True
        )  # shape (2*n_tiles, )
        cosine_ab = dot_product_ab / (
            norm_ab * norm_ab.T
        )  # shape (2*n_tiles, 2*n_tiles)

        # Compute top-k indices for each row of cosine_ab using argpartition.
        # We use argpartition to efficiently find the top-k elements (excluding self-matches)
        kmax = max(self.k)
        # ``top_kmax_indices_ab`` has shape (2*n_tiles, kmax), for instance
        # ``top_kmax_indices_ab[i, 0]`` represents the closest tile index ``ci`` accross
        # slide a and slide b to the tile at index ``i`` (row index), hence ``ci``
        # is spanning between 0 and 2*n_tiles but excludes the index ``i`` of the tile
        # itself
        top_kmax_indices_ab = self.ncp.argpartition(
            -cosine_ab, range(1, kmax + 1), axis=1
        )[:, 1 : kmax + 1]
        # Compute top-k accuracies by iterating over k values
        top_k_accuracies = []
        for k in self.k:
            top_k_indices_ab = top_kmax_indices_ab[:, :k]  # shape (2*n_tiles, k)
            top_k_indices_a = top_k_indices_ab[:n_tiles]  # shape (n_tiles, k)
            top_k_indices_b = top_k_indices_ab[n_tiles:]  # shape (n_tiles, k)

            top_k_accs = []
            for i, top_k_indices in enumerate([top_k_indices_a, top_k_indices_b]):
                # If ``i==0``, we look at the closest tiles of each tile of matrix a that
                # are present in matrix b, hence ``(n_tiles, 2 * n_tiles)``. See matrix
                # block decomposition above.
                other_slide_indices = (
                    self.ncp.arange(n_tiles, 2 * n_tiles)
                    if i == 0
                    else self.ncp.arange(0, n_tiles)
                )
                # We now count the number of times one of the top-k closest tiles to
                # tile ``i`` for slide a (resp. b) is the same tile but in slide b (resp. a)
                correct_matches = self.ncp.sum(
                    self.ncp.any(top_k_indices == other_slide_indices[:, None], axis=1)
                )
                _top_k_acc = correct_matches / n_tiles
                top_k_acc = (
                    float(_top_k_acc.get())
                    if self.device == "gpu"
                    else float(_top_k_acc)
                )
                top_k_accs.append(top_k_acc)

            # Average over the two directions
            top_k_accuracies.append(sum(top_k_accs) / 2)

        return np.array(top_k_accuracies)
