"""Module for cosine similarity metric."""

from plismbench.metrics.base import BasePlismMetric


class CosineSimilarity(BasePlismMetric):
    """Cosine similarity metric."""

    def __init__(self, device: str, use_mixed_precision: bool = True):
        super().__init__(device, use_mixed_precision)

    def compute_metric(self, matrix_a, matrix_b):
        """Compute cosine similarity metric."""
        # Compute cosine simlilarity for each pair of tiles between features
        # matrix a and b.
        if matrix_a.shape[0] != matrix_b.shape[0]:
            raise ValueError(
                f"Number of tiles must match. Got {matrix_a.shape[0]} and {matrix_b.shape[0]}."
            )

        # Put matrix_a and matrix_b on the gpu if needed
        matrix_a = self.ncp.asarray(matrix_a)  # shape (n_tiles, n_features)
        matrix_b = self.ncp.asarray(matrix_b)  # shape (n_tiles, n_features)

        if self.use_mixed_precision:
            matrix_a = matrix_a.astype(self.ncp.float16)
            matrix_b = matrix_b.astype(self.ncp.float16)

        # Compute cosine similarity
        dot_product_ab = self.ncp.matmul(
            matrix_a, matrix_b.T
        )  # shape (n_tiles, n_tiles)
        norm_a = self.ncp.linalg.norm(
            matrix_a, axis=1, keepdims=True
        )  # shape (n_tiles, )
        norm_b = self.ncp.linalg.norm(
            matrix_b, axis=1, keepdims=True
        )  # shape (n_tiles, )

        cosine_ab = dot_product_ab / (norm_a * norm_b.T)  # shape (n_tiles, n_tiles)

        _mean_cosine_ab = self.ncp.diag(cosine_ab).mean()
        mean_cosine_ab = (
            float(_mean_cosine_ab.get())
            if self.device == "gpu"
            else float(_mean_cosine_ab)
        )

        return mean_cosine_ab
