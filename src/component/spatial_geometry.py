"""Phase 6 — Spatial Geometric Component Prior Estimator.

Computes topological probability distributions over vehicle components
based on normalized bounding box location (x, y), aspect ratio, and box area.
Provides mathematical spatial grounding for vehicle damage localization.
"""

from typing import Dict, List, Tuple
import numpy as np

from .ontology import VEHICLE_COMPONENTS


class SpatialComponentPrior:
    """Probabilistic spatial topology estimator for vehicle body panels."""

    def __init__(self):
        # Component spatial centroids (mu_x, mu_y) and spreads (sigma_x, sigma_y)
        # Based on vehicle canonical perspective statistics
        self.component_anchors: Dict[str, Dict[str, float]] = {
            "roof": {
                "mu_x": 0.50, "sigma_x": 0.22,
                "mu_y": 0.16, "sigma_y": 0.10,
                "preferred_ar": 1.6, "ar_weight": 0.15,
                "base_prior": 0.08,
            },
            "windshield_glass": {
                "mu_x": 0.50, "sigma_x": 0.20,
                "mu_y": 0.30, "sigma_y": 0.11,
                "preferred_ar": 1.5, "ar_weight": 0.10,
                "base_prior": 0.10,
            },
            "hood": {
                "mu_x": 0.50, "sigma_x": 0.22,
                "mu_y": 0.45, "sigma_y": 0.12,
                "preferred_ar": 1.4, "ar_weight": 0.10,
                "base_prior": 0.18,
            },
            "front_bumper": {
                "mu_x": 0.50, "sigma_x": 0.25,
                "mu_y": 0.80, "sigma_y": 0.12,
                "preferred_ar": 2.2, "ar_weight": 0.20,
                "base_prior": 0.18,
            },
            "rear_bumper": {
                "mu_x": 0.50, "sigma_x": 0.25,
                "mu_y": 0.80, "sigma_y": 0.12,
                "preferred_ar": 2.0, "ar_weight": 0.15,
                "base_prior": 0.18,
            },
            "door": {
                "mu_x": 0.48, "sigma_x": 0.25,
                "mu_y": 0.54, "sigma_y": 0.14,
                "preferred_ar": 0.9, "ar_weight": 0.10,
                "base_prior": 0.25,
            },
            "fender": {
                "mu_x": 0.26, "sigma_x": 0.18,
                "mu_y": 0.58, "sigma_y": 0.13,
                "preferred_ar": 0.8, "ar_weight": 0.10,
                "base_prior": 0.16,
            },
            "quarter_panel": {
                "mu_x": 0.74, "sigma_x": 0.18,
                "mu_y": 0.54, "sigma_y": 0.14,
                "preferred_ar": 1.1, "ar_weight": 0.10,
                "base_prior": 0.14,
            },
            "lamp": {
                "mu_x": 0.28, "sigma_x": 0.16,
                "mu_y": 0.60, "sigma_y": 0.12,
                "preferred_ar": 1.3, "ar_weight": 0.10,
                "base_prior": 0.12,
            },
            "wheel_tire": {
                "mu_x": 0.32, "sigma_x": 0.22,
                "mu_y": 0.80, "sigma_y": 0.12,
                "preferred_ar": 1.0, "ar_weight": 0.20,
                "base_prior": 0.10,
            },
        }

    def compute_prior_distribution(
        self,
        norm_xc: float,
        norm_yc: float,
        norm_w: float,
        norm_h: float,
    ) -> Dict[str, float]:
        """Compute prior probability distribution across 10 components.

        Args:
            norm_xc: Normalized x centroid [0, 1].
            norm_yc: Normalized y centroid [0, 1].
            norm_w: Normalized bounding box width [0, 1].
            norm_h: Normalized bounding box height [0, 1].

        Returns:
            Dictionary mapping component names to probabilities summing to 1.0.
        """
        box_ar = norm_w / max(1e-4, norm_h)
        # Symmetrize horizontal coordinate for lateral parts (e.g. left vs right fender)
        sym_xc = norm_xc if norm_xc <= 0.5 else (1.0 - norm_xc)

        scores = {}
        for comp, p in self.component_anchors.items():
            mu_x = p["mu_x"]
            mu_y = p["mu_y"]
            sigma_x = p["sigma_x"]
            sigma_y = p["sigma_y"]

            # For bilateral components (lamp, fender, quarter_panel), use symmetrical distance
            if comp in ["fender", "quarter_panel", "lamp"]:
                dist_x = min(abs(norm_xc - 0.25), abs(norm_xc - 0.75))
            else:
                dist_x = abs(norm_xc - mu_x)

            dist_y = abs(norm_yc - mu_y)

            # 2D Gaussian density
            spatial_loglik = -0.5 * ((dist_x / sigma_x) ** 2 + (dist_y / sigma_y) ** 2)

            # Aspect ratio compatibility
            ar_diff = abs(np.log(box_ar + 1e-3) - np.log(p["preferred_ar"] + 1e-3))
            ar_penalty = -p["ar_weight"] * ar_diff

            log_score = spatial_loglik + ar_penalty + np.log(p["base_prior"])
            scores[comp] = log_score

        # Softmax normalization with numerical stability
        max_log = max(scores.values())
        exp_scores = {k: np.exp(v - max_log) for k, v in scores.items()}
        total_exp = sum(exp_scores.values())

        probs = {k: float(v / total_exp) for k, v in exp_scores.items()}
        return probs

    def get_top_k(
        self,
        norm_xc: float,
        norm_yc: float,
        norm_w: float,
        norm_h: float,
        k: int = 3,
    ) -> List[Tuple[str, float]]:
        """Return top-K most likely components with confidence values."""
        probs = self.compute_prior_distribution(norm_xc, norm_yc, norm_w, norm_h)
        sorted_comps = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        return sorted_comps[:k]
