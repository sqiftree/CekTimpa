"""SIFT/FLANN detector adapted from alternate (2a8d920).

Retains configurable keypoints, stage visualizations and scale-aware masks;
adds unique physical support and guarded multi-model geometry from main.
Similarity is the default; planar homography is an explicit experimental option.
"""

from __future__ import annotations

import cv2
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from .base import BaseDetector, BaseDetectionResult
from ..postprocess import extract_boundary
from ..utils import create_overlay
from ..geometric import verify_geometric_pairs
from .keypoint_visualization import _draw_self_matches
from .transform_description import _classify_transform
from .keypoint_mask import _build_sift_mask


# ─────────────────────────────────────────────────────────────────────────────
# Data Class Hasil Deteksi
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SIFTDetectionResult(BaseDetectionResult):
    """
    Hasil lengkap deteksi copy-move berbasis SIFT (Amerini dkk., 2011).
    Menyimpan visualisasi setiap langkah pipeline untuk keperluan edukasi dan forensik.
    """
    # ── Visualisasi Per Langkah (untuk UI explainer) ─────────────────────────
    step1_keypoints_img: Optional[np.ndarray] = None   # L2: Keypoints divisualisasi (rich keypoints)
    step2_candidates_img: Optional[np.ndarray] = None  # L3+L4a: Kandidat match setelah Lowe's Ratio Test
    step3_spatial_img: Optional[np.ndarray] = None     # L4b: Match setelah Spatial Distance Filter
    step4_ransac_img: Optional[np.ndarray] = None      # L5: Inlier match terverifikasi RANSAC

    # ── Statistik & Metadata SIFT ───────────────────────────────────────────
    n_keypoints: int = 0                  # Total keypoints terdeteksi (Langkah 2)
    n_candidates: int = 0                 # Kandidat matches (setelah ratio test)
    n_spatial: int = 0                    # Matches valid (setelah spatial filter)
    n_inliers: int = 0                    # Inlier RANSAC (matches terverifikasi)
    pairs: List[Tuple[int, int]] = field(default_factory=list)
    shifts: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int32))
    models: List[np.ndarray] = field(default_factory=list)
    homography: Optional[np.ndarray] = None  # Matriks transformasi H pertama (3×3)
    transform_type: str = "—"             # Deskripsi jenis transformasi terdeteksi


    @property
    def total_matches(self):
        return self.n_inliers


class SIFTDetector(BaseDetector):
    """
    Detektor Copy-Move Berbasis SIFT + FLANN + Iterative RANSAC (Amerini dkk., 2011).
    """

    def __init__(self, default_config: Optional[Dict[str, Any]] = None):
        self.default_config = default_config or {}

    def detect(self, image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> SIFTDetectionResult:
        cfg = {**self.default_config, **(config or {})}
        return detect_copymove_sift(image, cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Fungsi Bantuan Visualisasi
# ─────────────────────────────────────────────────────────────────────────────







# ─────────────────────────────────────────────────────────────────────────────
# Fungsi Utama Deteksi SIFT
# ─────────────────────────────────────────────────────────────────────────────

def detect_copymove_sift(image: np.ndarray, config: Dict[str, Any] = None) -> SIFTDetectionResult:
    """Detect self-correspondences; mask marks keypoint support, not full objects."""
    if image is None or image.size == 0 or image.dtype != np.uint8:
        raise ValueError("image must be a nonempty uint8 array")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be grayscale or RGB")
    cfg = config or {}
    ratio = float(cfg.get("ratio_threshold", .75))
    d_min = float(cfg.get("d_min", 30))
    minimum = int(cfg.get("min_inliers", 6))
    radius = int(cfg.get("kp_radius", 10))
    area_min = int(cfg.get("area_min", 100))
    model = cfg.get("geometry_model", "similarity")
    reprojection = float(cfg.get("ransac_threshold", 3))
    max_models = int(cfg.get("max_ransac_iter", 8))
    if not 0 < ratio < 1 or d_min < 0 or minimum < 4 or radius < 1 or area_min < 0:
        raise ValueError("invalid ratio, distance, inlier, radius or area threshold")
    if model not in {"similarity", "homography"} or reprojection <= 0 or max_models < 1:
        raise ValueError("invalid geometric verification parameters")
    start = time.perf_counter()
    timings = {}
    t = time.perf_counter()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    timings["L1_grayscale_ms"] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    sift = cv2.SIFT_create(
        nfeatures=int(cfg.get("n_features", 4000)),
        nOctaveLayers=int(cfg.get("n_octave_layers", 3)),
        contrastThreshold=float(cfg.get("contrast_threshold", .015)),
        edgeThreshold=float(cfg.get("edge_threshold", 8)),
        sigma=float(cfg.get("sigma", 1.6)))
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    timings["L2_sift_ms"] = (time.perf_counter() - t) * 1000
    result = SIFTDetectionResult(
        mask=np.zeros(gray.shape, np.uint8), boundary=np.zeros(gray.shape, np.uint8),
        overlay=image.copy(), is_forged=False, timings=timings, n_keypoints=len(keypoints),
        step1_keypoints_img=cv2.drawKeypoints(image, keypoints, None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS),
        step2_candidates_img=image.copy(), step3_spatial_img=image.copy(),
        step4_ransac_img=image.copy(), metadata=dict(geometry_model=model))
    if descriptors is not None and len(keypoints) >= 4:
        t = time.perf_counter()
        matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=int(cfg.get("flann_trees", 5))),
            dict(checks=int(cfg.get("flann_checks", 64))))
        raw = matcher.knnMatch(descriptors, descriptors, k=min(64, len(keypoints)))
        timings["L3_matching_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        candidates, spatial, seen = [], [], set()
        points = np.float32([kp.pt for kp in keypoints])
        for group in raw:
            # Ignore same-location SIFT orientations before choosing alternatives.
            indices = np.fromiter((m.trainIdx for m in group), dtype=np.int32)
            delta = points[indices] - points[group[0].queryIdx]
            eligible = np.einsum("ij,ij->i", delta, delta) >= max(d_min, 1.) ** 2
            others = [m for m, keep in zip(group, eligible) if keep and m.queryIdx != m.trainIdx]
            if len(others) < 2:
                continue
            best, second = others[:2]
            if second.distance <= 1e-6 or best.distance >= ratio * second.distance:
                continue
            pair = tuple(sorted((best.queryIdx, best.trainIdx)))
            if pair in seen:
                continue
            seen.add(pair)
            candidates.append(best)
            spatial.append(best)
        result.n_candidates = len(candidates)
        result.n_spatial = len(spatial)
        result.step2_candidates_img = _draw_self_matches(image, keypoints, candidates)
        result.step3_spatial_img = _draw_self_matches(image, keypoints, spatial)
        timings["L4_filter_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        pairs, shifts, models = verify_geometric_pairs(
            [(m.queryIdx, m.trainIdx) for m in spatial], points,
            minimum, reprojection, max_models, model)
        result.pairs, result.shifts, result.models = pairs, shifts, models
        result.n_inliers = len(pairs)
        result.homography = models[0] if models else None
        result.transform_type = _classify_transform(result.homography)
        inliers = [cv2.DMatch(i, j, 0) for i, j in pairs]
        result.step4_ransac_img = _draw_self_matches(image, keypoints, inliers)
        timings["L5_ransac_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        result.mask = _build_sift_mask(image.shape, keypoints, inliers, radius, area_min)
        result.boundary = extract_boundary(result.mask)
        result.overlay = create_overlay(image, result.mask, result.boundary)
        result.is_forged = bool(models and np.any(result.mask))
        timings["L6_mask_ms"] = (time.perf_counter() - t) * 1000
    timings["total_ms"] = (time.perf_counter() - start) * 1000
    result.metadata.update(min_inliers=minimum, kp_radius=radius, area_min=area_min,
                           ratio_threshold=ratio, d_min=d_min,
                           ransac_threshold=reprojection, n_models=len(result.models))
    return result
