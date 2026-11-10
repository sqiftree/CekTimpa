"""
Pipeline Utama Detektor Copy-Move Berbasis Blok & Keypoint Konvensional (Non-Deep Learning)
Mendukung Penyaringan Area Polos (Langit), Penekanan Garis 1D, dan Mode Keypoint ORB/SIFT
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
import time
import cv2
import numpy as np

from .base import BaseDetector, BaseDetectionResult
from ..preprocessing import to_grayscale, apply_prefilter
from ..blocks import extract_blocks
from ..features import standardize_feature_vectors
from .block_features import extract_block_features
from ..matching import match_features
from .keypoints import _detect_keypoints_conventional
from ..postprocess import build_detection_mask, extract_boundary
from ..utils import create_overlay


@dataclass
class DetectionResult(BaseDetectionResult):
    """Common result contract with block correspondences and diagnostics."""
    pairs: List[Tuple[int, int]] = field(default_factory=list)
    shifts: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int32))
    diagnostics: Dict[str, int] = field(default_factory=dict)

    @property
    def total_matches(self) -> int:
        return len(self.pairs)





def detect_copymove_block(
    image: np.ndarray,
    config: Dict[str, Any] = None
) -> DetectionResult:
    """
    Menjalankan pipeline deteksi Copy-Move berbasis Blok atau Keypoint Konvensional.
    """
    if image is None or image.size == 0 or image.dtype != np.uint8:
        raise ValueError("image must be a nonempty uint8 array")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be grayscale or RGB")
    if config is None:
        config = {}

    feature_type = str(config.get("feature", "A")).upper()
    block_size = int(config.get("block_size", 16))
    stride = int(config.get("stride", 2))
    d_min = float(config.get("d_min", 30.0))
    threshold_t = int(config.get("threshold_t", 6 if feature_type == "KP" else 25))
    area_min = int(config.get("area_min", 60))
    prefilter = str(config.get("prefilter", "none"))
    q_step = float(config.get("q_step", 1.0))
    eps = float(config.get("eps", 3.0))
    window = int(config.get("window", 10))

    # Parameter tambahan untuk menekan false positive langit/garis dan mendeteksi objek
    min_std = float(config.get("min_std", 0.0))
    min_grad_2d = float(config.get("min_grad_2d", 0.0))
    match_method = str(config.get("match_method", "kdtree"))
    shift_tol = int(config.get("shift_tol", 1))
    reject_collinear = bool(config.get("reject_collinear", False))

    if feature_type not in {"A", "B", "C", "D", "H", "DCT", "DCT_BILATERAL", "KP"}:
        raise ValueError(f"Tipe fitur '{feature_type}' tidak dikenali.")
    if block_size < 4 or stride < 1 or window < 1 or shift_tol < 1 or threshold_t < 1:
        raise ValueError("invalid block size, stride, window, shift tolerance or vote threshold")
    if not all(np.isfinite(v) for v in [d_min, q_step, eps, min_std, min_grad_2d]):
        raise ValueError("parameters must be finite")
    if q_step <= 0 or eps <= 0 or d_min < 0 or min_std < 0 or min_grad_2d < 0 or area_min < 0:
        raise ValueError("invalid distance, scale or area threshold")
    scaling = str(config.get("feature_scaling", "native"))
    if scaling not in {"native", "standardized"}:
        raise ValueError("feature_scaling must be native or standardized")
    if prefilter not in {"none", "gaussian", "median"}:
        raise ValueError("unknown prefilter")
    if feature_type == "A" and (int(config.get("g_sub", 4)) < 1 or block_size % int(config.get("g_sub", 4))):
        raise ValueError("subgrid must divide block size")
    metadata = {"feature": feature_type, "config": dict(config), "feature_scaling": scaling,
                "effective_q_step": 1.0 if scaling == "standardized" else q_step}

    t0 = time.perf_counter()

    # 1. Pra-pemrosesan & Konversi Grayscale
    gray = to_grayscale(image, method="luminance" if feature_type == "DCT_BILATERAL" else "intensity")
    filtered_gray = apply_prefilter(gray, prefilter)
    t1 = time.perf_counter()

    # Mode Khusus Keypoint Konvensional (SIFT/ORB) untuk Objek Manusia/Kompleks
    if feature_type == "KP":
        diagnostics = {}
        t_b0 = time.perf_counter()
        pairs, shifts, coords = _detect_keypoints_conventional(
            filtered_gray,
            block_size=block_size,
            d_min=d_min,
            threshold_t=threshold_t,
            shift_tol=max(shift_tol, 4),
            reprojection_error=float(config.get("ransac_error", 3.0)),
            diagnostics=diagnostics,
            ratio_threshold=config.get("ratio_threshold"),
            n_features=int(config.get("n_features", 4000)),
            contrast_threshold=float(config.get("contrast_threshold", .015))
        )
        t_b1 = time.perf_counter()

        mask = build_detection_mask(
            filtered_gray.shape,
            pairs,
            coords,
            block_size=block_size,
            area_min=area_min
        )
        boundary = extract_boundary(mask)
        overlay = create_overlay(image, mask, boundary)
        t_end = time.perf_counter()

        timings = {
            "preprocess_ms": (t1 - t0) * 1000.0,
            "blocks_ms": 0.0,
            "features_ms": (t_b1 - t_b0) * 1000.0,
            "matching_ms": 0.0,
            "postprocess_ms": (t_end - t_b1) * 1000.0,
            "total_ms": (t_end - t0) * 1000.0,
        }
        return DetectionResult(
            mask=mask,
            boundary=boundary,
            overlay=overlay,
            pairs=pairs,
            shifts=shifts,
            is_forged=bool(len(pairs) > 0 and np.any(mask > 0)),
            timings=timings,
            metadata=metadata,
            diagnostics=diagnostics
        )

    # 2. Ekstraksi Blok & Penyaringan Area Polos (Langit) / Garis 1D
    blocks, coords = extract_blocks(
        filtered_gray,
        block_size=block_size,
        stride=stride,
        min_std=min_std,
        min_grad_2d=min_grad_2d
    )
    t2 = time.perf_counter()

    if len(blocks) < 2:
        empty_mask = np.zeros(filtered_gray.shape, dtype=np.uint8)
        t_end = time.perf_counter()
        return DetectionResult(
            mask=empty_mask,
            boundary=empty_mask.copy(),
            overlay=image.copy(),
            pairs=[],
            shifts=np.empty((0, 2), dtype=np.int32),
            is_forged=False,
            metadata={**metadata, "n_blocks": len(blocks), "feature_dimensions": 0},
            timings={
                "preprocess_ms": (t1 - t0) * 1000.0,
                "blocks_ms": (t2 - t1) * 1000.0,
                "features_ms": 0.0,
                "matching_ms": 0.0,
                "postprocess_ms": 0.0,
                "total_ms": (t_end - t0) * 1000.0,
            }
        )

    # 3. Ekstraksi fitur melalui modul algoritma terpilih
    features = extract_block_features(
        feature_type, blocks, coords, filtered_gray, block_size, config)
    if scaling == "standardized":
        features = standardize_feature_vectors(features)
    metadata.update(n_blocks=len(blocks), feature_dimensions=features.shape[1])
    t3 = time.perf_counter()

    # 4. Pencocokan Fitur & Voting Shift Vector
    pairs, shifts = match_features(
        features=features,
        coords=coords,
        q_step=1.0 if scaling == "standardized" else q_step,
        # alternate defines bilateral epsilon in raw coefficient units;
        # main's matcher measures distances after division by Q.
        eps=eps / q_step if feature_type == "DCT_BILATERAL" and scaling == "native" else eps,
        window=window,
        d_min=d_min,
        threshold_t=threshold_t,
        method=match_method,
        shift_tol=shift_tol,
        reject_collinear=reject_collinear
    )
    t4 = time.perf_counter()

    # 5. Konstruksi Mask Biner, Boundary, dan Overlay
    mask = build_detection_mask(
        filtered_gray.shape,
        pairs,
        coords,
        block_size=block_size,
        area_min=area_min
    )
    boundary = extract_boundary(mask)
    overlay = create_overlay(image, mask, boundary)
    t5 = time.perf_counter()

    timings = {
        "preprocess_ms": (t1 - t0) * 1000.0,
        "blocks_ms": (t2 - t1) * 1000.0,
        "features_ms": (t3 - t2) * 1000.0,
        "matching_ms": (t4 - t3) * 1000.0,
        "postprocess_ms": (t5 - t4) * 1000.0,
        "total_ms": (t5 - t0) * 1000.0,
    }

    return DetectionResult(
        mask=mask,
        boundary=boundary,
        overlay=overlay,
        pairs=pairs,
        shifts=shifts,
        is_forged=bool(len(pairs) > 0 and np.any(mask > 0)),
        timings=timings,
        metadata=metadata
    )


# Adapter compatible with alternate's detector factory, preserving feature A-H.


class BlockDetector(BaseDetector):
    def __init__(self, default_config=None):
        self.default_config = default_config or {}

    def detect(self, image, config=None):
        return detect_copymove_block(image, {**self.default_config, **(config or {})})


class BilateralDetector(BlockDetector):
    def detect(self, image, config=None):
        cfg = {"block_size": 8, "q_step": 16., "eps": 5.,
               "min_std": 3., "match_method": "lexsort", **self.default_config, **(config or {}),
               "feature": "DCT_BILATERAL"}
        return detect_copymove_block(image, cfg)


BlockDetectionResult = DetectionResult
