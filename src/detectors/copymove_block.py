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
from ..preprocessing import to_grayscale, apply_prefilter, compute_bilateral_multiscale_detail
from ..blocks import extract_blocks
from ..features import (
    extract_feature_a,
    extract_feature_b,
    extract_feature_c,
    extract_feature_d,
    extract_feature_h,
    extract_feature_dct,
    extract_feature_dct_bilateral,
    extract_feature_b_local,
    standardize_feature_vectors,
)
from ..matching import match_features
from ..geometric import verify_similarity_pairs
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


def _detect_keypoints_conventional(
    gray_img: np.ndarray,
    block_size: int = 16,
    d_min: float = 30.0,
    threshold_t: int = 8,
    shift_tol: int = 4,
    reprojection_error: float = 3.0,
    diagnostics: Dict[str, int] = None,
    ratio_threshold: float = None,
    n_features: int = 4000,
    contrast_threshold: float = .015
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    """
    Deteksi Copy-Move berbasis titik kunci (Keypoint SIFT/ORB) murni matematika konvensional.
    Pencocokan deskriptor disaring dengan ratio test dan RANSAC similarity.
    Mask hanya menandai lingkungan keypoint, bukan segmentasi objek lengkap.
    """
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=contrast_threshold, edgeThreshold=8)
        norm_type = cv2.NORM_L2
        ratio_thresh = 0.78
    else:
        detector = cv2.ORB_create(nfeatures=n_features, scaleFactor=1.15, nlevels=8)
        norm_type = cv2.NORM_HAMMING
        ratio_thresh = 0.80

    if ratio_threshold is not None:
        ratio_thresh = float(ratio_threshold)
    if not 0 < ratio_thresh < 1 or n_features < 1 or contrast_threshold <= 0:
        raise ValueError("invalid keypoint ratio/count/contrast")
    keypoints, descriptors = detector.detectAndCompute(gray_img, None)
    if diagnostics is not None:
        diagnostics.update(keypoints=len(keypoints), candidate_pairs=0,
                           verified_pairs=0, models=0)
    if descriptors is None or len(keypoints) < 4:
        return [], np.empty((0, 2), dtype=np.int32), np.empty((0, 2), dtype=np.int32)

    half_b = block_size // 2
    h, w = gray_img.shape[:2]
    coords = np.array([
        [
            int(np.clip(kp.pt[1] - half_b, 0, max(0, h - block_size))),
            int(np.clip(kp.pt[0] - half_b, 0, max(0, w - block_size)))
        ]
        for kp in keypoints
    ], dtype=np.int32)

    matcher = cv2.BFMatcher(norm_type)
    knn_matches = matcher.knnMatch(descriptors, descriptors, k=min(64, len(descriptors)))

    d_min_sq = float(d_min) ** 2
    candidate_pairs = []
    seen_pairs = set()

    for m_list in knn_matches:
        # Self-match dibuang berdasarkan indeks, bukan posisi dalam hasil.
        valid_others = [m for m in m_list if m.queryIdx != m.trainIdx and (
            (coords[m.queryIdx, 0] - coords[m.trainIdx, 0]) ** 2 +
            (coords[m.queryIdx, 1] - coords[m.trainIdx, 1]) ** 2 >= d_min_sq
        )]
        if len(valid_others) < 2:
            continue

        best_m, second_m = valid_others[0], valid_others[1]
        if second_m.distance > 1e-6 and best_m.distance < ratio_thresh * second_m.distance:
            i, j = int(best_m.queryIdx), int(best_m.trainIdx)
            idx_a, idx_b = (i, j) if i < j else (j, i)
            if (idx_a, idx_b) in seen_pairs:
                continue
            seen_pairs.add((idx_a, idx_b))
            candidate_pairs.append((idx_a, idx_b))

    points_xy = np.float32([kp.pt for kp in keypoints])
    matched_pairs, model_shifts = verify_similarity_pairs(
        candidate_pairs, points_xy, min_inliers=max(4, int(threshold_t)),
        reprojection_error=reprojection_error)
    if diagnostics is not None:
        diagnostics.update(candidate_pairs=len(candidate_pairs),
                           verified_pairs=len(matched_pairs), models=len(model_shifts))
    return matched_pairs, model_shifts, coords



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

    # 3. Ekstraksi Fitur Blok
    if feature_type == "DCT_BILATERAL":
        detail = compute_bilateral_multiscale_detail(
            filtered_gray, d=int(config.get("bilateral_d", 3)),
            sigma_space=float(config.get("bilateral_sigma_s", 1.0)),
            sigma_color=float(config.get("bilateral_sigma_r", 25.5)))
        # Sample exactly the coordinates surviving the main texture filters.
        detail_blocks = np.stack([detail[y:y + block_size, x:x + block_size]
                                  for y, x in coords])
        features = extract_feature_dct_bilateral(
            blocks, detail_blocks, p_ratio=float(config.get("p_ratio", .25)))
    elif feature_type == "DCT":
        features = extract_feature_dct(blocks, n_coeffs=int(config.get("n_coeffs", 16)),
                                       include_dc=bool(config.get("include_dc", False)))
    elif feature_type == "A":
        features = extract_feature_a(blocks, g_sub=int(config.get("g_sub", 4)))
    elif feature_type == "B":
        context = config.get("gaussian_context", "block")
        if context == "block":
            features = extract_feature_b_local(blocks, sigma=float(config.get("gaussian_sigma", 1.)),
                                               g_samples=int(config.get("g_samples", 4)))
        elif context == "image":
            features = extract_feature_b(filtered_gray, coords, block_size=block_size,
                sigma=float(config.get("gaussian_sigma", 1.)), g_samples=int(config.get("g_samples", 4)))
        else:
            raise ValueError("gaussian_context must be block or image")
    elif feature_type == "C":
        features = extract_feature_c(blocks, low_freq_l=int(config.get("low_freq_l", 4)),
                                     normalize=bool(config.get("fft_normalize", False)))
    elif feature_type == "D":
        features = extract_feature_d(blocks, radial_bins=int(config.get("radial_bins", block_size // 2)),
                                     normalize=bool(config.get("fft_normalize", False)))
    elif feature_type == "H":
        features = extract_feature_h(blocks)
    else:
        raise ValueError(f"Tipe fitur '{feature_type}' tidak dikenali.")
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
