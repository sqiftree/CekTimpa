"""
Pipeline Utama Detektor Copy-Move Berbasis Blok & Keypoint Konvensional (Non-Deep Learning)
Mendukung Penyaringan Area Polos (Langit), Penekanan Garis 1D, dan Mode Keypoint ORB/SIFT
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import time
import cv2
import numpy as np

from ..preprocessing import to_grayscale, apply_prefilter
from ..blocks import extract_blocks
from ..features import (
    extract_feature_a,
    extract_feature_b,
    extract_feature_c,
    extract_feature_d,
    extract_feature_h,
)
from ..matching import match_features
from ..postprocess import build_detection_mask, extract_boundary
from ..utils import create_overlay


@dataclass
class DetectionResult:
    """Struktur data hasil deteksi Copy-Move."""
    mask: np.ndarray
    boundary: np.ndarray
    overlay: np.ndarray
    pairs: List[Tuple[int, int]]
    shifts: np.ndarray
    is_forged: bool
    timings: Dict[str, float]

    @property
    def total_matches(self) -> int:
        return len(self.pairs)


def _detect_keypoints_conventional(
    gray_img: np.ndarray,
    block_size: int = 16,
    d_min: float = 30.0,
    threshold_t: int = 8,
    shift_tol: int = 4
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    """
    Deteksi Copy-Move berbasis titik kunci (Keypoint SIFT/ORB) murni matematika konvensional.
    Sangat efektif untuk mendeteksi objek berbentuk kompleks/tidak beraturan seperti manusia,
    kendaraan, atau objek yang mengalami sedikit perubahan skala/rotasi.
    """
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=4000, contrastThreshold=0.015, edgeThreshold=8)
        norm_type = cv2.NORM_L2
        ratio_thresh = 0.78
    else:
        detector = cv2.ORB_create(nfeatures=4000, scaleFactor=1.15, nlevels=8)
        norm_type = cv2.NORM_HAMMING
        ratio_thresh = 0.80

    keypoints, descriptors = detector.detectAndCompute(gray_img, None)
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
    knn_matches = matcher.knnMatch(descriptors, descriptors, k=5)

    d_min_sq = float(d_min) ** 2
    candidate_pairs = []
    shift_votes = {}

    for m_list in knn_matches:
        # Lewati m_list[0] karena merupakan dirinya sendiri (jarak = 0)
        valid_others = [m for m in m_list[1:] if (
            (coords[m.queryIdx, 0] - coords[m.trainIdx, 0]) ** 2 +
            (coords[m.queryIdx, 1] - coords[m.trainIdx, 1]) ** 2 >= d_min_sq
        )]
        if len(valid_others) < 2:
            continue

        best_m, second_m = valid_others[0], valid_others[1]
        if best_m.distance <= ratio_thresh * (second_m.distance + 1e-6):
            i, j = int(best_m.queryIdx), int(best_m.trainIdx)
            idx_a, idx_b = (i, j) if i < j else (j, i)
            sy = int(coords[idx_a, 0] - coords[idx_b, 0])
            sx = int(coords[idx_a, 1] - coords[idx_b, 1])
            if sy < 0 or (sy == 0 and sx < 0):
                sy, sx = -sy, -sx

            tol = max(int(shift_tol), 2)
            shift_key = (int(round(sy / tol) * tol), int(round(sx / tol) * tol))
            candidate_pairs.append((idx_a, idx_b, shift_key))
            shift_votes[shift_key] = shift_votes.get(shift_key, 0) + 1

    eff_t = max(4, int(threshold_t) // 3)
    winning_keys = {k for k, v in shift_votes.items() if v >= eff_t}
    matched_pairs = list(dict.fromkeys([
        (i, j) for i, j, k in candidate_pairs if k in winning_keys
    ]))
    winning_shifts = (
        np.array(sorted(winning_keys), dtype=np.int32)
        if winning_keys else np.empty((0, 2), dtype=np.int32)
    )
    return matched_pairs, winning_shifts, coords


def detect_copymove_block(
    image: np.ndarray,
    config: Dict[str, Any] = None
) -> DetectionResult:
    """
    Menjalankan pipeline deteksi Copy-Move berbasis Blok atau Keypoint Konvensional.
    """
    if config is None:
        config = {}

    feature_type = str(config.get("feature", "A")).upper()
    block_size = int(config.get("block_size", 16))
    stride = int(config.get("stride", 2))
    d_min = float(config.get("d_min", 30.0))
    threshold_t = int(config.get("threshold_t", 25))
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

    t0 = time.perf_counter()

    # 1. Pra-pemrosesan & Konversi Grayscale
    gray = to_grayscale(image)
    filtered_gray = apply_prefilter(gray, prefilter)
    t1 = time.perf_counter()

    # Mode Khusus Keypoint Konvensional (SIFT/ORB) untuk Objek Manusia/Kompleks
    if feature_type == "KP":
        t_b0 = time.perf_counter()
        pairs, shifts, coords = _detect_keypoints_conventional(
            filtered_gray,
            block_size=block_size,
            d_min=d_min,
            threshold_t=threshold_t,
            shift_tol=max(shift_tol, 4)
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
            timings=timings
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
    if feature_type == "A":
        features = extract_feature_a(blocks)
    elif feature_type == "B":
        try:
            features = extract_feature_b(filtered_gray, coords, block_size=block_size)
        except TypeError:
            features = extract_feature_b(blocks)
    elif feature_type == "C":
        features = extract_feature_c(blocks)
    elif feature_type == "D":
        features = extract_feature_d(blocks)
    elif feature_type == "H":
        features = extract_feature_h(blocks)
    else:
        raise ValueError(f"Tipe fitur '{feature_type}' tidak dikenali.")
    t3 = time.perf_counter()

    # 4. Pencocokan Fitur & Voting Shift Vector
    pairs, shifts = match_features(
        features=features,
        coords=coords,
        q_step=q_step,
        eps=eps,
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
        timings=timings
    )
