"""
Pipeline End-to-End Deteksi Pemalsuan Copy-Move Berbasis Bilateral Filtering & DCT
================================================================================
Metode Rujukan: Kaplan, Aksakalli, Kilic, & Erer (BAJECE 2020)
"A Novel Approach for Copy-move Forgery Detection using Bilateral Filtering"
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from .base import BaseDetector, BaseDetectionResult
from ..preprocessing import (
    to_grayscale,
    compute_bilateral_multiscale_detail,
)
from ..blocks import extract_blocks
from ..features import extract_feature_dct_bilateral
from ..matching import match_features
from ..postprocess import build_detection_mask, extract_boundary
from ..utils import create_overlay, draw_shift_lines


@dataclass
class BlockDetectionResult(BaseDetectionResult):
    """
    Hasil deteksi copy-move berbasis blok DCT-Bilateral (Kaplan dkk., 2020).
    """
    pairs: List[Tuple[int, int]] = field(default_factory=list)
    shifts: np.ndarray = field(default_factory=lambda: np.empty((0, 2), dtype=np.int32))
    shift_overlay: Optional[np.ndarray] = None


# Alias untuk menjaga kompatibilitas ke belakang (backward compatibility)
DetectionResult = BlockDetectionResult


class BlockDetector(BaseDetector):
    """
    Detektor Copy-Move Berbasis Blok DCT + Bilateral Filter (Kaplan dkk., 2020).
    """

    def __init__(self, default_config: Optional[Dict[str, Any]] = None):
        self.default_config = default_config or {}

    def detect(self, image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> BlockDetectionResult:
        cfg = {**self.default_config, **(config or {})}
        return detect_copymove_block(image, cfg)


def detect_copymove_block(
    image: np.ndarray,
    config: Optional[Dict[str, Any]] = None,
) -> BlockDetectionResult:
    """
    Menjalankan alur deteksi pemalsuan copy-move menggunakan Bilateral Filtering
    dan Discrete Cosine Transform (DCT) berdasarkan metode Kaplan dkk. (BAJECE 2020):
    1. Preprocessing — konversi grayscale (luminance) & dekomposisi bilateral multiskala D^2(I)
    2. Ekstraksi Blok beririsan (overlapping blocks)
    3. Ekstraksi Fitur Hibrida DCT-Bilateral (DCT 2D * layer detail bilateral + zig-zag scan)
    4. Kuantisasi & Pencocokan Leksikografis
    5. Voting Vektor Pergeseran (Shift-Vector Density)
    6. Pasca-Pemrosesan Morfologi (Closing, Opening, Hole Filling, Boundary)

    Args:
        image: Citra masukan RGB (H, W, 3) atau 2D uint8
        config: Kamus parameter konfigurasi deteksi

    Returns:
        BlockDetectionResult berisi mask, boundary, overlay, shifts, dan statistik.
    """
    if image is None or image.size == 0:
        raise ValueError("Citra masukan tidak boleh kosong (None / size 0).")

    if config is None:
        config = {}

    block_size = int(config.get("block_size", 8))
    stride = int(config.get("stride", 2))
    d_min = float(config.get("d_min", 30.0))
    threshold_t = int(config.get("threshold_t", 25))
    area_min = int(config.get("area_min", 60))
    q_step = float(config.get("q_step", 16.0))
    eps = float(config.get("eps", 5.0))
    window = int(config.get("window", 10))
    p_ratio = float(config.get("p_ratio", 0.25))

    # Parameter Bilateral Filter (Kaplan dkk., 2020 Sec. IV-B)
    d_bf = int(config.get("bilateral_d", 3))
    sigma_s = float(config.get("bilateral_sigma_s", 1.0))
    sigma_r = float(config.get("bilateral_sigma_r", 25.5))

    timings: Dict[str, float] = {}
    t_start = time.time()

    # 1. Preprocessing (Grayscale + Bilateral Filter Multiscale Decomposition)
    t0 = time.time()
    gray = to_grayscale(image, method="luminance")
    detail = compute_bilateral_multiscale_detail(
        gray,
        d=d_bf,
        sigma_space=sigma_s,
        sigma_color=sigma_r,
    )
    timings["preprocessing_ms"] = (time.time() - t0) * 1000

    # 2. Ekstraksi Blok
    t0 = time.time()
    blocks, coords = extract_blocks(gray, block_size=block_size, stride=stride)
    detail_blocks, _ = extract_blocks(detail, block_size=block_size, stride=stride)

    # Penapisan blok homogen / polos (Fridrich dkk., 2003 & Kaplan dkk., 2020)
    min_var = float(config.get("min_var", 10.0))
    if min_var > 0 and len(blocks) > 0:
        block_vars = np.var(blocks, axis=(1, 2))
        valid_idx = block_vars >= min_var
        if np.sum(valid_idx) > 10:
            blocks = blocks[valid_idx]
            detail_blocks = detail_blocks[valid_idx]
            coords = coords[valid_idx]

    timings["blocks_ms"] = (time.time() - t0) * 1000

    # 3. Ekstraksi Fitur DCT-Bilateral
    t0 = time.time()
    features = extract_feature_dct_bilateral(
        blocks=blocks,
        detail_blocks=detail_blocks,
        p_ratio=p_ratio,
    )
    timings["features_ms"] = (time.time() - t0) * 1000

    # 4. Pencocokan & Voting Vektor
    t0 = time.time()
    matched_pairs, shifts = match_features(
        features=features,
        coords=coords,
        q_step=q_step,
        eps=eps,
        window=window,
        d_min=d_min,
        threshold_t=threshold_t,
    )
    timings["matching_ms"] = (time.time() - t0) * 1000

    # 5. Pasca-Pemrosesan Morfologi
    t0 = time.time()
    mask = build_detection_mask(
        image_shape=image.shape,
        matched_pairs=matched_pairs,
        coords=coords,
        block_size=block_size,
        area_min=area_min,
    )
    boundary = extract_boundary(mask)
    overlay = create_overlay(image, mask, boundary)
    shift_overlay = (
        draw_shift_lines(image, matched_pairs, coords, block_size)
        if len(matched_pairs) > 0
        else image.copy()
    )
    timings["postprocess_ms"] = (time.time() - t0) * 1000

    timings["total_ms"] = (time.time() - t_start) * 1000
    is_forged = bool(len(shifts) > 0 and np.any(mask > 0))

    metadata = {
        "method": "dct_bilateral",
        "block_size": block_size,
        "stride": stride,
        "n_blocks": len(blocks),
        "n_pairs": len(matched_pairs),
        "n_shifts": len(shifts),
        "threshold_t": threshold_t,
    }

    return BlockDetectionResult(
        mask=mask,
        boundary=boundary,
        overlay=overlay,
        pairs=matched_pairs,
        shifts=shifts,
        is_forged=is_forged,
        timings=timings,
        shift_overlay=shift_overlay,
        metadata=metadata,
    )
