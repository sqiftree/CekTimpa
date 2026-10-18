"""
Pipeline End-to-End Deteksi Copy-Move Berbasis Blok
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import numpy as np
import time

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
    mask: np.ndarray
    boundary: np.ndarray
    overlay: np.ndarray
    pairs: List[Tuple[int, int]]
    shifts: np.ndarray
    is_forged: bool  # True berarti 'Terindikasi copy-move', bukan vonis palsu
    timings: Dict[str, float]

def detect_copymove_block(
    image: np.ndarray,
    config: Dict[str, Any] = None
) -> DetectionResult:
    """
    Menjalankan alur lengkap deteksi copy-move berbasis blok:
    1. Preprocessing (Grayscale / Restorasi)
    2. Ekstraksi Blok
    3. Ekstraksi Fitur (H / A / B / C / D)
    4. Kuantisasi & Pencocokan Leksikografis
    5. Voting Vektor Pergeseran
    6. Pasca-Pemrosesan Morfologi (Closing, Opening, Hole Filling, Boundary)
    """
    if config is None:
        config = {}

    feat_type = config.get("feature", "A")
    block_size = config.get("block_size", 16)
    stride = config.get("stride", 2)
    d_min = config.get("d_min", 30.0)
    threshold_t = config.get("threshold_t", 50)
    area_min = config.get("area_min", 100)
    prefilter = config.get("prefilter", "none")
    q_step = config.get("q_step", 1.0)
    eps = config.get("eps", 2.0)
    window = config.get("window", 10)

    timings = {}
    t_start = time.time()

    # 1. Preprocessing
    t0 = time.time()
    gray = to_grayscale(image, method="intensity")
    if prefilter != "none":
        gray = apply_prefilter(gray, filter_type=prefilter)
    timings["preprocessing_ms"] = (time.time() - t0) * 1000

    # 2. Ekstraksi Blok
    t0 = time.time()
    blocks, coords = extract_blocks(gray, block_size=block_size, stride=stride)
    timings["blocks_ms"] = (time.time() - t0) * 1000

    # 3. Ekstraksi Fitur
    t0 = time.time()
    if feat_type == "A":
        features = extract_feature_a(blocks, g_sub=4)
    elif feat_type == "B":
        features = extract_feature_b(gray, coords, block_size=block_size)
    elif feat_type == "C":
        features = extract_feature_c(blocks, low_freq_l=4)
    elif feat_type == "D":
        features = extract_feature_d(blocks)
    elif feat_type == "H":
        features = extract_feature_h(blocks)
    else:
        features = extract_feature_a(blocks, g_sub=4)
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
        threshold_t=threshold_t
    )
    timings["matching_ms"] = (time.time() - t0) * 1000

    # 5. Pasca-Pemrosesan Morfologi
    t0 = time.time()
    mask = build_detection_mask(
        image_shape=image.shape,
        matched_pairs=matched_pairs,
        coords=coords,
        block_size=block_size,
        area_min=area_min
    )
    boundary = extract_boundary(mask)
    overlay = create_overlay(image, mask, boundary)
    timings["postprocess_ms"] = (time.time() - t0) * 1000

    timings["total_ms"] = (time.time() - t_start) * 1000
    is_forged = len(shifts) > 0 and np.any(mask > 0)

    return DetectionResult(
        mask=mask,
        boundary=boundary,
        overlay=overlay,
        pairs=matched_pairs,
        shifts=shifts,
        is_forged=is_forged,
        timings=timings
    )
