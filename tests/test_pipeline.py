"""
Unit test untuk integrasi pipeline deteksi copy-move dan modul-modulnya.
"""

import numpy as np
import pytest

from src.blocks import extract_blocks
from src.matching import match_features
from src.detectors.copymove_block import detect_copymove_block


def test_extract_blocks_basic():
    img = np.zeros((64, 64), dtype=np.uint8)
    img[16:48, 16:48] = 200
    blocks, coords = extract_blocks(img, block_size=16, stride=8, min_std=0.0, min_grad_2d=0.0)
    assert len(blocks) == len(coords)
    assert len(blocks) > 0


def test_extract_blocks_filtering():
    # Plain flat image
    img = np.full((64, 64), 100, dtype=np.uint8)
    blocks, coords = extract_blocks(img, block_size=16, stride=8, min_std=5.0)
    # Should filter all flat blocks
    assert len(blocks) == 0


def test_detect_copymove_synthetic():
    arr = np.full((128, 128, 3), 235, dtype=np.uint8)
    yy, xx = np.meshgrid(np.arange(24), np.arange(24), indexing="ij")
    pattern = ((np.sin(yy / 2.0) * np.cos(xx / 2.0) + 1.0) * 80).astype(np.uint8)
    patch = np.stack([30 + pattern // 2, 80 + pattern, np.full_like(pattern, 220)], axis=-1)

    # Place at (10, 10) and (60, 60)
    arr[10:34, 10:34] = patch
    arr[60:84, 60:84] = patch

    config = {
        "feature": "A",
        "block_size": 8,
        "stride": 2,
        "d_min": 20.0,
        "threshold_t": 5,
        "q_step": 1.0,
        "eps": 4.0,
        "min_std": 0.0,
        "min_grad_2d": 0.0,
        "match_method": "kdtree"
    }
    result = detect_copymove_block(arr, config)
    assert result is not None
    assert hasattr(result, "is_forged")
    assert hasattr(result, "total_matches")
