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


def test_asymmetric_knn_pair_is_retained():
    # Point 2 finds point 0, although point 0's two closer neighbors hide point 2.
    features = np.array([[0.], [.1], [-.3], [.2]], dtype=np.float32)
    coords = np.array([[0, 0], [0, 100], [100, 0], [100, 100]])
    pairs, _ = match_features(features, coords, window=2, eps=1, d_min=1, threshold_t=1)
    assert (0, 2) in pairs
    assert len(pairs) == len(set(pairs))


def test_spatial_filter_expands_neighbor_search():
    features = np.array([[0.], [.01], [.02], [.03], [.04]], dtype=np.float32)
    coords = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [100, 100]])
    pairs, _ = match_features(features, coords, window=2, eps=1, d_min=30, threshold_t=1)
    assert (0, 4) in pairs
    # Reciprocal matches must not double the vote for an isolated pair.
    pairs, _ = match_features(features[:2], coords[[0, 4]], eps=1, d_min=30, threshold_t=2)
    assert pairs == []


def test_dct_brightness_invariance():
    from src.features import extract_feature_dct
    blocks = np.random.default_rng(42).uniform(40, 180, (3, 8, 8)).astype(np.float32)
    assert np.allclose(extract_feature_dct(blocks), extract_feature_dct(blocks + 20), atol=1e-5)


@pytest.mark.parametrize('feature', ['A', 'B', 'C', 'D', 'H', 'DCT'])
def test_localization_and_negative(feature):
    from src.metrics import calculate_pixel_metrics
    rng = np.random.default_rng(12)
    original = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    forged = original.copy()
    forged[80:112, 80:112] = original[16:48, 16:48]
    gt = np.zeros((128, 128), dtype=np.uint8)
    gt[16:48, 16:48] = gt[80:112, 80:112] = 255
    config = dict(feature=feature, block_size=8, stride=2, eps=.01,
                  threshold_t=8, d_min=30, area_min=60)
    result = detect_copymove_block(forged, config)
    assert result.is_forged
    assert calculate_pixel_metrics(result.mask, gt)['f1'] > .95
    assert not detect_copymove_block(original, config).is_forged


def test_keypoint_reciprocal_votes_do_not_pass_threshold(monkeypatch):
    import cv2
    import src.detectors.copymove_block as detector_module

    class Detector:
        def detectAndCompute(self, image, mask):
            points = [cv2.KeyPoint(float(x), float(y), 1)
                      for y, x in [(20, 20), (20, 100), (40, 20), (40, 100), (100, 50)]]
            return points, np.zeros((5, 128), dtype=np.float32)

    class Matcher:
        def knnMatch(self, query, train, k):
            matches = []
            for i, best in [(0, 1), (1, 0), (2, 3), (3, 2)]:
                matches.append([cv2.DMatch(i, i, 0), cv2.DMatch(i, best, 1),
                                cv2.DMatch(i, 4, 10)])
            return matches

    monkeypatch.setattr(cv2, 'SIFT_create', lambda **kwargs: Detector())
    monkeypatch.setattr(cv2, 'BFMatcher', lambda norm: Matcher())
    pairs, shifts, _ = detector_module._detect_keypoints_conventional(
        np.zeros((128, 128), dtype=np.uint8), threshold_t=12)
    # Reciprocal observations cannot create twelve unique inliers.
    assert pairs == []
    assert len(shifts) == 0
