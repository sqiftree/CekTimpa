"""Numerical invariants and meaningful configurable detector regressions."""
import numpy as np
import pytest
from src.features import (extract_feature_b, extract_feature_b_local, extract_feature_c,
                          extract_feature_dct, standardize_feature_vectors)
from src.detectors.copymove_block import detect_copymove_block


def test_local_gaussian_ignores_surrounding_pixels():
    patch = np.random.default_rng(3).integers(30, 200, (16, 16), dtype=np.uint8)
    image = np.zeros((64, 64), np.uint8)
    image[32:, :] = 255
    image[8:24, 8:24] = patch
    image[40:56, 40:56] = patch
    coords = np.array([[8, 8], [40, 40]])
    old = extract_feature_b(image, coords)
    local = extract_feature_b_local(np.stack([patch, patch]))
    assert not np.array_equal(old[0], old[1])
    np.testing.assert_array_equal(local[0], local[1])


def test_fft_per_pixel_scale_and_dct_dc():
    for b in [8, 16]:
        blocks = np.full((1, b, b), 80, np.float32)
        normalized = extract_feature_c(blocks, normalize=True)
        assert normalized[0, 0] == 80
        dc = extract_feature_dct(blocks, include_dc=True)
        brighter = extract_feature_dct(blocks + 10, include_dc=True)
        assert brighter[0, 0] - dc[0, 0] == pytest.approx(10, abs=1e-5)


def test_standardization_handles_constant_columns_and_units():
    values = np.float32([[1, 100, 7], [2, 200, 7], [3, 300, 7]])
    z = standardize_feature_vectors(values)
    assert np.isfinite(z).all()
    np.testing.assert_allclose(z[:, 0], z[:, 1], atol=1e-6)
    assert np.all(z[:, 2] == 0)


@pytest.mark.parametrize('config', [
    {'feature': 'unknown'}, {'stride': 0}, {'q_step': 0}, {'eps': float('nan')},
    {'feature': 'A', 'g_sub': 3}, {'feature': 'DCT', 'n_coeffs': 1000},
    {'feature': 'C', 'low_freq_l': 20}, {'feature_scaling': 'invalid'},
])
def test_invalid_configuration_is_rejected(config):
    image = np.random.default_rng(4).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        detect_copymove_block(image, {'block_size': 8, **config})


@pytest.mark.parametrize('feature', ['A', 'B', 'C', 'D', 'H', 'DCT', 'DCT_BILATERAL'])
def test_standardized_pipeline_detects_known_translation(feature):
    image = np.random.default_rng(9).integers(0, 256, (96, 96, 3), dtype=np.uint8)
    image[56:80, 56:80] = image[8:32, 8:32]
    result = detect_copymove_block(image, dict(feature=feature, block_size=8,
        feature_scaling='standardized', eps=.0001, threshold_t=4, d_min=20))
    assert result.is_forged
    assert result.metadata['feature_dimensions'] > 0
    assert (48, 48) in map(tuple, result.shifts)
    alternate_q = detect_copymove_block(image, dict(feature=feature, block_size=8,
        feature_scaling='standardized', eps=.0001, q_step=16, threshold_t=4, d_min=20))
    np.testing.assert_array_equal(result.mask, alternate_q.mask)
    assert alternate_q.metadata['effective_q_step'] == 1.
