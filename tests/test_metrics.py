import numpy as np
import pytest
from src.metrics import calculate_pixel_metrics, calculate_image_metrics, calculate_psnr


def test_confusion_counts_and_scores():
    m = calculate_pixel_metrics(np.array([255, 255, 0, 0]), np.array([1, 0, 1, 0]))
    assert [m[k] for k in ('tp', 'fp', 'fn', 'tn')] == [1, 1, 1, 1]
    assert m['f1'] == .5
    assert m['iou'] == pytest.approx(1 / 3)
    m = calculate_image_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert m['balanced_accuracy'] == .5
    assert m['fpr'] == .5


def test_empty_positive_convention_and_invalid_input():
    assert calculate_pixel_metrics(np.zeros((2, 2)), np.zeros((2, 2)))['f1'] == 0
    with pytest.raises(ValueError):
        calculate_pixel_metrics(np.zeros((2, 2)), np.zeros((3, 3)))
    with pytest.raises(ValueError):
        calculate_image_metrics([2], [1])
    assert np.isnan(calculate_image_metrics([1], [1])['balanced_accuracy'])


def test_psnr_avoids_uint8_overflow():
    assert calculate_psnr(np.array([0], dtype=np.uint8), np.array([255], dtype=np.uint8)) == 0
    assert np.isinf(calculate_psnr(np.zeros(2), np.zeros(2)))


def test_alternate_attacks_are_reproducible_and_preserve_rgb():
    from src.attacks import apply_gaussian_noise, apply_gamma_correction, apply_jpeg_compression
    image = np.full((32, 32, 3), [180, 80, 30], dtype=np.uint8)
    assert np.array_equal(apply_gaussian_noise(image, seed=9), apply_gaussian_noise(image, seed=9))
    assert np.all(apply_gamma_correction(image, gamma=2) < image)
    compressed = apply_jpeg_compression(image, 80)
    assert compressed.shape == image.shape
    assert compressed[..., 0].mean() > compressed[..., 2].mean()
