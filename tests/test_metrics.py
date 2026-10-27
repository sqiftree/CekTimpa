"""
Unit Tests untuk Modul Metrik dan Simulasi Gangguan (Attacks)
============================================================
"""

import numpy as np
import pytest

from src.metrics import (
    calculate_mse,
    calculate_psnr,
    calculate_pixel_metrics,
    calculate_image_metrics,
)
from src.attacks import (
    apply_gaussian_noise,
    apply_salt_pepper_noise,
    apply_gaussian_blur,
    apply_gamma_correction,
    apply_histogram_equalization,
    apply_jpeg_compression,
)


def test_metrics_identical_images():
    """Citra identik harus menghasilkan MSE=0 dan PSNR=inf."""
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    assert calculate_mse(img, img) == 0.0
    assert calculate_psnr(img, img) == float("inf")


def test_pixel_metrics_perfect_match():
    """Prediksi sempurna harus menghasilkan Precision=1, Recall=1, F1=1, IoU=1."""
    gt = np.zeros((50, 50), dtype=np.uint8)
    gt[10:30, 10:30] = 255
    pred = gt.copy()

    metrics = calculate_pixel_metrics(pred, gt)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["iou"] == 1.0
    assert metrics["specificity"] == 1.0


def test_pixel_metrics_partial_match():
    """Uji kasus parsial dengan TP, FP, dan FN."""
    gt = np.zeros((10, 10), dtype=np.uint8)
    pred = np.zeros((10, 10), dtype=np.uint8)

    gt[0:4, 0:4] = 255    # 16 piksel
    pred[2:6, 0:4] = 255  # 16 piksel, overlap 8 piksel

    metrics = calculate_pixel_metrics(pred, gt)
    assert metrics["tp"] == 8
    assert metrics["fp"] == 8
    assert metrics["fn"] == 8
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5


def test_image_metrics_evaluation():
    """Uji metrik evaluasi tingkat citra."""
    preds = [True, True, False, False, True]
    labels = [True, False, False, True, True]

    res = calculate_image_metrics(preds, labels)
    assert res["tp"] == 2
    assert res["fp"] == 1
    assert res["fn"] == 1
    assert res["tn"] == 1
    assert res["total"] == 5
    assert res["accuracy"] == 0.6


def test_attacks_shapes_and_types():
    """Memastikan fungsi attack mengembalikan citra dengan shape dan dtype uint8 yang konsisten."""
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    noisy = apply_gaussian_noise(img, sigma=5.0)
    assert noisy.shape == img.shape and noisy.dtype == np.uint8

    sp = apply_salt_pepper_noise(img, amount=0.05)
    assert sp.shape == img.shape and sp.dtype == np.uint8

    blur = apply_gaussian_blur(img, sigma=1.5)
    assert blur.shape == img.shape and blur.dtype == np.uint8

    gamma = apply_gamma_correction(img, gamma=1.5)
    assert gamma.shape == img.shape and gamma.dtype == np.uint8

    he = apply_histogram_equalization(img)
    assert he.shape == img.shape and he.dtype == np.uint8

    jpg = apply_jpeg_compression(img, quality=50)
    assert jpg.shape == img.shape and jpg.dtype == np.uint8
