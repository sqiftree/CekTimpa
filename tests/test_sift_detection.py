"""
Unit Tests untuk Detektor Copy-Move Berbasis SIFT (Amerini dkk., 2011)
=====================================================================
Memvalidasi:
1. Invariance terhadap translasi
2. Invariance terhadap rotasi (fitur unggulan SIFT vs Block-based)
3. Ketahanan citra asli (tidak memicu false positive)
4. Kontrak struktur SIFTDetectionResult
"""

import numpy as np
import cv2
import pytest

from src.detectors.copymove_sift import detect_copymove_sift, SIFTDetectionResult, SIFTDetector
from src.detectors import detect_copymove, get_detector


def _create_textured_patch(size: int = 40) -> np.ndarray:
    """Membuat patch bertekstur kaya agar memiliki keypoints SIFT yang jelas."""
    patch = np.zeros((size, size, 3), dtype=np.uint8)
    # Tambahkan pola lingkaran, persegi, dan garis kontras tinggi
    cv2.circle(patch, (size // 2, size // 2), size // 3, (255, 200, 50), -1)
    cv2.rectangle(patch, (size // 4, size // 4), (3 * size // 4, 3 * size // 4), (30, 80, 220), 2)
    cv2.line(patch, (0, 0), (size, size), (200, 20, 20), 2)
    cv2.line(patch, (size, 0), (0, size), (20, 200, 20), 2)
    return patch


def test_sift_detector_clean_image():
    """Memastikan citra asli tanpa duplikasi tidak memicu false positive pada SIFT."""
    from src.forgery_generator import generate_synthetic_forgery

    # Uji 1: Citra sintetis standar otentik (objek tunggal)
    clean_img, _, _ = generate_synthetic_forgery("clean_single")
    config = {
        "min_inliers": 8,
        "ratio_threshold": 0.75,
        "d_min": 35.0,
    }
    result = detect_copymove_sift(clean_img, config)
    assert isinstance(result, SIFTDetectionResult)
    assert result.is_forged is False
    assert np.all(result.mask == 0)

    # Uji 2: Patch tunggal bertekstur dengan d_min melampaui dimensi objek tunggal
    img = np.full((200, 200, 3), 240, dtype=np.uint8)
    patch = _create_textured_patch(40)
    img[30:70, 30:70] = patch
    result_patch = detect_copymove_sift(img, {"d_min": 50.0, "min_inliers": 8})
    assert result_patch.is_forged is False
    assert np.all(result_patch.mask == 0)


def test_sift_detector_forged_translation():
    """Memastikan SIFT mendeteksi duplikasi copy-move dengan translasi murni."""
    img = np.full((220, 220, 3), 240, dtype=np.uint8)
    patch = _create_textured_patch(45)

    # Sumber (30, 30) dan Duplikat (130, 130) -> translasi dx=100, dy=100
    img[30:75, 30:75] = patch
    img[130:175, 130:175] = patch

    config = {
        "min_inliers": 6,
        "ratio_threshold": 0.80,
        "d_min": 30.0,
    }
    result = detect_copymove_sift(img, config)

    assert isinstance(result, SIFTDetectionResult)
    assert result.is_forged is True
    assert result.n_inliers >= config["min_inliers"]
    assert np.any(result.mask > 0)
    assert result.homography is not None


def test_sift_detector_forged_rotation():
    """Memastikan SIFT mampu mendeteksi duplikasi yang dirotasi (keunggulan vs Bilateral)."""
    img = np.full((250, 250, 3), 240, dtype=np.uint8)
    patch = _create_textured_patch(50)

    # Sumber di (30, 30)
    img[30:80, 30:80] = patch

    # Duplikat dirotasi 45 derajat lalu ditempel di (140, 140)
    center = (25, 25)
    rot_mat = cv2.getRotationMatrix2D(center, 45, 1.0)
    rotated_patch = cv2.warpAffine(patch, rot_mat, (50, 50), borderValue=(240, 240, 240))
    img[140:190, 140:190] = rotated_patch

    config = {
        "min_inliers": 5,
        "ratio_threshold": 0.85,
        "d_min": 35.0,
    }
    result = detect_copymove_sift(img, config)

    assert isinstance(result, SIFTDetectionResult)
    assert result.is_forged is True
    assert result.n_inliers >= config["min_inliers"]
    assert "Affine" in result.transform_type or "Rotasi" in result.transform_type or result.homography is not None


def test_unified_detector_interface():
    """Memvalidasi antarmuka factory get_detector dan detect_copymove."""
    detector_sift = get_detector("sift")
    assert isinstance(detector_sift, SIFTDetector)

    img = np.full((100, 100, 3), 200, dtype=np.uint8)
    res_sift = detect_copymove(img, method="sift")
    assert isinstance(res_sift, SIFTDetectionResult)

    res_block = detect_copymove(img, method="bilateral", config={"threshold_t": 5})
    assert hasattr(res_block, "mask")
    assert hasattr(res_block, "is_forged")


def test_invalid_inputs():
    """Memastikan validasi input melempar ValueError pada citra kosong atau metode salah."""
    with pytest.raises(ValueError):
        detect_copymove(np.empty((0, 0, 3), dtype=np.uint8), method="sift")

    with pytest.raises(ValueError):
        get_detector("metode_tidak_ada")
