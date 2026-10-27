import os
import cv2
import numpy as np
import pytest

from src.preprocessing import (
    to_grayscale,
    apply_bilateral_filter,
    compute_bilateral_multiscale_detail,
)
from src.features import (
    get_dct_matrix,
    get_zigzag_indices,
    extract_feature_dct_bilateral,
)
from src.detectors.copymove_block import detect_copymove_block, DetectionResult


def test_to_grayscale_luminance():
    """Memastikan konversi grayscale menghasilkan array 2D uint8 dengan formula Luminance."""
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[:, :, 0] = 100  # R
    rgb[:, :, 1] = 150  # G
    rgb[:, :, 2] = 200  # B

    gray = to_grayscale(rgb, method="luminance")
    assert gray.shape == (64, 64)
    assert gray.dtype == np.uint8
    expected = int(np.round(0.299 * 100 + 0.587 * 150 + 0.114 * 200))
    assert abs(int(gray[0, 0]) - expected) <= 1


def test_bilateral_multiscale_detail():
    """Memastikan layer detail D^2 = BF1 - BF2 terhitung dengan benar."""
    img = np.random.randint(50, 200, (64, 64), dtype=np.uint8)
    detail = compute_bilateral_multiscale_detail(img, d=3, sigma_space=1.0, sigma_color=25.5)

    assert detail.shape == (64, 64)
    assert detail.dtype == np.float32
    assert not np.all(detail == 0)


def test_dct_matrix_orthogonality():
    """Memastikan matriks DCT-II 1D berukuran b x b ortogonal (C @ C.T = I)."""
    b = 8
    c = get_dct_matrix(b)
    assert c.shape == (b, b)
    identity = c @ c.T
    np.testing.assert_allclose(identity, np.eye(b), atol=1e-5)


def test_zigzag_indices():
    """Memastikan pola zig-zag menghasilkan 64 indeks unik untuk blok 8x8."""
    indices = get_zigzag_indices(b=8)
    assert indices.shape == (64, 2)
    unique_coords = set(tuple(p) for p in indices)
    assert len(unique_coords) == 64
    assert tuple(indices[0]) == (0, 0)
    assert tuple(indices[1]) == (0, 1)
    assert tuple(indices[2]) == (1, 0)


def test_extract_feature_dct_bilateral():
    """Memastikan ekstraksi fitur DCT-Bilateral menghasilkan vektor shape (N, 16) untuk P=0.25."""
    n_blocks = 10
    b = 8
    blocks = np.random.randint(0, 255, (n_blocks, b, b), dtype=np.uint8)
    detail = np.random.randn(n_blocks, b, b).astype(np.float32)

    feats = extract_feature_dct_bilateral(blocks, detail, p_ratio=0.25)
    assert feats.shape == (n_blocks, 16)
    assert feats.dtype == np.float32


def test_detector_clean_image():
    """Memastikan citra asli tanpa salin-tempel tidak memicu false positive."""
    np.random.seed(123)
    clean_rgb = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

    config = {
        "block_size": 8,
        "stride": 2,
        "d_min": 20.0,
        "threshold_t": 10,
        "q_step": 16.0,
        "eps": 5.0,
    }
    result = detect_copymove_block(clean_rgb, config)

    assert isinstance(result, DetectionResult)
    assert bool(result.is_forged) is False
    assert len(result.shifts) == 0
    assert len(result.pairs) == 0


def test_detector_forged_image():
    """Memastikan manipulasi salin-tempel berhasil dideteksi dengan translasi eksak."""
    np.random.seed(123)
    img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)

    # Duplikasi patch dari (16, 16) ke (64, 64) -> translasi dy=48, dx=48
    patch = img[16:48, 16:48].copy()
    img[64:96, 64:96] = patch

    config = {
        "block_size": 8,
        "stride": 2,
        "d_min": 20.0,
        "threshold_t": 10,
        "q_step": 16.0,
        "eps": 5.0,
    }
    result = detect_copymove_block(img, config)

    assert isinstance(result, DetectionResult)
    assert bool(result.is_forged) is True
    assert len(result.shifts) == 1
    assert tuple(result.shifts[0]) == (48, 48)
    assert len(result.pairs) > 0
    assert result.mask.shape == (128, 128)
    assert np.any(result.mask > 0)


def test_detector_coverage_dataset():
    """Menguji deteksi copy-move pada dataset COVERAGE riil (10t.tif)."""
    coverage_file = "dataset/OneDrive_2_10-25-2026/10t.tif"
    if not os.path.exists(coverage_file):
        pytest.skip("Dataset COVERAGE 10t.tif belum tersedia di direktori dataset/")

    img_bgr = cv2.imread(coverage_file)
    assert img_bgr is not None, "Gagal membaca citra COVERAGE 10t.tif"
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    config = {
        "block_size": 8,
        "stride": 2,
        "d_min": 30.0,
        "threshold_t": 20,
        "q_step": 16.0,
        "eps": 5.0,
        "area_min": 50,
        "bilateral_d": 3,
        "bilateral_sigma_s": 1.0,
        "bilateral_sigma_r": 25.5,
    }
    result = detect_copymove_block(img_rgb, config)

    assert isinstance(result, DetectionResult)
    assert bool(result.is_forged) is True
    assert len(result.shifts) > 0
    assert len(result.pairs) > 0
    assert result.mask.shape == (img_rgb.shape[0], img_rgb.shape[1])
    assert np.any(result.mask > 0)


def test_synthetic_forgery_clean_single():
    """Memastikan citra sintetis asli (objek tunggal) tidak terdeteksi sebagai pemalsuan."""
    from src.forgery_generator import generate_synthetic_forgery

    img, gt_mask, meta = generate_synthetic_forgery("clean_single")
    config = {
        "block_size": 8,
        "stride": 2,
        "d_min": 30.0,
        "threshold_t": 25,
        "q_step": 16.0,
        "eps": 5.0,
        "area_min": 60,
    }
    result = detect_copymove_block(img, config)

    assert result.is_forged is False
    assert len(result.shifts) == 0
    assert len(result.pairs) == 0
    assert np.count_nonzero(result.mask) == 0


def test_synthetic_forgery_tampered():
    """Memastikan citra sintetis manipulasi copy-move berhasil dideteksi dengan benar."""
    from src.forgery_generator import generate_synthetic_forgery

    img, gt_mask, meta = generate_synthetic_forgery("tampered")
    config = {
        "block_size": 8,
        "stride": 2,
        "d_min": 30.0,
        "threshold_t": 25,
        "q_step": 16.0,
        "eps": 5.0,
        "area_min": 60,
    }
    result = detect_copymove_block(img, config)

    assert bool(result.is_forged) is True
    assert len(result.shifts) >= 1
    assert tuple(result.shifts[0]) == meta["shift_vector"]
    assert len(result.pairs) > 50
    assert np.count_nonzero(result.mask) > 500

