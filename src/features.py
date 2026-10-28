"""
Modul Ekstraksi Fitur Blok (Domain Spasial & Domain Frekuensi)
Materi PCD:
- Fitur A: Rata-rata Sub-blok (Box Filter / Spatial Sampling)
- Fitur B: Gaussian Lowpass + Subsampling
- Fitur C: Magnitudo DFT 2D Frekuensi Rendah
- Fitur D: DFT Radial (Properti Rotasi Fourier)
- Fitur H: Statistik Histogram (Mean, Variance, Skewness)
"""

import numpy as np
import cv2
from scipy.fft import dctn


def extract_feature_dct(blocks: np.ndarray, n_coeffs: int = 16, include_dc: bool = False) -> np.ndarray:
    """Orthonormal DCT-II, low-frequency zigzag, excluding DC.

    Divide by block width to keep coefficient scale comparable across sizes.
    DC is omitted for invariance to additive brightness (without clipping).
    """
    if blocks.ndim != 3 or blocks.shape[1] != blocks.shape[2]:
        raise ValueError("blocks must have shape (N, B, B)")
    b = blocks.shape[1]
    if not 1 <= n_coeffs <= b * b - (not include_dc):
        raise ValueError("n_coeffs must be between 1 and B*B-1")
    zigzag = []
    for diagonal in range(2 * b - 1):
        cells = [(y, diagonal - y) for y in range(b) if 0 <= diagonal - y < b]
        zigzag.extend(cells[::-1] if diagonal % 2 == 0 else cells)
    start = 0 if include_dc else 1
    selected = zigzag[start:start + n_coeffs]
    coefficients = dctn(blocks.astype(np.float32), axes=(1, 2), norm="ortho")
    ys, xs = zip(*selected)
    return (coefficients[:, ys, xs] / b).astype(np.float32)

def extract_feature_a(blocks: np.ndarray, g_sub: int = 4) -> np.ndarray:
    """
    Fitur A: Rata-rata sub-blok G x G (Box filter).
    Membagi blok B x B menjadi G x G sub-blok, lalu menghitung rata-rata intensitas tiap sub-blok.

    Returns:
        np.ndarray shape (N, G * G) bertipe float32
    """
    n, b, _ = blocks.shape
    if g_sub < 1 or b % g_sub:
        raise ValueError("g_sub must divide block width")
    sub_size = b // g_sub

    # Reshape ke (N, G, sub_size, G, sub_size) untuk vektorisasi instan
    reshaped = blocks.reshape(n, g_sub, sub_size, g_sub, sub_size)
    sub_means = reshaped.mean(axis=(2, 4), dtype=np.float32)
    return sub_means.reshape(n, g_sub * g_sub)

def extract_feature_h(blocks: np.ndarray) -> np.ndarray:
    """
    Fitur H: Statistik intensitas blok (Mean, Variance, Skewness).
    """
    blocks_f = blocks.astype(np.float32)
    mean = blocks_f.mean(axis=(1, 2))
    var = blocks_f.var(axis=(1, 2))
    std = np.sqrt(var) + 1e-6
    diff = blocks_f - mean[:, None, None]
    skew = np.mean(diff ** 3, axis=(1, 2)) / (std ** 3)

    return np.column_stack([mean, var, skew]).astype(np.float32)

def extract_feature_b(
    image_gray: np.ndarray,
    coords: np.ndarray,
    block_size: int = 16,
    sigma: float = 1.0,
    g_samples: int = 4
) -> np.ndarray:
    """
    Fitur B: Filter Gaussian Lowpass + subsampling teratur di dalam blok.
    """
    if not np.isfinite(sigma) or sigma <= 0 or not 1 <= g_samples <= block_size:
        raise ValueError("positive sigma and 1 <= g_samples <= B required")
    blurred = cv2.GaussianBlur(image_gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    step = block_size // g_samples

    n_blocks = len(coords)
    features = np.empty((n_blocks, g_samples * g_samples), dtype=np.float32)

    for i in range(n_blocks):
        y, x = coords[i]
        patch = blurred[y : y + block_size : step, x : x + block_size : step]
        features[i] = patch[:g_samples, :g_samples].ravel()

    return features

def extract_feature_c(blocks: np.ndarray, low_freq_l: int = 4, normalize: bool = False) -> np.ndarray:
    """
    Fitur C: Magnitudo DFT 2D frekuensi rendah |F(u,v)|.
    """
    if not 1 <= low_freq_l <= blocks.shape[1]:
        raise ValueError("low_freq_l must be between 1 and block width")
    fft_blocks = np.fft.fft2(blocks.astype(np.float32))
    mag = np.abs(fft_blocks[:, :low_freq_l, :low_freq_l])
    if normalize:
        mag /= blocks.shape[1] ** 2
    return mag.reshape(len(blocks), -1).astype(np.float32)

def extract_feature_d(blocks: np.ndarray, radial_bins: int = None, normalize: bool = False) -> np.ndarray:
    """
    Fitur D: DFT Radial R(r) berdasar properti rotasi Fourier.
    """
    n, b, _ = blocks.shape
    fft_shifted = np.fft.fftshift(np.fft.fft2(blocks.astype(np.float32)), axes=(1, 2))
    mag = np.abs(fft_shifted)

    center = b // 2
    y, x = np.ogrid[:b, :b]
    r_map = np.sqrt((x - center)**2 + (y - center)**2).astype(np.int32)

    max_r = b // 2 if radial_bins is None else int(radial_bins)
    if not 1 <= max_r <= b // 2:
        raise ValueError("radial_bins must be between 1 and B//2")
    features = np.zeros((n, max_r), dtype=np.float32)
    for r in range(1, max_r + 1):
        mask = (r_map == r)
        if np.any(mask):
            features[:, r - 1] = mag[:, mask].mean(axis=1)

    if normalize:
        features /= b ** 2
    return features


def get_dct_matrix(b: int = 8) -> np.ndarray:
    """
    Menghasilkan matriks transformasi DCT-II 1D berukuran b x b yang ortogonal.
    Sesuai Persamaan (1) dan (2) pada Kaplan dkk. (2020).
    D_2D(B) = C @ B @ C.T
    """
    c = np.zeros((b, b), dtype=np.float32)
    for u in range(b):
        alpha = 1.0 / np.sqrt(b) if u == 0 else np.sqrt(2.0 / b)
        for x in range(b):
            c[u, x] = alpha * np.cos(np.pi * u * (2 * x + 1) / (2.0 * b))
    return c


def get_zigzag_indices(b: int = 8) -> np.ndarray:
    """
    Menghasilkan urutan indeks zig-zag (baris, kolom) untuk matriks b x b.
    """
    rows, cols = b, b
    order = []
    for s in range(rows + cols - 1):
        if s % 2 == 0:
            # Bergerak diagonal ke kanan-atas
            r = min(s, rows - 1)
            c = s - r
            while r >= 0 and c < cols:
                order.append((r, c))
                r -= 1
                c += 1
        else:
            # Bergerak diagonal ke kiri-bawah
            c = min(s, cols - 1)
            r = s - c
            while c >= 0 and r < rows:
                order.append((r, c))
                c -= 1
                r += 1
    return np.array(order, dtype=np.int32)


def extract_feature_dct_bilateral(
    blocks: np.ndarray,
    detail_blocks: np.ndarray,
    p_ratio: float = 0.25,
) -> np.ndarray:
    """
    Ekstraksi Fitur Hibrida DCT-Bilateral.

    Langkah-langkah:
    1. Transformasi DCT-II 2D diterapkan pada setiap blok: D(u, v) (Eq. 1)
    2. Perkalian element-wise antara blok DCT dan blok Bilateral detail:
       DCT_Bilateral = DCT(B) * Bilateral_detail(B) (Step-5)
    3. Zigzag scanning untuk mengelompokkan koefisien frekuensi rendah (Step-6)
    4. Mengambil P x B^2 koefisien pertama (P = 0.25 -> 16 nilai untuk blok 8x8)

    Args:
        blocks: np.ndarray berukuran (N, b, b) dari citra grayscale
        detail_blocks: np.ndarray berukuran (N, b, b) dari layer detail bilateral D^2
        p_ratio: rasio reduksi dimensi (default: 0.25 sesuai Kaplan dkk., 2020)

    Returns:
        np.ndarray berukuran (N, int(p_ratio * b^2)) bertipe float32
    """
    if blocks.ndim != 3 or blocks.shape != detail_blocks.shape:
        raise ValueError("blocks and detail_blocks must have equal (N, B, B) shapes")
    if blocks.shape[1] != blocks.shape[2] or not 0 < p_ratio <= 1:
        raise ValueError("square blocks and 0 < p_ratio <= 1 required")
    n, b, _ = blocks.shape
    c_dct = get_dct_matrix(b)

    # 1. 2D DCT vectorized: C @ block @ C.T
    blocks_f = blocks.astype(np.float32)
    dct_blocks = c_dct @ blocks_f @ c_dct.T

    # 2. Perkalian element-wise dengan layer detail bilateral (
    detail_f = detail_blocks.astype(np.float32)
    dct_bilateral = dct_blocks * detail_f

    # 3 & 4. Zig-zag scan dan pengambilan P * B^2 koefisien pertama
    n_coef = max(1, int(p_ratio * b * b))
    zigzag_order = get_zigzag_indices(b)
    rows_idx = zigzag_order[:n_coef, 0]
    cols_idx = zigzag_order[:n_coef, 1]

    features = dct_bilateral[:, rows_idx, cols_idx]
    return features.astype(np.float32)


def extract_feature_b_local(blocks, sigma=1.0, g_samples=4):
    """Gaussian smoothing inside each block; outside context cannot alter matches."""
    b = blocks.shape[1]
    if not np.isfinite(sigma) or sigma <= 0 or not 1 <= g_samples <= b:
        raise ValueError("positive sigma and 1 <= g_samples <= B required")
    samples = np.linspace(0, b - 1, g_samples, dtype=int)
    features = np.empty((len(blocks), g_samples ** 2), dtype=np.float32)
    for i, block in enumerate(blocks):
        blurred = cv2.GaussianBlur(block.astype(np.float32), (0, 0), sigma,
                                   borderType=cv2.BORDER_REFLECT_101)
        features[i] = blurred[np.ix_(samples, samples)].ravel()
    return features


def standardize_feature_vectors(features):
    """Per-image z-scores / sqrt(D): Euclidean distance is standardized RMS.

    No labels or dataset fit. Constant dimensions contribute zero; distributions
    still vary by image, so this is an optional experiment, not calibration.
    """
    values = np.asarray(features, dtype=np.float32)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > 1e-6, scale, 1.)
    return ((values - mean) / scale / np.sqrt(values.shape[1])).astype(np.float32)
