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

def extract_feature_a(blocks: np.ndarray, g_sub: int = 4) -> np.ndarray:
    """
    Fitur A: Rata-rata sub-blok G x G (Box filter).
    Membagi blok B x B menjadi G x G sub-blok, lalu menghitung rata-rata intensitas tiap sub-blok.
    
    Returns:
        np.ndarray shape (N, G * G) bertipe float32
    """
    n, b, _ = blocks.shape
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
    blurred = cv2.GaussianBlur(image_gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    step = block_size // g_samples
    
    n_blocks = len(coords)
    features = np.empty((n_blocks, g_samples * g_samples), dtype=np.float32)
    
    for i in range(n_blocks):
        y, x = coords[i]
        patch = blurred[y : y + block_size : step, x : x + block_size : step]
        features[i] = patch[:g_samples, :g_samples].ravel()
        
    return features

def extract_feature_c(blocks: np.ndarray, low_freq_l: int = 4) -> np.ndarray:
    """
    Fitur C: Magnitudo DFT 2D frekuensi rendah |F(u,v)|.
    """
    fft_blocks = np.fft.fft2(blocks.astype(np.float32))
    mag = np.abs(fft_blocks[:, :low_freq_l, :low_freq_l])
    return mag.reshape(len(blocks), -1).astype(np.float32)

def extract_feature_d(blocks: np.ndarray) -> np.ndarray:
    """
    Fitur D: DFT Radial R(r) berdasar properti rotasi Fourier.
    """
    n, b, _ = blocks.shape
    fft_shifted = np.fft.fftshift(np.fft.fft2(blocks.astype(np.float32)), axes=(1, 2))
    mag = np.abs(fft_shifted)
    
    center = b // 2
    y, x = np.ogrid[:b, :b]
    r_map = np.sqrt((x - center)**2 + (y - center)**2).astype(np.int32)
    
    max_r = b // 2
    features = np.zeros((n, max_r), dtype=np.float32)
    for r in range(1, max_r + 1):
        mask = (r_map == r)
        if np.any(mask):
            features[:, r - 1] = mag[:, mask].mean(axis=1)
            
    return features
