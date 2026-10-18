"""
Modul Ekstraksi Fitur Blok (Domain Spasial & Domain Frekuensi)
Materi PCD:
- Histogram Statistics (Fitur H)
- Spatial Filtering: Box Filter (Fitur A) & Gaussian Lowpass (Fitur B)
- 2D Discrete Fourier Transform / DFT (Fitur C & D)
"""

import numpy as np

def extract_feature_h(blocks: np.ndarray) -> np.ndarray:
    """Fitur H: Statistik histogram blok (Mean, Variance, Skewness)."""
    # TODO: Implementasi oleh mahasiswa
    raise NotImplementedError("Fitur H belum diimplementasikan")

def extract_feature_a(blocks: np.ndarray, g_sub: int = 4) -> np.ndarray:
    """Fitur A: Rata-rata sub-blok G x G (Box filter)."""
    # TODO: Implementasi oleh mahasiswa
    raise NotImplementedError("Fitur A belum diimplementasikan")

def extract_feature_b(image: np.ndarray, coords: np.ndarray, block_size: int = 16, sigma: float = 1.0) -> np.ndarray:
    """Fitur B: Gaussian lowpass filter + subsampling dalam blok."""
    # TODO: Implementasi oleh mahasiswa
    raise NotImplementedError("Fitur B belum diimplementasikan")

def extract_feature_c(blocks: np.ndarray, low_freq_l: int = 4) -> np.ndarray:
    """Fitur C: Magnitudo DFT 2D frekuensi rendah |F(u,v)|."""
    # TODO: Implementasi oleh mahasiswa
    raise NotImplementedError("Fitur C belum diimplementasikan")

def extract_feature_d(blocks: np.ndarray) -> np.ndarray:
    """Fitur D: DFT Radial R(r) berdasar properti rotasi Fourier."""
    # TODO: Implementasi oleh mahasiswa
    raise NotImplementedError("Fitur D belum diimplementasikan")
