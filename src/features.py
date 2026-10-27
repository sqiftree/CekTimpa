"""
Modul Ekstraksi Fitur Blok Hibrida DCT-Bilateral
Metode Rujukan: Kaplan, Aksakalli, Kilic, & Erer (BAJECE 2020)
"A Novel Approach for Copy-move Forgery Detection using Bilateral Filtering"
"""

import numpy as np


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
