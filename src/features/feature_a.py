"""Sub-block mean feature (A)."""
import numpy as np

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
