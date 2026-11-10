"""Low-frequency 2D Fourier magnitude feature (C)."""
import numpy as np

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
