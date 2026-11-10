"""Render detection masks over source images."""
import numpy as np
def create_overlay(image_rgb: np.ndarray, mask: np.ndarray, boundary: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    Membuat visualisasi overlay warna semi-transparan pada area terdeteksi
    dan garis batas tegas (boundary) di tepi manipulasi.
    """
    output = image_rgb.copy()
    
    # Beri warna merah transparan pada area mask
    mask_bool = mask > 0
    if np.any(mask_bool):
        color_fill = np.array([239, 68, 68], dtype=np.uint8)  # Warna merah #EF4444
        blended = (1.0 - alpha) * output[mask_bool] + alpha * color_fill
        output[mask_bool] = blended.astype(np.uint8)

    # Beri warna merah terang tebal pada garis batas (boundary)
    boundary_bool = boundary > 0
    if np.any(boundary_bool):
        output[boundary_bool] = [255, 0, 0]

    return output
