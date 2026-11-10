"""RGB to HSI color conversion."""
from typing import Tuple
import numpy as np
def rgb_to_hsi(image_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mengonversi citra RGB ke model warna HSI (Hue, Saturation, Intensity).
    Penanganan khusus pembagian nol saat R=G=B atau saturasi rendah.
    """
    img = image_rgb.astype(np.float32) / 255.0
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    # Intensitas I
    intensity = (r + g + b) / 3.0

    # Saturasi S
    min_rgb = np.minimum(np.minimum(r, g), b)
    sum_rgb = r + g + b
    saturation = np.zeros_like(intensity)
    non_zero = sum_rgb > 1e-6
    saturation[non_zero] = 1.0 - (3.0 / sum_rgb[non_zero]) * min_rgb[non_zero]

    # Hue H
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + 1e-7
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))

    hue = theta.copy()
    hue[b > g] = 2.0 * np.pi - hue[b > g]
    hue = hue / (2.0 * np.pi)  # Normalisasi ke [0, 1]

    return hue, saturation, intensity
