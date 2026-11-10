"""Peak signal-to-noise ratio."""
import numpy as np
def calculate_psnr(image_a, image_b, data_range=255.0):
    """PSNR for 0..255 images; pass data_range=1 for normalized images."""
    a, b = np.asarray(image_a), np.asarray(image_b)
    if a.shape != b.shape or a.size == 0 or data_range <= 0:
        raise ValueError("equal nonempty shapes and positive data_range required")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("images must contain finite values")
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else float(10 * np.log10(data_range ** 2 / mse))
