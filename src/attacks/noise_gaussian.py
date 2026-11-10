"""Image degradation algorithm."""
import numpy as np
from typing import Optional

def apply_gaussian_noise(
    image: np.ndarray,
    mean: float = 0.0,
    sigma: float = 10.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Menambahkan derau Gaussian additive: I_noisy = clip(I + N(mean, sigma^2), 0, 255).
    """
    rng = np.random.RandomState(seed)
    noise = rng.normal(mean, sigma, image.shape)
    noisy_img = image.astype(np.float32) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)
