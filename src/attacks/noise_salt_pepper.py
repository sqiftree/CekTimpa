"""Image degradation algorithm."""
import numpy as np
from typing import Optional

def apply_salt_pepper_noise(
    image: np.ndarray,
    amount: float = 0.02,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Menambahkan derau impulsif salt-and-pepper.
    amount: proporsi total piksel yang terkontaminasi (default 0.02 = 2%).
    """
    rng = np.random.RandomState(seed)
    noisy = image.copy()
    h, w = image.shape[:2]
    num_noisy = int(amount * h * w)

    # Salt (piksel putih)
    coords_salt = [rng.randint(0, dim, num_noisy // 2) for dim in (h, w)]
    noisy[tuple(coords_salt)] = 255

    # Pepper (piksel hitam)
    coords_pepper = [rng.randint(0, dim, num_noisy // 2) for dim in (h, w)]
    noisy[tuple(coords_pepper)] = 0

    return noisy
