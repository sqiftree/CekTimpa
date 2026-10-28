"""
Modul Simulasi Gangguan Citra (Attacks / Degradasi Forensik Citra)
================================================================
Materi Pengolahan Citra Digital (PCD):
- Noise Models: Gaussian Noise & Salt-and-Pepper
- Spatial Filtering: Gaussian Blur (Konvolusi Spasial)
- Intensity Transformation: Power-law (Gamma) & Contrast Stretching
- Histogram Equalization (Perataan Histogram pada channel Luminance Y / V)
- JPEG Compression (Lossy Transform Coding & Quantization)
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import cv2


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


def apply_gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Menerapkan penghalusan Gaussian blur dengan kernel ukuran ganjil otomatis (6*sigma + 1).
    """
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=float(sigma), sigmaY=float(sigma))


def apply_gamma_correction(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """
    Menerapkan transformasi power-law (gamma correction):
    s = c * r^gamma
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    exponent = gamma
    table = np.array([((i / 255.0) ** exponent) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)


def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Menerapkan perataan histogram (histogram equalization).
    Untuk citra berwarna (RGB), equalization diterapkan pada channel Luminance (YCrCb)
    agar tidak merusak chromaticity warna asli.
    """
    if image.ndim == 2:
        return cv2.equalizeHist(image)

    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def apply_jpeg_compression(image: np.ndarray, quality: int = 75) -> np.ndarray:
    """
    Menerapkan simulasi kompresi JPEG lossy dengan encode/decode in-memory.
    quality: tingkat kualitas JPEG (1-100, semakin kecil semakin terkompresi).
    """
    quality = max(1, min(100, int(quality)))
    is_rgb = (image.ndim == 3)
    bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if is_rgb else image

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode(".jpg", bgr_img, encode_param)
    decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR if is_rgb else cv2.IMREAD_GRAYSCALE)

    if is_rgb:
        return cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)
    return decimg
