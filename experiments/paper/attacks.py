"""Gangguan citra untuk eksperimen paper (mandiri, tidak bergantung src/attacks).

Perilaku mengikuti src/attacks milik repo: noise Gaussian aditif, salt & pepper,
blur Gaussian, koreksi gamma, histogram equalization (kanal Y), dan JPEG in-memory.
Semua menerima dan mengembalikan citra RGB uint8.
"""
from typing import Optional

import cv2
import numpy as np


def apply_gaussian_noise(image: np.ndarray, mean: float = 0.0, sigma: float = 10.0,
                         seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.RandomState(seed)
    noisy = image.astype(np.float32) + rng.normal(mean, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_salt_pepper_noise(image: np.ndarray, amount: float = 0.02,
                            seed: Optional[int] = None) -> np.ndarray:
    rng = np.random.RandomState(seed)
    noisy = image.copy()
    h, w = image.shape[:2]
    n = int(amount * h * w)
    noisy[tuple(rng.randint(0, d, n // 2) for d in (h, w))] = 255
    noisy[tuple(rng.randint(0, d, n // 2) for d in (h, w))] = 0
    return noisy


def apply_gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    k = int(2 * np.ceil(3 * sigma) + 1)
    return cv2.GaussianBlur(image, (k, k), sigmaX=float(sigma), sigmaY=float(sigma))


def apply_gamma_correction(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)


def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.equalizeHist(image)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def apply_jpeg_compression(image: np.ndarray, quality: int = 75) -> np.ndarray:
    quality = max(1, min(100, int(quality)))
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
    _, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR if image.ndim == 3 else cv2.IMREAD_GRAYSCALE)
    return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB) if image.ndim == 3 else dec
