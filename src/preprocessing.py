"""
Modul Preprocessing Citra: Bilateral Filtering & Grayscale
Metode Rujukan: Kaplan dkk. (BAJECE 2020)
"""

import numpy as np
import cv2


def to_grayscale(image_rgb: np.ndarray, method: str = "luminance") -> np.ndarray:
    """
    Mengonversi citra RGB ke skala keabuan (grayscale).
    Berdasarkan Step-1 Kaplan dkk. (2020):
    Y = 0.299 * R + 0.587 * G + 0.114 * B (Luminance ITU-R 601)
    """
    if len(image_rgb.shape) == 2:
        return image_rgb.astype(np.uint8)

    img_float = image_rgb.astype(np.float32)
    if method == "intensity":
        gray = np.mean(img_float, axis=2)
    else:
        gray = 0.299 * img_float[:, :, 0] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 2]

    return np.clip(gray, 0, 255).astype(np.uint8)


def apply_bilateral_filter(
    image: np.ndarray,
    d: int = 3,
    sigma_color: float = 25.5,
    sigma_space: float = 1.0,
) -> np.ndarray:
    """
    Menerapkan Bilateral Filtering (Kaplan dkk., 2020).
    Filter bilateral mempertimbangkan jarak spasial (sigma_space) dan perbedaan
    intensitas spektral (sigma_color), sehingga menghaluskan derau pada area
    homogen namun tetap mempertahankan ketajaman tepi batas objek.
    """
    return cv2.bilateralFilter(
        image,
        d=int(d),
        sigmaColor=float(sigma_color),
        sigmaSpace=float(sigma_space),
    )


def compute_bilateral_multiscale_detail(
    image: np.ndarray,
    d: int = 3,
    sigma_space: float = 1.0,
    sigma_color: float = 25.5,
) -> np.ndarray:
    """
    Menghitung dekomposisi multiskala bilateral untuk menghasilkan layer detail (D^2).
    Sesuai Persamaan (3)–(7) pada Kaplan dkk. (2020):
      Level 1: BF^1(I) menggunakan sigma_s dan sigma_r
      Level 2: BF^2(I) diterapkan pada BF^1 dengan menggandakan sigma_s (2 * sigma_s)
               dan membagi dua sigma_r (sigma_r / 2)
      Detail Layer: D^2(I) = BF^1(I) - BF^2(I)
    """
    bf1 = apply_bilateral_filter(
        image,
        d=d,
        sigma_color=sigma_color,
        sigma_space=sigma_space,
    ).astype(np.float32)

    bf2 = cv2.bilateralFilter(
        bf1.astype(np.uint8) if bf1.dtype != np.uint8 else bf1,
        d=d,
        sigmaColor=float(sigma_color / 2.0),
        sigmaSpace=float(sigma_space * 2.0),
    ).astype(np.float32)

    detail_layer = bf1 - bf2
    return detail_layer
