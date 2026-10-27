"""
CekTimpa: Paket Forensik Citra Digital (Copy-Move Forgery Detection)
Departemen Informatika, Universitas Sebelas Maret (UNS)
"""

from __future__ import annotations

__version__ = "1.0.0"

from .detectors import (
    BaseDetector,
    BaseDetectionResult,
    SIFTDetector,
    SIFTDetectionResult,
    BlockDetector,
    BlockDetectionResult,
    detect_copymove,
    get_detector,
)
from .forgery_generator import generate_synthetic_forgery
from .metrics import (
    calculate_mse,
    calculate_psnr,
    calculate_pixel_metrics,
    calculate_image_metrics,
)
from .attacks import (
    apply_gaussian_noise,
    apply_salt_pepper_noise,
    apply_gaussian_blur,
    apply_gamma_correction,
    apply_histogram_equalization,
    apply_jpeg_compression,
)

__all__ = [
    "BaseDetector",
    "BaseDetectionResult",
    "SIFTDetector",
    "SIFTDetectionResult",
    "BlockDetector",
    "BlockDetectionResult",
    "detect_copymove",
    "get_detector",
    "generate_synthetic_forgery",
    "calculate_mse",
    "calculate_psnr",
    "calculate_pixel_metrics",
    "calculate_image_metrics",
    "apply_gaussian_noise",
    "apply_salt_pepper_noise",
    "apply_gaussian_blur",
    "apply_gamma_correction",
    "apply_histogram_equalization",
    "apply_jpeg_compression",
]
