"""
CekTimpa: Paket Modul Detektor Copy-Move Forgery Detection
==========================================================
Menyediakan antarmuka terpadu untuk algoritma deteksi copy-move:
1. SIFT + FLANN + RANSAC (Amerini et al., IEEE TIFS 2011) — Invariant rotasi, skala, perspektif.
2. DCT + Bilateral Filter (Kaplan et al., BAJECE 2020) — Block-based dense pixel analysis.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import numpy as np

from .base import BaseDetector, BaseDetectionResult
from .copymove_block import (
    BlockDetector,
    BlockDetectionResult,
    DetectionResult,  # Backward compatibility alias
    detect_copymove_block,
)
from .copymove_sift import (
    SIFTDetector,
    SIFTDetectionResult,
    detect_copymove_sift,
)


DETECTOR_REGISTRY = {
    "sift": SIFTDetector,
    "block": BlockDetector,
    "bilateral": BlockDetector,
    "dct_bilateral": BlockDetector,
}


def get_detector(method: str = "sift", default_config: Optional[Dict[str, Any]] = None) -> BaseDetector:
    """
    Factory method untuk mendapatkan instans detektor berdasarkan nama metode.

    Args:
        method: Nama algoritma ("sift", "block", "bilateral", "dct_bilateral")
        default_config: Konfigurasi default detektor (opsional)

    Returns:
        Instans turunan BaseDetector
    """
    key = method.lower().strip()
    if key not in DETECTOR_REGISTRY:
        supported = ", ".join(sorted(DETECTOR_REGISTRY.keys()))
        raise ValueError(f"Metode deteksi '{method}' tidak didukung. Pilih salah satu dari: {supported}")
    return DETECTOR_REGISTRY[key](default_config=default_config)


def detect_copymove(
    image: np.ndarray,
    method: str = "sift",
    config: Optional[Dict[str, Any]] = None,
) -> BaseDetectionResult:
    """
    Fungsi terpadu untuk menjalankan deteksi copy-move menggunakan metode yang dipilih.

    Args:
        image: Citra masukan RGB (H, W, 3) atau 2D uint8
        method: Algoritma yang digunakan ("sift" atau "bilateral" / "block")
        config: Kamus parameter konfigurasi deteksi

    Returns:
        BaseDetectionResult (SIFTDetectionResult atau BlockDetectionResult)
    """
    detector = get_detector(method=method)
    return detector.detect(image, config=config)


__all__ = [
    "BaseDetector",
    "BaseDetectionResult",
    "BlockDetector",
    "BlockDetectionResult",
    "DetectionResult",
    "SIFTDetector",
    "SIFTDetectionResult",
    "detect_copymove_block",
    "detect_copymove_sift",
    "detect_copymove",
    "get_detector",
    "DETECTOR_REGISTRY",
]
