"""
CekTimpa: Abstraksi Dasar & Standar Antarmuka Detektor (Clean Architecture)
========================================================================
Menyediakan kontrak abstrak BaseDetector dan struktur BaseDetectionResult
untuk memastikan keseragaman protokol deteksi copy-move (Block-based vs Keypoint-based).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class BaseDetectionResult:
    """
    Struktur data standar untuk hasil deteksi pemalsuan copy-move.
    
    Attributes:
        mask: np.ndarray biner (H, W) uint8 dengan nilai 255 pada area manipulasi.
        boundary: np.ndarray biner (H, W) uint8 garis tepi luar manipulasi.
        overlay: np.ndarray RGB (H, W, 3) citra asli dengan penanda visual.
        is_forged: True jika citra terindikasi memiliki manipulasi salin-tempel.
        timings: Dictionary durasi eksekusi tiap tahapan dalam satuan milidetik (ms).
        metadata: Metadata tambahan spesifik algoritma (parameter, inlier count, dll).
    """
    mask: np.ndarray
    boundary: np.ndarray
    overlay: np.ndarray
    is_forged: bool
    timings: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def forged_pixels_count(self) -> int:
        """Jumlah piksel terdeteksi sebagai area hasil salin-tempel."""
        return int(np.sum(self.mask > 0))

    @property
    def forged_area_percentage(self) -> float:
        """Persentase luas area terdeteksi terhadap luas total citra (0.0 - 100.0%)."""
        if self.mask.size == 0:
            return 0.0
        return float((self.forged_pixels_count / self.mask.size) * 100.0)

    @property
    def total_time_ms(self) -> float:
        """Total waktu eksekusi pipeline dalam milidetik (ms)."""
        return float(self.timings.get("total_ms", 0.0))


class BaseDetector(ABC):
    """
    Antarmuka abstrak untuk seluruh mesin detektor pemalsuan copy-move.
    """

    @abstractmethod
    def detect(self, image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> BaseDetectionResult:
        """
        Menjalankan proses deteksi pada citra masukan.
        
        Args:
            image: Array citra RGB (H, W, 3) atau 2D uint8
            config: Opsi konfigurasi parameter algoritma
            
        Returns:
            Objek BaseDetectionResult atau turunannya
        """
        pass
