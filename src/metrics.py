"""
Modul Metrik Evaluasi Kinerja Sistem & Kualitas Citra
Materi PCD:
- Image Quality Assessment: MSE & PSNR
- Klasifikasi Piksel: Precision, Recall, F1-Score
- Klasifikasi Citra: TPR, FPR, Akurasi
"""

from typing import Tuple, Dict
import numpy as np

def calculate_psnr(image_a: np.ndarray, image_b: np.ndarray) -> float:
    """Menghitung nilai Peak Signal-to-Noise Ratio (PSNR) dalam dB."""
    raise NotImplementedError()

def calculate_pixel_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, float]:
    """
    Menghitung metrik tingkat piksel: Precision, Recall, dan F1-Score.
    """
    raise NotImplementedError()

def calculate_image_metrics(predictions: list, true_labels: list) -> Dict[str, float]:
    """
    Menghitung metrik tingkat citra: TPR, FPR, dan Akurasi.
    """
    raise NotImplementedError()
