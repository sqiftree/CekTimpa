"""
Modul Pasca-Pemrosesan Morfologi Citra
Materi PCD:
- Operasi Morfologi Biner: Erosi, Dilasi, Opening, Closing
- Hole Filling (Pengisian Lubang)
- Boundary Extraction: beta(A) = A - (A (-) SE)
- Connected Components Area Thresholding
"""

from typing import Tuple, List
import numpy as np
import cv2

def build_detection_mask(
    image_shape: Tuple[int, int],
    matched_pairs: List[Tuple[int, int]],
    coords: np.ndarray,
    block_size: int = 16,
    area_min: int = 100
) -> np.ndarray:
    """
    Membangun mask biner dari pasangan blok yang cocok, lalu membersihkannya
    dengan operasi morfologi (closing, opening, hole filling, dan luas komponen minimum).
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if not matched_pairs:
        return mask

    # 1. Menandai piksel blok pasangan yang cocok
    for idx_a, idx_b in matched_pairs:
        ya, xa = coords[idx_a]
        yb, xb = coords[idx_b]
        mask[ya : ya + block_size, xa : xa + block_size] = 255
        mask[yb : yb + block_size, xb : xb + block_size] = 255

    # 2. Morfologi: Closing (menutup celah antar blok yang berdekatan)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)

    # 3. Morfologi: Opening (membuang blok isolasi kecil)
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    # 4. Hole Filling (mengisi rongga di dalam area yang terdeteksi)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, 255, -1)

    # 5. Penapisan luas komponen minimum (Area Thresholding)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= area_min:
            clean_mask[labels == i] = 255

    return clean_mask

def extract_boundary(mask: np.ndarray) -> np.ndarray:
    """
    Mengekstrak garis tepi area biner dengan rumus materi PCD:
    beta(A) = A - (A (-) SE)
    """
    if np.sum(mask) == 0:
        return np.zeros_like(mask)
        
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(mask, se)
    boundary = cv2.subtract(mask, eroded)
    return boundary
