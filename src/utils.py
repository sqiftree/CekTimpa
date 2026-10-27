"""
Modul Utilitas & Helper Visualisasi
====================================
Menyediakan fungsi-fungsi bantu untuk:
  - Konversi citra (grayscale, pixelation)
  - Visualisasi hasil deteksi (overlay, shift lines)
"""

from __future__ import annotations

import numpy as np
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# Image Conversion Helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Mengonversi citra RGB ke grayscale menggunakan luminance ITU-R 601.
      Y = 0.299·R + 0.587·G + 0.114·B

    Args:
        image: Citra RGB uint8 (H, W, 3) atau sudah grayscale (H, W)

    Returns:
        Citra grayscale uint8 (H, W)
    """
    if image.ndim == 2:
        return image.copy()
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    gray = 0.299 * r.astype(np.float32) + 0.587 * g.astype(np.float32) + 0.114 * b.astype(np.float32)
    return np.clip(gray, 0, 255).astype(np.uint8)


def pixelate(image: np.ndarray, block_size: int = 16) -> np.ndarray:
    """
    Menerapkan efek pikselasi (mosaic/pixelation) pada citra dengan merata-rata
    setiap blok berukuran block_size × block_size menjadi satu warna solid.

    Args:
        image     : Citra RGB uint8 (H, W, 3) atau grayscale (H, W)
        block_size: Ukuran blok pikselasi dalam piksel (default 16)

    Returns:
        Citra hasil pikselasi dengan shape dan dtype sama dengan masukan
    """
    h, w = image.shape[:2]
    block_size = max(1, int(block_size))
    output = image.copy()

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            y_end = min(y + block_size, h)
            x_end = min(x + block_size, w)
            block = image[y:y_end, x:x_end]
            avg_color = block.mean(axis=(0, 1)).astype(np.uint8)
            output[y:y_end, x:x_end] = avg_color

    return output


# ─────────────────────────────────────────────────────────────────────────────
# Visualization Helpers
# ─────────────────────────────────────────────────────────────────────────────

def create_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    boundary: np.ndarray,
    alpha: float = 0.4,
    border_thickness: int = 3,
) -> np.ndarray:
    """
    Membuat visualisasi dengan garis batas merah tebal di tepi area manipulasi.
    Tidak ada pengisian area (fill) — hanya kontur merah agar konten
    citra asli di dalam area tetap terlihat jelas.

    Args:
        image_rgb       : Citra masukan RGB (H, W, 3)
        mask            : Mask biner area terdeteksi (H, W) uint8
        boundary        : Tidak digunakan (dipertahankan untuk kompatibilitas API)
        alpha           : Tidak digunakan (dipertahankan untuk kompatibilitas API)
        border_thickness: Ketebalan garis batas merah dalam piksel (default 3)

    Returns:
        Citra RGB dengan kontur merah pada area terdeteksi
    """
    output = image_rgb.copy()

    if np.any(mask > 0):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, (255, 0, 0), thickness=border_thickness, lineType=cv2.LINE_AA)

    return output


def draw_shift_lines(
    image_rgb: np.ndarray,
    matched_pairs: list,
    coords: np.ndarray,
    block_size: int = 16,
    max_lines: int = 100,
) -> np.ndarray:
    """
    Menggambar garis penghubung antar pasangan blok kembar yang terdeteksi.

    Args:
        image_rgb    : Citra masukan RGB (H, W, 3)
        matched_pairs: Daftar tuple (idx_a, idx_b) pasangan blok kembar
        coords       : Array koordinat (N, 2) berisi (y, x) tiap blok
        block_size   : Ukuran blok untuk menghitung titik pusat (default 16)
        max_lines    : Batas maksimum garis yang digambar (default 100)

    Returns:
        Citra RGB dengan garis penghubung antar blok kembar
    """
    output = image_rgb.copy()
    step = max(1, len(matched_pairs) // max_lines)
    half_b = block_size // 2

    for idx_a, idx_b in matched_pairs[::step]:
        ya, xa = coords[idx_a]
        yb, xb = coords[idx_b]
        pt_a = (int(xa + half_b), int(ya + half_b))
        pt_b = (int(xb + half_b), int(yb + half_b))
        cv2.line(output, pt_a, pt_b, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(output, pt_a, 2, (0, 255, 0), -1)
        cv2.circle(output, pt_b, 2, (255, 0, 0), -1)

    return output
