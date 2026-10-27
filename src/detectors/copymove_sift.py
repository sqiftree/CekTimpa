"""
Pipeline Deteksi Pemalsuan Copy-Move Berbasis SIFT
===================================================
Metode Rujukan: Amerini, Ballan, Caldelli, Del Bimbo, Serra
"A SIFT-Based Forensic Method for Copy–Move Attack Detection and Transformation Recovery"
IEEE Transactions on Information Forensics and Security, 2011

Keunggulan vs DCT-Bilateral (Kaplan, 2020):
✅ Invariant terhadap ROTASI penuh 0°–360°
✅ Invariant terhadap SKALA (multi-scale DoG pyramid)
✅ Invariant terhadap pencahayaan dan perspektif 3D
✅ Toleran kompresi JPEG dan noise fotografi
✅ Cocok untuk foto objek fisik di dunia nyata

Pipeline 6 Langkah (Amerini dkk., 2011):
─────────────────────────────────────────
  L1: Grayscale Conversion
      Y = 0.299R + 0.587G + 0.114B

  L2: SIFT Keypoint Detection & Description
      Scale Space: L(x,y,σ) = G(x,y,σ) * I(x,y)
      DoG: D(x,y,σ) = L(x,y,kσ) − L(x,y,σ)
      Descriptor: 4×4 patch × 8-bin HOG = 128-dim L2-normalized

  L3: KNN Self-Matching (FLANN, k=3)
      Match setiap deskriptor d_i terhadap semua d_j (j≠i)

  L4: Lowe's Ratio Test + Spatial Filter
      d(m₁)/d(m₂) < r  AND  ‖p_i − p_j‖ ≥ d_min

  L5: Iterative RANSAC Geometric Verification
      Temukan semua H secara berulang — satu H per kelompok salinan independen

  L6: Mask Generation & Morphological Post-processing
      Lingkaran pada keypoint inlier → Closing → Area Threshold
"""

from __future__ import annotations

import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

from .base import BaseDetector, BaseDetectionResult
from ..postprocess import extract_boundary
from ..utils import create_overlay


# ─────────────────────────────────────────────────────────────────────────────
# Data Class Hasil Deteksi
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SIFTDetectionResult(BaseDetectionResult):
    """
    Hasil lengkap deteksi copy-move berbasis SIFT (Amerini dkk., 2011).
    Menyimpan visualisasi setiap langkah pipeline untuk keperluan edukasi dan forensik.
    """
    # ── Visualisasi Per Langkah (untuk UI explainer) ─────────────────────────
    step1_keypoints_img: Optional[np.ndarray] = None   # L2: Keypoints divisualisasi (rich keypoints)
    step2_candidates_img: Optional[np.ndarray] = None  # L3+L4a: Kandidat match setelah Lowe's Ratio Test
    step3_spatial_img: Optional[np.ndarray] = None     # L4b: Match setelah Spatial Distance Filter
    step4_ransac_img: Optional[np.ndarray] = None      # L5: Inlier match terverifikasi RANSAC

    # ── Statistik & Metadata SIFT ───────────────────────────────────────────
    n_keypoints: int = 0                  # Total keypoints terdeteksi (Langkah 2)
    n_candidates: int = 0                 # Kandidat matches (setelah ratio test)
    n_spatial: int = 0                    # Matches valid (setelah spatial filter)
    n_inliers: int = 0                    # Inlier RANSAC (matches terverifikasi)
    homography: Optional[np.ndarray] = None  # Matriks transformasi H pertama (3×3)
    transform_type: str = "—"             # Deskripsi jenis transformasi terdeteksi


class SIFTDetector(BaseDetector):
    """
    Detektor Copy-Move Berbasis SIFT + FLANN + Iterative RANSAC (Amerini dkk., 2011).
    """

    def __init__(self, default_config: Optional[Dict[str, Any]] = None):
        self.default_config = default_config or {}

    def detect(self, image: np.ndarray, config: Optional[Dict[str, Any]] = None) -> SIFTDetectionResult:
        cfg = {**self.default_config, **(config or {})}
        return detect_copymove_sift(image, cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Fungsi Bantuan Visualisasi
# ─────────────────────────────────────────────────────────────────────────────

def _draw_self_matches(
    image: np.ndarray,
    keypoints: list,
    matches: list,
    max_lines: int = 250,
    src_color: Tuple[int, int, int] = (255, 60, 0),
    dst_color: Tuple[int, int, int] = (0, 220, 50),
    line_color: Tuple[int, int, int] = (255, 200, 0),
) -> np.ndarray:
    """
    Menggambar garis penghubung antar pasangan keypoint yang cocok
    dalam satu citra (visualisasi self-matching).

    Warna default:
      Merah  = titik sumber (queryIdx) — area yang di-copy
      Hijau  = titik tujuan (trainIdx) — area tempat di-paste
      Kuning = garis penghubung antar pasangan
    """
    vis = image.copy()
    if not matches:
        return vis

    step = max(1, len(matches) // max_lines)

    # Gambar garis dulu (di bawah titik)
    for m in matches[::step]:
        pt1 = tuple(map(int, keypoints[m.queryIdx].pt))
        pt2 = tuple(map(int, keypoints[m.trainIdx].pt))
        cv2.line(vis, pt1, pt2, line_color, 1, cv2.LINE_AA)

    # Gambar titik di atas garis
    for m in matches[::step]:
        pt1 = tuple(map(int, keypoints[m.queryIdx].pt))
        pt2 = tuple(map(int, keypoints[m.trainIdx].pt))
        cv2.circle(vis, pt1, 4, src_color, -1)
        cv2.circle(vis, pt2, 4, dst_color, -1)

    return vis


def _classify_transform(H: Optional[np.ndarray]) -> str:
    """
    Mengklasifikasikan jenis transformasi geometri dari matriks homografi H.

    Kategori:
      Translasi Murni : H ≈ [[1,0,tx],[0,1,ty],[0,0,1]]
      Affine          : H[2] ≈ [0,0,1] (tanpa perspektif)
      Proyektif (3D)  : H[2] ≠ [0,0,1] (ada distorsi perspektif)
    """
    if H is None:
        return "Tidak Terdeteksi"

    h11, h12 = float(H[0, 0]), float(H[0, 1])
    h21, h22 = float(H[1, 0]), float(H[1, 1])
    h31, h32 = float(H[2, 0]), float(H[2, 1])

    # Komponen perspektif
    if abs(h31) > 1e-4 or abs(h32) > 1e-4:
        return "Proyektif / Perspektif 3D"

    # Rotasi + Skala (Affine)
    is_identity = (abs(h11 - 1) < 0.05 and abs(h12) < 0.05 and
                   abs(h21) < 0.05 and abs(h22 - 1) < 0.05)
    if is_identity:
        tx, ty = float(H[0, 2]), float(H[1, 2])
        return f"Translasi Murni  Δx={tx:+.1f}px  Δy={ty:+.1f}px"

    # Sudut rotasi dari kolom pertama matriks rotasi
    angle = float(np.degrees(np.arctan2(h21, h11)))
    scale = float(np.sqrt(h11 ** 2 + h21 ** 2))
    tx, ty = float(H[0, 2]), float(H[1, 2])
    return f"Affine  Rotasi={angle:+.1f}°  Skala={scale:.2f}×  Δx={tx:+.1f}px  Δy={ty:+.1f}px"


def _build_sift_mask(
    image_shape: tuple,
    keypoints: list,
    inlier_matches: list,
    kp_radius: int = 20,
    area_min: int = 100,
) -> np.ndarray:
    """
    Membangun mask biner dari posisi keypoint inlier RANSAC.

    Algoritma:
    1. Gambar lingkaran terisi di setiap keypoint inlier (radius ∝ skala keypoint)
    2. Morphological Closing (kernel ellipse 15×15) untuk menyambung celah
    3. Area Thresholding: buang komponen < area_min px²
    """
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for m in inlier_matches:
        pt_q = tuple(map(int, keypoints[m.queryIdx].pt))
        pt_t = tuple(map(int, keypoints[m.trainIdx].pt))
        # Radius proporsional dengan skala keypoint (size = diameter lingkaran DoG)
        r_q = max(kp_radius, int(keypoints[m.queryIdx].size * 0.5))
        r_t = max(kp_radius, int(keypoints[m.trainIdx].size * 0.5))
        cv2.circle(mask, pt_q, r_q, 255, -1)
        cv2.circle(mask, pt_t, r_t, 255, -1)

    # Morphological Closing: isi celah antar keypoint berdekatan
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Area Thresholding: hapus komponen terlalu kecil
    if np.any(mask > 0):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        clean = np.zeros_like(mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= area_min:
                clean[labels == i] = 255
        if np.any(clean > 0):
            mask = clean

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# Fungsi Utama Deteksi SIFT
# ─────────────────────────────────────────────────────────────────────────────

def detect_copymove_sift(
    image: np.ndarray,
    config: Dict[str, Any] = None,
) -> SIFTDetectionResult:
    """
    Mendeteksi pemalsuan copy-move menggunakan SIFT + FLANN + Iterative RANSAC.

    Urutan Langkah:
      L1 → Grayscale       : Konversi luminance ITU-R 601
      L2 → SIFT            : Scale-space extrema + 128-dim descriptor
      L3 → KNN Matching    : FLANN k=3 self-matching
      L4 → Filtering       : Lowe's Ratio Test + Spatial Distance Filter
      L5 → Iterative RANSAC: Temukan semua H secara berulang (multi-region)
      L6 → Mask            : Lingkaran keypoint → Morphology → Area filter

    Mengapa Iterative RANSAC?
      Single-shot RANSAC hanya menemukan SATU homografi H terbaik. Jika ada
      N salinan di posisi berbeda (N shift vector berbeda), hanya salinan dengan
      inlier terbanyak yang terdeteksi. Iterative RANSAC: temukan H₁ → hapus
      inlier → temukan H₂ → hapus inlier → ... → semua wilayah terdeteksi.

    Args:
        image  : Citra RGB uint8 shape (H, W, 3)
        config : Dict parameter (lihat default di bawah)

    Returns:
        SIFTDetectionResult dengan visualisasi per langkah dan statistik deteksi
    """
    if image is None or image.size == 0:
        raise ValueError("Citra masukan tidak boleh kosong (None / size 0).")

    if config is None:
        config = {}

    # ── Parameter Konfigurasi ─────────────────────────────────────────────────
    n_features       = int(config.get("n_features", 0))           # 0 = tidak dibatasi
    n_octave_layers  = int(config.get("n_octave_layers", 3))      # Jumlah layer DoG per oktaf
    contrast_thresh  = float(config.get("contrast_threshold", 0.04))  # Filter low-contrast keypoint
    edge_thresh      = float(config.get("edge_threshold", 10.0))  # Filter edge keypoint
    sigma            = float(config.get("sigma", 1.6))            # Gaussian σ awal
    ratio_thresh     = float(config.get("ratio_threshold", 0.75)) # Lowe's ratio threshold r
    d_min            = float(config.get("d_min", 30.0))           # Jarak spasial minimum (px)
    ransac_thresh    = float(config.get("ransac_threshold", 5.0)) # RANSAC reprojection error (px)
    min_inliers      = int(config.get("min_inliers", 10))         # Minimum inlier per iterasi H
    max_ransac_iter  = int(config.get("max_ransac_iter", 10))     # Maks iterasi RANSAC multi-region
    area_min         = int(config.get("area_min", 100))           # Minimum luas komponen mask (px²)
    kp_radius        = int(config.get("kp_radius", 18))           # Radius lingkaran mask (px)
    flann_trees      = int(config.get("flann_trees", 5))          # KD-tree count (FLANN)
    flann_checks     = int(config.get("flann_checks", 50))        # Search checks (FLANN)

    timings: Dict[str, float] = {}
    t_start = time.time()

    h, w = image.shape[:2]
    empty_mask = np.zeros((h, w), dtype=np.uint8)

    def _empty_result(kp_img=None, note="") -> SIFTDetectionResult:
        """Helper: kembalikan result kosong dengan pesan status."""
        base = image.copy()
        return SIFTDetectionResult(
            step1_keypoints_img  = kp_img if kp_img is not None else base,
            step2_candidates_img = base,
            step3_spatial_img    = base,
            step4_ransac_img     = base,
            mask          = empty_mask,
            boundary      = empty_mask,
            overlay       = base,
            n_keypoints   = 0,
            n_candidates  = 0,
            n_spatial     = 0,
            n_inliers     = 0,
            homography    = None,
            transform_type= note or "—",
            is_forged     = False,
            timings       = {**timings, "total_ms": (time.time() - t_start) * 1000},
        )

    # ══════════════════════════════════════════════════════════════════════════
    # LANGKAH 1: Konversi Grayscale (Luminance ITU-R 601)
    # Y = 0.299R + 0.587G + 0.114B
    # ══════════════════════════════════════════════════════════════════════════
    t0 = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()
    timings["L1_grayscale_ms"] = (time.time() - t0) * 1000

    # ══════════════════════════════════════════════════════════════════════════
    # LANGKAH 2: SIFT Keypoint Detection & Descriptor Computation
    # - Gaussian Scale Space: L(x,y,σ) = G(x,y,σ) ∗ I(x,y)
    # - DoG: D(x,y,σ) = L(x,y,kσ) − L(x,y,σ)
    # - Lokalisasi sub-piksel via Taylor expansion
    # - Orientasi dominan dari gradient histogram (36 bins)
    # - Descriptor: 4×4 patch × 8-bin gradient histogram = 128-dim vector
    # ══════════════════════════════════════════════════════════════════════════
    t0 = time.time()
    sift = cv2.SIFT_create(
        nfeatures         = n_features,
        nOctaveLayers     = n_octave_layers,
        contrastThreshold = contrast_thresh,
        edgeThreshold     = edge_thresh,
        sigma             = sigma,
    )
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    timings["L2_sift_ms"] = (time.time() - t0) * 1000

    n_kp = len(keypoints)

    # Visualisasi Langkah 2: Keypoints dengan ukuran (skala) dan orientasi
    kp_vis = cv2.drawKeypoints(
        cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB),
        keypoints, None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    if n_kp < 4 or descriptors is None:
        return _empty_result(kp_vis, f"Keypoints tidak cukup ({n_kp} < 4)")

    # ══════════════════════════════════════════════════════════════════════════
    # LANGKAH 3: KNN Self-Matching (FLANN, k=3)
    # Self-matching: cocokkan setiap deskriptor dᵢ terhadap semua dⱼ (j≠i)
    # k=3: indeks 0 = self-match (dibuang), 1 = best, 2 = second-best
    # FLANN (Fast Library for Approximate Nearest Neighbors): KD-tree O(N log N)
    # ══════════════════════════════════════════════════════════════════════════
    t0 = time.time()
    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=flann_trees),
        dict(checks=flann_checks),
    )
    raw_matches = flann.knnMatch(descriptors, descriptors, k=3)
    timings["L3_matching_ms"] = (time.time() - t0) * 1000

    # ══════════════════════════════════════════════════════════════════════════
    # LANGKAH 4a: Lowe's Ratio Test
    # Kriteria: d(m₁) / d(m₂) < r  (default r = 0.75)
    # Tujuan: memastikan m₁ jauh lebih baik dari m₂ (match diskriminatif)
    # ══════════════════════════════════════════════════════════════════════════
    t0 = time.time()
    ratio_passed = []
    for group in raw_matches:
        # Buang self-match (queryIdx == trainIdx)
        diff = [m for m in group if m.queryIdx != m.trainIdx]
        if len(diff) < 2:
            continue
        m, n = diff[0], diff[1]
        if m.distance < ratio_thresh * n.distance:
            ratio_passed.append(m)

    n_cand = len(ratio_passed)

    # Visualisasi Langkah 3: Candidate matches (oranye)
    cand_vis = _draw_self_matches(
        image, keypoints, ratio_passed, max_lines=250,
        src_color=(255, 100, 0),
        dst_color=(255, 160, 0),
        line_color=(255, 165, 0),
    )

    # ══════════════════════════════════════════════════════════════════════════
    # LANGKAH 4b: Spatial Distance Filter
    # Kriteria: ‖pᵢ − pⱼ‖₂ ≥ d_min  (default 30 px)
    # Tujuan: cegah pasangan keypoint bertetangga (bukan duplikasi, tapi lokal)
    # ══════════════════════════════════════════════════════════════════════════
    spatial_passed = []
    for m in ratio_passed:
        pt_q = np.array(keypoints[m.queryIdx].pt)
        pt_t = np.array(keypoints[m.trainIdx].pt)
        if np.linalg.norm(pt_q - pt_t) >= d_min:
            spatial_passed.append(m)

    n_spatial = len(spatial_passed)
    timings["L4_filter_ms"] = (time.time() - t0) * 1000

    # Visualisasi Langkah 4: Spatial-filtered matches (cyan)
    spatial_vis = _draw_self_matches(
        image, keypoints, spatial_passed, max_lines=250,
        src_color=(0, 150, 255),
        dst_color=(0, 220, 255),
        line_color=(0, 200, 255),
    )

    if n_spatial < min_inliers:
        return SIFTDetectionResult(
            step1_keypoints_img  = kp_vis,
            step2_candidates_img = cand_vis,
            step3_spatial_img    = spatial_vis,
            step4_ransac_img     = image.copy(),
            mask          = empty_mask,
            boundary      = empty_mask,
            overlay       = image.copy(),
            n_keypoints   = n_kp,
            n_candidates  = n_cand,
            n_spatial     = n_spatial,
            n_inliers     = 0,
            homography    = None,
            transform_type= f"Match tidak cukup setelah filter spasial ({n_spatial} < {min_inliers})",
            is_forged     = False,
            timings       = {**timings, "total_ms": (time.time() - t_start) * 1000},
        )

    # ══════════════════════════════════════════════════════════════════════════
    # LANGKAH 5: Iterative RANSAC Homography Estimation (Multi-Region)
    # Algoritma:
    #   1. Pilih 4 pasangan secara acak dari remaining_matches
    #   2. Estimasi H dengan Direct Linear Transform (DLT)
    #   3. Hitung reprojection error: eᵢ = ‖pⱼ − H·pᵢ‖
    #   4. Inlier: eᵢ < ransac_threshold  (default 5 px)
    #   5. Simpan inlier → hapus dari pool → ulangi untuk H berikutnya
    #
    # Mengapa iteratif?
    #   Single-shot RANSAC hanya mendeteksi SATU transformasi dominan.
    #   Jika ada N salinan di lokasi berbeda (masing-masing dengan shift vector
    #   berbeda), hanya salinan dengan inlier terbanyak yang terdeteksi.
    #   Iterative RANSAC mendeteksi SEMUA transformasi independen secara berurutan.
    # ══════════════════════════════════════════════════════════════════════════
    t0 = time.time()

    remaining_matches = list(spatial_passed)
    inlier_matches: List = []
    H: Optional[np.ndarray] = None  # H dari iterasi pertama (untuk laporan klasifikasi)

    for _iter in range(max_ransac_iter):
        if len(remaining_matches) < 4:
            break

        src_pts = np.float32([keypoints[m.queryIdx].pt for m in remaining_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in remaining_matches]).reshape(-1, 1, 2)

        H_iter, ransac_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
        if ransac_mask is None or H_iter is None:
            break

        iter_inliers  = [m for m, f in zip(remaining_matches, ransac_mask.ravel()) if f]
        iter_outliers = [m for m, f in zip(remaining_matches, ransac_mask.ravel()) if not f]

        if len(iter_inliers) < min_inliers:
            # Tidak cukup inlier — hentikan iterasi
            break

        # Simpan H pertama untuk laporan klasifikasi transformasi di UI
        if H is None:
            H = H_iter

        inlier_matches.extend(iter_inliers)
        remaining_matches = iter_outliers  # Lanjutkan dengan sisa outlier

    n_inliers = len(inlier_matches)
    transform_type = _classify_transform(H)
    timings["L5_ransac_ms"] = (time.time() - t0) * 1000

    # Visualisasi Langkah 5: RANSAC inliers (hijau = verified)
    ransac_vis = _draw_self_matches(
        image, keypoints, inlier_matches, max_lines=250,
        src_color=(0, 50, 220),
        dst_color=(0, 200, 50),
        line_color=(50, 255, 100),
    )

    if n_inliers < min_inliers or H is None:
        return SIFTDetectionResult(
            step1_keypoints_img  = kp_vis,
            step2_candidates_img = cand_vis,
            step3_spatial_img    = spatial_vis,
            step4_ransac_img     = ransac_vis,
            mask          = empty_mask,
            boundary      = empty_mask,
            overlay       = image.copy(),
            n_keypoints   = n_kp,
            n_candidates  = n_cand,
            n_spatial     = n_spatial,
            n_inliers     = n_inliers,
            homography    = H,
            transform_type= transform_type,
            is_forged     = False,
            timings       = {**timings, "total_ms": (time.time() - t_start) * 1000},
        )

    # ══════════════════════════════════════════════════════════════════════════
    # LANGKAH 6: Mask Generation & Morphological Post-processing
    # 1. Gambar lingkaran terisi pada posisi setiap keypoint inlier
    # 2. Morphological Closing (kernel ellipse 15×15)
    # 3. Area Thresholding: buang komponen < area_min px²
    # 4. Extract boundary: β(A) = A − (A ⊖ SE)
    # 5. Overlay warna merah pada citra asli
    # ══════════════════════════════════════════════════════════════════════════
    t0 = time.time()
    mask     = _build_sift_mask(image.shape, keypoints, inlier_matches, kp_radius, area_min)
    boundary = extract_boundary(mask)
    overlay  = create_overlay(image, mask, boundary)
    timings["L6_mask_ms"] = (time.time() - t0) * 1000
    timings["total_ms"]   = (time.time() - t_start) * 1000

    is_forged = bool(n_inliers >= min_inliers and np.any(mask > 0))

    return SIFTDetectionResult(
        step1_keypoints_img  = kp_vis,
        step2_candidates_img = cand_vis,
        step3_spatial_img    = spatial_vis,
        step4_ransac_img     = ransac_vis,
        mask          = mask,
        boundary      = boundary,
        overlay       = overlay,
        n_keypoints   = n_kp,
        n_candidates  = n_cand,
        n_spatial     = n_spatial,
        n_inliers     = n_inliers,
        homography    = H,
        transform_type= transform_type,
        is_forged     = is_forged,
        timings       = timings,
    )
