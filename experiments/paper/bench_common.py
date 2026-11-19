"""Komponen bersama untuk eksperimen paper: dataset, metode, kondisi uji.

Empat metode yang dibandingkan:
  SIFT  : keypoint + FLANN + RANSAC (acuan utama)
  A     : fitur spasial, rata-rata sub-blok (box filter)
  B     : fitur spasial, Gaussian lowpass + subsampling
  DCT   : fitur frekuensi, koefisien DCT frekuensi rendah

Ground truth hanya dipakai SESUDAH deteksi untuk menghitung metrik.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from experiments.paper.attacks import (apply_gamma_correction, apply_gaussian_blur,
                         apply_gaussian_noise, apply_histogram_equalization,
                         apply_jpeg_compression, apply_salt_pepper_noise)

METHODS = ["SIFT", "A", "B", "DCT"]

# Parameter awal. BELUM dikalibrasi: kalibrasi dilakukan pada subset 'tune'
# (lihat --subset tune dan skenario e2), lalu dibekukan sebelum subset 'test'.
# Timpa lewat --config-json tanpa mengubah kode.
# stride=1 wajib untuk perbandingan adil: dengan stride>1 blok sumber dan salinan hanya
# sejajar jika vektor geser kelipatan stride (diuji: stride 2 gagal pada geser ganjil).
BLOCK_BASE = dict(block_size=16, stride=1, d_min=30, threshold_t=8,
                  area_min=100, min_std=2.0, eps=1.0, shift_tol=1,
                  match_method="lexsort")
DEFAULT_CONFIG: Dict[str, dict] = {
    "SIFT": dict(ratio_threshold=0.75, d_min=30, min_inliers=6, area_min=100),
    "A": dict(BLOCK_BASE, feature="A", g_sub=4),
    "B": dict(BLOCK_BASE, feature="B", gaussian_sigma=1.0, g_samples=4),
    "DCT": dict(BLOCK_BASE, feature="DCT", n_coeffs=16),
}


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
@dataclass
class Sample:
    sample_id: str
    image: np.ndarray            # RGB uint8
    label: int                   # 1 = forged, 0 = authentic
    gt_mask: np.ndarray          # 0/255, sumber + salinan (kosong jika label 0)


def _procedural_texture(seed: int, size: int = 512) -> np.ndarray:
    """Citra sumber sintetis bertekstur multi-skala (BUKAN foto nyata)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((size, size, 3), np.float32)
    for sigma, amp in [(2, 40), (6, 50), (18, 60), (50, 50)]:
        noise = rng.normal(0, 1, (size, size, 3)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), sigma)
        noise /= noise.std() + 1e-6
        out += amp * noise
    return np.clip(128 + out, 0, 255).astype(np.uint8)


def _forge_translation(img: np.ndarray, rng: np.random.Generator,
                       patch: int = 80) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    for _ in range(100):
        y1, x1 = rng.integers(0, h - patch), rng.integers(0, w - patch)
        y2, x2 = rng.integers(0, h - patch), rng.integers(0, w - patch)
        if abs(y1 - y2) > patch + 40 or abs(x1 - x2) > patch + 40:
            break
    forged = img.copy()
    forged[y2:y2 + patch, x2:x2 + patch] = img[y1:y1 + patch, x1:x1 + patch]
    gt = np.zeros((h, w), np.uint8)
    gt[y1:y1 + patch, x1:x1 + patch] = 255
    gt[y2:y2 + patch, x2:x2 + patch] = 255
    return forged, gt


def load_synthetic(n: int, seed: int, sources_dir: Optional[Path] = None) -> List[Sample]:
    """n sumber -> 2n sampel (satu authentic, satu forged per sumber).

    Dengan sources_dir, foto nyata di folder itu dipotong 512x512 sebagai sumber;
    tanpa itu dipakai tekstur prosedural.
    """
    samples: List[Sample] = []
    files: List[Path] = []
    if sources_dir:
        files = sorted(p for p in Path(sources_dir).iterdir()
                       if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})
    for i in range(n):
        rng = np.random.default_rng(seed + i)
        if files:
            bgr = cv2.imread(str(files[i % len(files)]))
            if bgr is None:
                continue
            src = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if min(src.shape[:2]) < 512:
                continue
            y = int(rng.integers(0, src.shape[0] - 511))
            x = int(rng.integers(0, src.shape[1] - 511))
            src = src[y:y + 512, x:x + 512]
        else:
            src = _procedural_texture(seed + i)
        forged, gt = _forge_translation(src, rng)
        sid = f"syn{i:04d}"
        samples.append(Sample(f"{sid}_O", src, 0, np.zeros(gt.shape, np.uint8)))
        samples.append(Sample(f"{sid}_F", forged, 1, gt))
    return samples


IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _index_files(data_dir: Path) -> Dict[str, Path]:
    """Indeks semua citra di bawah data_dir berdasarkan nama tanpa ekstensi (huruf kecil).

    Tidak peduli struktur folder (datar, atau dipisah forged/original/masks).
    Varian pasca-proses seperti 073_F_JC4 punya nama berbeda sehingga tidak ikut terambil.
    """
    idx: Dict[str, Path] = {}
    for p in sorted(Path(data_dir).rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            idx[p.stem.lower()] = p
    return idx


def _read_mask(path: Optional[Path]) -> Optional[np.ndarray]:
    if path is None:
        return None
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if m is None else (m >= 128)


def load_comofod(data_dir: Path, mask_mode: str = "B") -> List[Sample]:
    """Baca CoMoFoD: NNN_F (palsu), NNN_O (asli), mask NNN_B dan/atau NNN_M.

    mask_mode: "B" (hanya _B), "M" (hanya _M), "union" (gabungan _B dan _M, bila keduanya
    menandai sumber dan salinan secara terpisah). Periksa dulu dengan check_masks.py.
    Nama berkas hanya dipakai untuk label ground truth, tidak untuk hasil detektor.
    """
    idx = _index_files(Path(data_dir))
    samples: List[Sample] = []
    for stem, fpath in idx.items():
        m = re.match(r"^(\d+)_f$", stem)
        if not m:
            continue
        num = m.group(1)
        mb, mm = _read_mask(idx.get(f"{num}_b")), _read_mask(idx.get(f"{num}_m"))
        if mask_mode == "B":
            mask = mb
        elif mask_mode == "M":
            mask = mm
        else:
            parts = [x for x in (mb, mm) if x is not None]
            mask = None
            for x in parts:
                if mask is None:
                    mask = x.copy()
                elif x.shape == mask.shape:
                    mask |= x
        forged = cv2.imread(str(fpath))
        if forged is None or mask is None:
            continue
        gt = np.where(mask, 255, 0).astype(np.uint8)
        if gt.shape != forged.shape[:2] or not gt.any():
            continue
        samples.append(Sample(f"{num}_F", cv2.cvtColor(forged, cv2.COLOR_BGR2RGB), 1, gt))
        orig_p = idx.get(f"{num}_o")
        if orig_p is not None:
            orig = cv2.imread(str(orig_p))
            if orig is not None and orig.shape == forged.shape:
                samples.append(Sample(f"{num}_O", cv2.cvtColor(orig, cv2.COLOR_BGR2RGB),
                                      0, np.zeros(gt.shape, np.uint8)))
    return samples


def subsample_sets(samples: List[Sample], n_sets: int, seed: int) -> List[Sample]:
    """Pilih n_sets set (pasangan _F/_O) SECARA ACAK dengan seed tetap, sebelum pembagian."""
    if n_sets <= 0:
        return samples
    keys = sorted({s.sample_id.rsplit("_", 1)[0] for s in samples})
    if n_sets >= len(keys):
        return samples
    chosen = {keys[i] for i in np.random.default_rng(seed + 7).permutation(len(keys))[:n_sets]}
    return [s for s in samples if s.sample_id.rsplit("_", 1)[0] in chosen]


def split_by_source(samples: List[Sample], subset: str, seed: int,
                    tune_frac: float = 0.3) -> List[Sample]:
    """Pisah 30% tuning / 70% testing berdasarkan sumber (par _O/_F tidak terpisah)."""
    if subset == "all":
        return samples
    keys = sorted({s.sample_id.rsplit("_", 1)[0] for s in samples})
    perm = np.random.default_rng(seed).permutation(len(keys))
    n_tune = max(1, int(round(tune_frac * len(keys))))
    tune_keys = {keys[i] for i in perm[:n_tune]}
    want_tune = subset == "tune"
    return [s for s in samples if (s.sample_id.rsplit("_", 1)[0] in tune_keys) == want_tune]


# --------------------------------------------------------------------------
# Kondisi uji (gangguan / prefilter / parameter)
# --------------------------------------------------------------------------
@dataclass
class Condition:
    scenario: str
    name: str                      # mis. "gaussian_noise"
    level: str                     # mis. "10" -> nilai parameter gangguan
    attack: str = "none"           # nama gangguan (lihat ATTACKS)
    attack_args: Dict[str, float] = field(default_factory=dict)
    prefilter: str = "none"        # none | median | gaussian (diterapkan ke semua metode)
    overrides: Dict[str, dict] = field(default_factory=dict)  # per metode


ATTACKS = {
    "none": lambda img, seed, **k: img,
    "gaussian_noise": lambda img, seed, sigma: apply_gaussian_noise(img, sigma=sigma, seed=seed),
    "salt_pepper": lambda img, seed, amount: apply_salt_pepper_noise(img, amount=amount, seed=seed),
    "gaussian_blur": lambda img, seed, sigma: apply_gaussian_blur(img, sigma=sigma),
    "jpeg": lambda img, seed, quality: apply_jpeg_compression(img, quality=int(quality)),
    "gamma": lambda img, seed, gamma: apply_gamma_correction(img, gamma=gamma),
    "hist_eq": lambda img, seed, **k: apply_histogram_equalization(img),
}


def apply_attack(cond: "Condition", img: np.ndarray, seed: int) -> np.ndarray:
    return ATTACKS[cond.attack](img, seed, **cond.attack_args)


def _prefilter(img: np.ndarray, kind: str) -> np.ndarray:
    if kind == "median":
        return cv2.medianBlur(img, 3)
    if kind == "gaussian":
        return cv2.GaussianBlur(img, (0, 0), 1.0)
    return img


def _pick(full, reduced, levels):
    return reduced if levels == "reduced" else full


def build_conditions(scenarios: List[str], levels: str = "full") -> List[Condition]:
    conds: List[Condition] = []
    for sc in scenarios:
        if sc == "e1":
            conds.append(Condition("e1", "clean", "0"))
        elif sc == "e2":
            for b in (8, 16, 32):
                ov = {m: dict(block_size=b) for m in ("A", "B", "DCT")}
                conds.append(Condition("e2", "block_size", str(b), overrides=ov))
            for t in (5, 10, 20, 40):
                ov = {m: dict(threshold_t=t) for m in ("A", "B", "DCT")}
                conds.append(Condition("e2", "threshold_t", str(t), overrides=ov))
            for r in (0.6, 0.7, 0.8):
                conds.append(Condition("e2", "sift_ratio", str(r),
                                       overrides={"SIFT": dict(ratio_threshold=r)}))
            for k in (4, 6, 10):
                conds.append(Condition("e2", "sift_min_inliers", str(k),
                                       overrides={"SIFT": dict(min_inliers=k)}))
        elif sc == "e3":
            for s in _pick((2, 5, 10, 20), (5, 20), levels):
                conds.append(Condition("e3", "gaussian_noise", str(s), "gaussian_noise", dict(sigma=s)))
            for a in _pick((0.01, 0.02, 0.05), (0.02, 0.05), levels):
                conds.append(Condition("e3", "salt_pepper", str(a), "salt_pepper", dict(amount=a)))
            for s in _pick((0.5, 1.0, 1.5, 2.0), (1.0, 2.0), levels):
                conds.append(Condition("e3", "gaussian_blur", str(s), "gaussian_blur", dict(sigma=s)))
            for q in _pick((90, 70, 50, 30), (70, 30), levels):
                conds.append(Condition("e3", "jpeg", str(q), "jpeg", dict(quality=q)))
        elif sc == "e4":
            for g in _pick((0.7, 1.3, 1.6), (0.7, 1.6), levels):
                conds.append(Condition("e4", "gamma", str(g), "gamma", dict(gamma=g)))
            conds.append(Condition("e4", "hist_eq", "1", "hist_eq"))
        elif sc == "e5":
            noises = [("salt_pepper", "0.02", dict(amount=0.02)),
                      ("gaussian_noise", "10", dict(sigma=10))]
            for nm, lv, kw in noises:
                for pf in ("none", "median", "gaussian"):
                    conds.append(Condition("e5", f"{nm}+{pf}", lv, nm, kw, prefilter=pf))
        else:
            raise ValueError(f"skenario tidak dikenal: {sc}")
    return conds


def method_config(method: str, cond: Condition, base: Dict[str, dict]) -> dict:
    cfg = dict(base[method])
    cfg.update(cond.overrides.get(method, {}))
    return cfg
