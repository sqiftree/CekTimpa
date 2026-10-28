"""
Modul Generator Pemalsuan Copy-Move Sintetis & Ground Truth
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np

def make_forgery(
    image: np.ndarray,
    patch_size: int = 64,
    shape: str = "rect",
    rotate_deg: float = 0.0,
    scale_factor: float = 1.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Menghasilkan citra pemalsuan copy-move sintetis beserta Ground Truth Mask (GT).
    
    Returns:
        forged_img: Citra hasil salin-tempel
        gt_mask: Mask biner ground truth (area sumber + area salinan)
        metadata: Parameter lokasi dan transformasi penempelan
    """
    # TODO: Implementasi oleh mahasiswa
    raise NotImplementedError("Generator pemalsuan belum diimplementasikan")


def generate_synthetic_forgery(
    mode: str = "tampered",
    image_size: Tuple[int, int] = (256, 256),
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Menghasilkan citra sintetis untuk benchmark dan demonstrasi deteksi copy-move.

    Modes:
    - 'tampered' / 'Sampel 1': Memuat objek lingkaran bertekstur pada (32, 32)
      yang diduplikasi (copy-move) secara eksak ke (144, 144) dengan vektor translasi (112, 112).
    - 'clean_single' / 'Sampel 2': Citra asli otentik dengan satu objek pada (32, 32)
      tanpa manipulasi salin-tempel.
    - 'clean_multi' / 'Sampel 3': Citra asli otentik dengan dua objek berbeda bentuk,
      warna, dan frekuensi (lingkaran biru-cyan dan segitiga amber), tanpa duplikasi.

    Returns:
        image: np.ndarray shape (H, W, 3) bertipe uint8 (RGB)
        ground_truth: np.ndarray shape (H, W) bertipe uint8 (255 pada area copy-move, 0 lainnya)
        metadata: Dict berisi informasi koordinat objek, vektor pergeseran, dan deskripsi
    """
    h, w = image_size
    if min(h, w) < 192:
        raise ValueError("Synthetic demo requires image dimensions >= 192")
    rng = np.random.RandomState(seed)

    # 1. Background: Gradien kontinu lembut non-periodik + derau fotografi ringan
    # Latar belakang non-periodik mencegah false positive dari pola kisi berulang (periodic grid)
    y, x = np.ogrid[:h, :w]
    bg_r = np.clip(210 + 25 * (y / h) - 15 * (x / w), 0, 255)
    bg_g = np.clip(220 + 15 * (y / h) + 10 * (x / w), 0, 255)
    bg_b = np.clip(235 - 30 * (y / h) - 10 * (x / w), 0, 255)
    bg = np.stack([bg_r, bg_g, bg_b], axis=-1).astype(np.float32)

    noise = rng.normal(0, 1.5, (h, w, 3))
    image = np.clip(bg + noise, 0, 255).astype(np.uint8)

    # Ground truth mask (0 = area bersih, 255 = area copy-move)
    gt_mask = np.zeros((h, w), dtype=np.uint8)

    # 2. Objek 1: Lencana Lingkaran Bertekstur Kaya (Bilateral & DCT-friendly)
    # Variansi internal tinggi dan spektrum frekuensi luas
    obj_size = 48
    oy, ox = np.ogrid[:obj_size, :obj_size]
    cy, cx = obj_size / 2.0, obj_size / 2.0
    dist = np.sqrt((oy - cy) ** 2 + (ox - cx) ** 2)
    mask_circle = dist <= 22.0

    obj1 = np.zeros((obj_size, obj_size, 3), dtype=np.uint8)
    wave = (np.sin(dist * 0.4) * np.cos(ox * 0.3) * 60 + 128).astype(np.uint8)
    obj1[:, :, 0] = np.clip(wave * 0.2 + 20, 0, 255)   # Saluran R
    obj1[:, :, 1] = np.clip(wave * 0.6 + 60, 0, 255)   # Saluran G
    obj1[:, :, 2] = np.clip(wave * 1.0 + 120, 0, 255)  # Saluran B
    obj1_noise = rng.randint(-12, 12, (obj_size, obj_size, 3)).astype(np.int16)
    obj1 = np.clip(obj1.astype(np.int16) + obj1_noise, 0, 255).astype(np.uint8)

    # Lokasi Objek 1: (y1, x1) = (32, 32)
    y1, x1 = 32, 32
    for c in range(3):
        patch = image[y1 : y1 + obj_size, x1 : x1 + obj_size, c]
        patch[mask_circle] = obj1[:, :, c][mask_circle]
        image[y1 : y1 + obj_size, x1 : x1 + obj_size, c] = patch

    # Tentukan mode
    mode_lower = mode.lower()
    is_tampered = "sampel 1" in mode_lower or "tampered" in mode_lower or "duplikasi" in mode_lower
    is_multi = "sampel 3" in mode_lower or "multi" in mode_lower

    metadata: Dict[str, Any] = {
        "is_tampered": is_tampered,
        "object1_box": (x1, y1, obj_size, obj_size),
    }

    if is_tampered:
        # 3. Kasus Manipulasi Copy-Move:
        # Objek 1 disalin persis ke lokasi target (y2, x2) = (144, 144)
        y2, x2 = 144, 144
        dy, dx = y2 - y1, x2 - x1  # Translasi spasial: (112, 112)

        for c in range(3):
            patch_target = image[y2 : y2 + obj_size, x2 : x2 + obj_size, c]
            patch_target[mask_circle] = obj1[:, :, c][mask_circle]
            image[y2 : y2 + obj_size, x2 : x2 + obj_size, c] = patch_target

        # Tandai area sumber dan target pada ground truth mask
        gt_mask[y1 : y1 + obj_size, x1 : x1 + obj_size][mask_circle] = 255
        gt_mask[y2 : y2 + obj_size, x2 : x2 + obj_size][mask_circle] = 255

        metadata.update({
            "shift_vector": (dy, dx),
            "object2_box": (x2, y2, obj_size, obj_size),
            "description": f"🔴 Manipulasi Copy-Move: Objek lingkaran (32, 32) disalin ke ({y2}, {x2}) dengan pergeseran translasi Δ=({dy}, {dx}).",
        })

    elif is_multi:
        # Kasus Citra Asli dengan Multi-Objek Berbeda Bentuk & Warna
        # Objek 2 berupa segitiga amber-oranye bergradien non-periodik pada (144, 144)
        y2, x2 = 144, 144
        mask_tri = (oy >= 6) & (ox >= (oy - 6) * 0.5) & (ox <= obj_size - (oy - 6) * 0.5)

        obj2 = np.zeros((obj_size, obj_size, 3), dtype=np.uint8)
        obj2[:, :, 0] = np.clip(225 - 2 * oy + 1 * ox, 0, 255)
        obj2[:, :, 1] = np.clip(110 + 1 * oy - 1 * ox, 0, 255)
        obj2[:, :, 2] = np.clip(35 + 1 * oy, 0, 255)
        obj2_noise = rng.randint(-10, 10, (obj_size, obj_size, 3)).astype(np.int16)
        obj2 = np.clip(obj2.astype(np.int16) + obj2_noise, 0, 255).astype(np.uint8)

        for c in range(3):
            patch_target = image[y2 : y2 + obj_size, x2 : x2 + obj_size, c]
            patch_target[mask_tri] = obj2[:, :, c][mask_tri]
            image[y2 : y2 + obj_size, x2 : x2 + obj_size, c] = patch_target

        metadata.update({
            "shift_vector": None,
            "object2_box": (x2, y2, obj_size, obj_size),
            "description": "🟢 Citra Asli (Otentik Multi-Objek): Terdapat dua objek independen dengan bentuk dan warna berbeda. Tidak ada pemalsuan salin-tempel.",
        })

    else:
        # Kasus Citra Asli Objek Tunggal
        metadata.update({
            "shift_vector": None,
            "object2_box": None,
            "description": "🟢 Citra Asli (Otentik Objek Tunggal): Hanya memuat satu objek lingkaran bertekstur pada (32, 32) tanpa manipulasi salin-tempel.",
        })

    return image, gt_mask, metadata
