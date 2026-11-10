"""Describe the type and parameters of a fitted transform."""
import numpy as np
from typing import Optional
def _classify_transform(H: Optional[np.ndarray]) -> str:
    """
    Mengklasifikasikan jenis transformasi geometri dari matriks homografi H.

    Kategori:
      Translasi Murni : H ≈ [[1,0,tx],[0,1,ty],[0,0,1]]
      Affine          : H[2] ≈ [0,0,1] (tanpa perspektif)
      Proyektif planar  : H[2] ≠ [0,0,1] (ada distorsi perspektif)
    """
    if H is None:
        return "Tidak Terdeteksi"

    h11, h12 = float(H[0, 0]), float(H[0, 1])
    h21, h22 = float(H[1, 0]), float(H[1, 1])
    h31, h32 = float(H[2, 0]), float(H[2, 1])

    # Komponen perspektif
    if abs(h31) > 1e-4 or abs(h32) > 1e-4:
        return "Homografi planar"

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
