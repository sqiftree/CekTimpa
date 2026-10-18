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
