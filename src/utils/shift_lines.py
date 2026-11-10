"""Render matched block displacement lines."""
import numpy as np
import cv2
def draw_shift_lines(
    image_rgb: np.ndarray,
    matched_pairs: list,
    coords: np.ndarray,
    block_size: int = 16,
    max_lines: int = 100
) -> np.ndarray:
    """
    Menggambar garis penghubung antar pasangan blok kembar yang terdeteksi.
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
