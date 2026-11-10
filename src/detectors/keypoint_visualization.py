"""Draw matched keypoints for detector explanations."""
import cv2
import numpy as np
from typing import Tuple
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
      Merah  = endpoint A (queryIdx), tidak menentukan sumber asli
      Hijau  = endpoint B (trainIdx), tidak menentukan target manipulasi
      Kuning = garis penghubung antar pasangan
    """
    vis = image.copy()
    if not matches:
        return vis

    step = max(1, int(np.ceil(len(matches) / max_lines)))

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
