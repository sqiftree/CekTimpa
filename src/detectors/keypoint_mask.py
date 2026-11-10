"""Build a mask from verified SIFT keypoint support."""
import cv2
import numpy as np
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
        mask = clean

    return mask
