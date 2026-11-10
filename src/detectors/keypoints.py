"""Conventional SIFT/ORB keypoint copy-move candidate detection."""
from typing import Dict, List, Tuple
import cv2
import numpy as np
from ..geometric import verify_similarity_pairs

def _detect_keypoints_conventional(
    gray_img: np.ndarray,
    block_size: int = 16,
    d_min: float = 30.0,
    threshold_t: int = 8,
    shift_tol: int = 4,
    reprojection_error: float = 3.0,
    diagnostics: Dict[str, int] = None,
    ratio_threshold: float = None,
    n_features: int = 4000,
    contrast_threshold: float = .015
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    """
    Deteksi Copy-Move berbasis titik kunci (Keypoint SIFT/ORB) murni matematika konvensional.
    Pencocokan deskriptor disaring dengan ratio test dan RANSAC similarity.
    Mask hanya menandai lingkungan keypoint, bukan segmentasi objek lengkap.
    """
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=contrast_threshold, edgeThreshold=8)
        norm_type = cv2.NORM_L2
        ratio_thresh = 0.78
    else:
        detector = cv2.ORB_create(nfeatures=n_features, scaleFactor=1.15, nlevels=8)
        norm_type = cv2.NORM_HAMMING
        ratio_thresh = 0.80

    if ratio_threshold is not None:
        ratio_thresh = float(ratio_threshold)
    if not 0 < ratio_thresh < 1 or n_features < 1 or contrast_threshold <= 0:
        raise ValueError("invalid keypoint ratio/count/contrast")
    keypoints, descriptors = detector.detectAndCompute(gray_img, None)
    if diagnostics is not None:
        diagnostics.update(keypoints=len(keypoints), candidate_pairs=0,
                           verified_pairs=0, models=0)
    if descriptors is None or len(keypoints) < 4:
        return [], np.empty((0, 2), dtype=np.int32), np.empty((0, 2), dtype=np.int32)

    half_b = block_size // 2
    h, w = gray_img.shape[:2]
    coords = np.array([
        [
            int(np.clip(kp.pt[1] - half_b, 0, max(0, h - block_size))),
            int(np.clip(kp.pt[0] - half_b, 0, max(0, w - block_size)))
        ]
        for kp in keypoints
    ], dtype=np.int32)

    matcher = cv2.BFMatcher(norm_type)
    knn_matches = matcher.knnMatch(descriptors, descriptors, k=min(64, len(descriptors)))

    d_min_sq = float(d_min) ** 2
    candidate_pairs = []
    seen_pairs = set()

    for m_list in knn_matches:
        # Self-match dibuang berdasarkan indeks, bukan posisi dalam hasil.
        valid_others = [m for m in m_list if m.queryIdx != m.trainIdx and (
            (coords[m.queryIdx, 0] - coords[m.trainIdx, 0]) ** 2 +
            (coords[m.queryIdx, 1] - coords[m.trainIdx, 1]) ** 2 >= d_min_sq
        )]
        if len(valid_others) < 2:
            continue

        best_m, second_m = valid_others[0], valid_others[1]
        if second_m.distance > 1e-6 and best_m.distance < ratio_thresh * second_m.distance:
            i, j = int(best_m.queryIdx), int(best_m.trainIdx)
            idx_a, idx_b = (i, j) if i < j else (j, i)
            if (idx_a, idx_b) in seen_pairs:
                continue
            seen_pairs.add((idx_a, idx_b))
            candidate_pairs.append((idx_a, idx_b))

    points_xy = np.float32([kp.pt for kp in keypoints])
    matched_pairs, model_shifts = verify_similarity_pairs(
        candidate_pairs, points_xy, min_inliers=max(4, int(threshold_t)),
        reprojection_error=reprojection_error)
    if diagnostics is not None:
        diagnostics.update(candidate_pairs=len(candidate_pairs),
                           verified_pairs=len(matched_pairs), models=len(model_shifts))
    return matched_pairs, model_shifts, coords
