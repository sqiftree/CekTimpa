"""Geometric model fitting and correspondence verification."""
import cv2
import numpy as np
from .similarity import estimate as estimate_similarity
from .homography import estimate as estimate_homography

def verify_geometric_pairs(pairs, points_xy, min_inliers=6, reprojection_error=3.0,
                            max_models=8, model="similarity"):
    """Return unique verified pairs and representative (dy, dx) per model.

    Spatial orientation avoids arbitrary descriptor-index direction. Fit both
    directions to retain correspondences when spatial ordering flips. Require
    non-collinear, distinct physical locations on both sides of each model.
    Returned shifts summarize displacement; they are NOT affine parameters.
    """
    if model not in {"similarity", "homography"}:
        raise ValueError("model must be similarity or homography")
    if min_inliers < 4 or reprojection_error <= 0:
        raise ValueError('min_inliers >= 4 and reprojection_error > 0 required')
    unique = sorted({(min(i, j), max(i, j)) for i, j in pairs if i != j})
    remaining = []
    for i, j in unique:
        a, b = points_xy[i], points_xy[j]
        remaining.append((i, j) if tuple(a) < tuple(b) else (j, i))
    accepted, shifts, matrices = [], [], []
    for _ in range(max_models):
        if len(remaining) < min_inliers:
            break
        # Include inverse direction because rotations can reverse spatial order.
        directed = remaining + [(j, i) for i, j in remaining]
        source = np.float32([points_xy[i] for i, _ in directed])
        target = np.float32([points_xy[j] for _, j in directed])
        estimator = estimate_similarity if model == "similarity" else estimate_homography
        matrix, inliers = estimator(source, target, reprojection_error)
        if matrix is None or inliers is None or not np.isfinite(matrix).all():
            break
        projected = cv2.perspectiveTransform(source.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        keep = inliers.ravel().astype(bool) & (
            np.linalg.norm(projected - target, axis=1) <= reprojection_error)
        selected = np.flatnonzero(keep)
        selected_pairs = {tuple(sorted(directed[k])) for k in selected}
        # Remove this hypothesis even when rejected, then inspect remaining ones.
        remaining = [(i, j) for i, j in remaining
                     if tuple(sorted((i, j))) not in selected_pairs]
        if len(selected_pairs) < min_inliers:
            break
        # Reject collapsed/extreme transforms. Homography is a planar model,
        # not evidence of arbitrary 3D viewpoint invariance.
        area_source = cv2.contourArea(cv2.convexHull(source[keep]))
        area_target = cv2.contourArea(cv2.convexHull(target[keep]))
        if area_source <= 0 or not .25 <= area_target / area_source <= 4.0:
            continue
        valid = True
        for points in (source[keep], target[keep]):
            locations = np.unique(np.rint(points / 2).astype(int), axis=0)
            if len(locations) < min_inliers:
                valid = False
                break
            # Eigenvalues measure spatial spread, not descriptor similarity.
            minor = np.linalg.eigvalsh(np.cov(points.T))[0]
            if minor < 4:
                valid = False
                break
        if valid:
            accepted.extend(sorted(selected_pairs))
            matrices.append(matrix)
            dy_dx = np.median(target[keep] - source[keep], axis=0)[::-1]
            shifts.append(np.rint(dy_dx).astype(np.int32))
    return (sorted(set(accepted)), np.asarray(shifts, dtype=np.int32).reshape(-1, 2),
            matrices)


def verify_similarity_pairs(pairs, points_xy, min_inliers=6, reprojection_error=3.0,
                            max_models=8):
    """Backward-compatible similarity-only interface."""
    accepted, shifts, _ = verify_geometric_pairs(
        pairs, points_xy, min_inliers, reprojection_error, max_models)
    return accepted, shifts
