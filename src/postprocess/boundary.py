"""Extract object boundaries from binary masks."""
import numpy as np
import cv2
def extract_boundary(mask: np.ndarray) -> np.ndarray:
    """
    Mengekstrak garis tepi area biner dengan rumus materi PCD:
    beta(A) = A - (A (-) SE)
    """
    if np.sum(mask) == 0:
        return np.zeros_like(mask)
        
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(mask, se)
    boundary = cv2.subtract(mask, eroded)
    return boundary
