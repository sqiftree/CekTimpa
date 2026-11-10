"""Median spatial prefilter."""
import cv2

def apply_median_prefilter(image, ksize=3):
    return cv2.medianBlur(image, ksize)
