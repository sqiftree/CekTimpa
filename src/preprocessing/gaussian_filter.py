"""Gaussian spatial prefilter."""
import cv2

def apply_gaussian_prefilter(image, sigma=1.0):
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
