"""Arithmetic RGB intensity grayscale conversion."""
import numpy as np

def to_intensity_grayscale(image_rgb):
    if image_rgb.ndim == 2:
        return image_rgb.astype(np.uint8)
    gray = np.mean(image_rgb.astype(np.float32), axis=2)
    return np.clip(gray, 0, 255).astype(np.uint8)
