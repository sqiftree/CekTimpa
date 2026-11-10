"""ITU-R 601 luminance grayscale conversion."""
import numpy as np

def to_luminance_grayscale(image_rgb):
    if image_rgb.ndim == 2:
        return image_rgb.astype(np.uint8)
    img = image_rgb.astype(np.float32)
    gray = .299*img[:,:,0] + .587*img[:,:,1] + .114*img[:,:,2]
    return np.clip(gray, 0, 255).astype(np.uint8)
