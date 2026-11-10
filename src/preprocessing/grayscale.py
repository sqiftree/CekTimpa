"""Grayscale method selector and compatibility API."""
from .intensity import to_intensity_grayscale
from .luminance import to_luminance_grayscale
from .hsi import rgb_to_hsi

def to_grayscale(image_rgb, method="intensity"):
    if method == "intensity":
        return to_intensity_grayscale(image_rgb)
    return to_luminance_grayscale(image_rgb)

__all__=["to_grayscale","rgb_to_hsi","to_intensity_grayscale","to_luminance_grayscale"]
