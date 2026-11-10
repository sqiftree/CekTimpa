"""Image attack methods, exported for backwards compatibility."""
from .noise_gaussian import apply_gaussian_noise
from .noise_salt_pepper import apply_salt_pepper_noise
from .blur_gaussian import apply_gaussian_blur
from .gamma import apply_gamma_correction
from .histogram_equalization import apply_histogram_equalization
from .jpeg import apply_jpeg_compression

__all__ = ["apply_gaussian_noise", "apply_salt_pepper_noise", "apply_gaussian_blur",
           "apply_gamma_correction", "apply_histogram_equalization", "apply_jpeg_compression"]
