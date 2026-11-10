"""Filter selection compatibility API."""
from .gaussian_filter import apply_gaussian_prefilter
from .median_filter import apply_median_prefilter
from .bilateral_filter import apply_bilateral_filter
from .bilateral_detail import compute_bilateral_multiscale_detail

def apply_prefilter(image, filter_type="none", **kwargs):
    if filter_type == "median":
        return apply_median_prefilter(image, kwargs.get("ksize", 3))
    if filter_type == "gaussian":
        return apply_gaussian_prefilter(image, kwargs.get("sigma", 1.0))
    return image
