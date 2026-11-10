"""Select and run a block feature extraction algorithm."""
import numpy as np
from ..preprocessing import compute_bilateral_multiscale_detail
from ..features import (
    extract_feature_a, extract_feature_b, extract_feature_b_local,
    extract_feature_c, extract_feature_d, extract_feature_h,
    extract_feature_dct, extract_feature_dct_bilateral,
)

def extract_block_features(feature_type, blocks, coords, gray_image, block_size, config):
    if feature_type == "DCT_BILATERAL":
        detail = compute_bilateral_multiscale_detail(
            gray_image, d=int(config.get("bilateral_d", 3)),
            sigma_space=float(config.get("bilateral_sigma_s", 1.0)),
            sigma_color=float(config.get("bilateral_sigma_r", 25.5)))
        detail_blocks = np.stack([detail[y:y + block_size, x:x + block_size]
                                  for y, x in coords])
        return extract_feature_dct_bilateral(
            blocks, detail_blocks, p_ratio=float(config.get("p_ratio", .25)))
    if feature_type == "DCT":
        return extract_feature_dct(blocks, n_coeffs=int(config.get("n_coeffs", 16)),
                                   include_dc=bool(config.get("include_dc", False)))
    if feature_type == "A":
        return extract_feature_a(blocks, g_sub=int(config.get("g_sub", 4)))
    if feature_type == "B":
        context = config.get("gaussian_context", "block")
        sigma = float(config.get("gaussian_sigma", 1.))
        samples = int(config.get("g_samples", 4))
        if context == "block":
            return extract_feature_b_local(blocks, sigma=sigma, g_samples=samples)
        if context == "image":
            return extract_feature_b(gray_image, coords, block_size=block_size,
                                     sigma=sigma, g_samples=samples)
        raise ValueError("gaussian_context must be block or image")
    if feature_type == "C":
        return extract_feature_c(blocks, low_freq_l=int(config.get("low_freq_l", 4)),
                                 normalize=bool(config.get("fft_normalize", False)))
    if feature_type == "D":
        return extract_feature_d(blocks, radial_bins=int(config.get("radial_bins", block_size // 2)),
                                 normalize=bool(config.get("fft_normalize", False)))
    if feature_type == "H":
        return extract_feature_h(blocks)
    raise ValueError(f"Tipe fitur '{feature_type}' tidak dikenali.")
