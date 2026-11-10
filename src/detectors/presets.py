"""Image-aware defaults for the interactive detector settings."""

import numpy as np


def image_adaptive_preset(image: np.ndarray, feature: str) -> dict:
    """Return practical starting parameters based on image scale and texture."""
    if image is None or image.size == 0 or image.dtype != np.uint8:
        raise ValueError("image must be a nonempty uint8 array")
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    else:
        raise ValueError("image must be grayscale or RGB")

    height, width = gray.shape
    short_side = min(height, width)
    sample_step = max(1, int(np.ceil(max(height, width) / 512)))
    sample = gray[::sample_step, ::sample_step].astype(np.float32)
    if min(sample.shape) > 1:
        grad_y, grad_x = np.gradient(sample)
        texture = float(np.median(np.hypot(grad_x, grad_y)))
    else:
        texture = 0.0
    contrast = float(np.std(sample))

    if short_side < 256:
        block_size, stride = 8, 1
    elif short_side < 640:
        block_size, stride = 12, 2
    elif short_side < 1600:
        block_size, stride = 16, 2
    else:
        block_size, stride = 24, 3

    threshold_t = 6 if short_side < 512 else (8 if short_side < 1200 else 12)
    if texture > 10:
        threshold_t += 2
    elif texture < 1.5:
        threshold_t = max(4, threshold_t - 2)

    result = {
        "block_size": block_size,
        "stride": stride,
        "d_min": int(np.clip(round(short_side * 0.06), 10, 100)),
        "threshold_t": threshold_t,
        "min_std": float(np.clip(round(contrast * 0.08), 2, 15)),
        "min_grad_2d": float(np.clip(round(texture * 0.4 * 2) / 2, 0.5, 6)),
        "area_min": int(np.clip(round(height * width * 0.00005 / 20) * 20, 20, 2000)),
        "n_features": int(np.clip(round(height * width / (640 * 480) * 3000), 1000, 12000)),
        "contrast_threshold": float(0.01 if texture < 1.5 else (0.02 if texture > 10 else 0.015)),
        "ransac_error": float(np.clip(round(short_side * 0.003 * 2) / 2, 2, 6)),
        "texture": texture,
        "short_side": short_side,
    }

    if feature == "SIFT":
        result["ratio_threshold"] = 0.75
    elif feature == "KP":
        result.update(ratio_threshold=0.78, block_size=16)
    return result
