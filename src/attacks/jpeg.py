"""Image degradation algorithm."""
import cv2

def apply_jpeg_compression(image: np.ndarray, quality: int = 75) -> np.ndarray:
    """
    Menerapkan simulasi kompresi JPEG lossy dengan encode/decode in-memory.
    quality: tingkat kualitas JPEG (1-100, semakin kecil semakin terkompresi).
    """
    quality = max(1, min(100, int(quality)))
    is_rgb = (image.ndim == 3)
    bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if is_rgb else image

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode(".jpg", bgr_img, encode_param)
    decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR if is_rgb else cv2.IMREAD_GRAYSCALE)

    if is_rgb:
        return cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)
    return decimg
