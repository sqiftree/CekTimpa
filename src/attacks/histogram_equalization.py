"""Image degradation algorithm."""
import cv2

def apply_histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Menerapkan perataan histogram (histogram equalization).
    Untuk citra berwarna (RGB), equalization diterapkan pada channel Luminance (YCrCb)
    agar tidak merusak chromaticity warna asli.
    """
    if image.ndim == 2:
        return cv2.equalizeHist(image)

    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
