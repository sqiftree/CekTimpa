"""Unified classical detector interface, combining main and alternate."""
from .base import BaseDetector, BaseDetectionResult
from .copymove_block import (detect_copymove_block, DetectionResult,
                             BlockDetectionResult, BlockDetector, BilateralDetector)
from .copymove_sift import detect_copymove_sift, SIFTDetectionResult, SIFTDetector

DETECTOR_REGISTRY = {"sift": SIFTDetector, "block": BlockDetector,
                     "bilateral": BilateralDetector, "dct_bilateral": BilateralDetector}


def get_detector(method="sift", default_config=None):
    key = method.lower().strip()
    if key not in DETECTOR_REGISTRY:
        raise ValueError(f"Unknown detector: {method}")
    return DETECTOR_REGISTRY[key](default_config=default_config)


def detect_copymove(image, method="sift", config=None):
    return get_detector(method).detect(image, config)
