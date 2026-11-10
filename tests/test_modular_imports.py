"""Public and focused module imports remain available after modularization."""
from src.features import extract_feature_a, extract_feature_dct
from src.features.feature_a import extract_feature_a as extract_feature_a_module
from src.features.feature_dct import extract_feature_dct as extract_feature_dct_module
from src.preprocessing import to_grayscale
from src.preprocessing.grayscale import to_grayscale as to_grayscale_module
from src.preprocessing.filters import apply_prefilter
from src.preprocessing.median_filter import apply_median_prefilter
from src.preprocessing.gaussian_filter import apply_gaussian_prefilter
from src.matching import match_features
from src.matching.kdtree import find_candidates as kdtree_candidates
from src.matching.lexsort import find_candidates as lexsort_candidates
from src.matching.shift_voting import select_winners
from src.attacks import apply_gaussian_noise
from src.attacks.noise_gaussian import apply_gaussian_noise as gaussian_noise_module
from src.metrics import calculate_psnr
from src.metrics.psnr import calculate_psnr as psnr_module


def test_legacy_feature_exports_match_focused_modules():
    assert extract_feature_a is extract_feature_a_module
    assert extract_feature_dct is extract_feature_dct_module


def test_legacy_preprocessing_exports_match_focused_modules():
    assert to_grayscale is to_grayscale_module


def test_focused_method_modules_and_legacy_facades_are_available():
    assert callable(apply_prefilter)
    assert callable(apply_median_prefilter)
    assert callable(apply_gaussian_prefilter)
    assert callable(match_features)
    assert callable(kdtree_candidates)
    assert callable(lexsort_candidates)
    assert callable(select_winners)
    assert apply_gaussian_noise is gaussian_noise_module
    assert calculate_psnr is psnr_module
