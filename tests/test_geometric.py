import cv2
import numpy as np
import pytest
from src.geometric import verify_similarity_pairs
from src.detectors.copymove_block import detect_copymove_block


@pytest.mark.parametrize('angle,scale', [(0, 1), (20, 1.15), (-30, .8)])
def test_similarity_rejects_outliers(angle, scale):
    rng = np.random.default_rng(71)
    source = rng.uniform(10, 90, (20, 2)).astype(np.float32)
    matrix = cv2.getRotationMatrix2D((50, 50), angle, scale)
    matrix[:, 2] += [170, 80]
    target = source @ matrix[:, :2].T + matrix[:, 2]
    points = np.vstack([source, target, rng.uniform(0, 300, (10, 2))]).astype(np.float32)
    pairs = [(i, i + 20) for i in range(20)] + [(i, i + 40) for i in range(10)]
    cv2.setRNGSeed(42)
    accepted, shifts = verify_similarity_pairs(pairs, points, min_inliers=8, reprojection_error=1)
    assert set(accepted) == {(i, i + 20) for i in range(20)}
    assert len(shifts) == 1


def test_collinear_or_repeated_locations_are_not_evidence():
    source = np.float32([[x, 10] for x in range(0, 100, 10)])
    points = np.vstack([source, source + [150, 0]])
    accepted, _ = verify_similarity_pairs([(i, i + 10) for i in range(10)], points)
    assert accepted == []
    source = np.tile([[10, 10]], (10, 1))
    accepted, _ = verify_similarity_pairs([(i, i + 10) for i in range(10)],
                                         np.vstack([source, source + [150, 0]]))
    assert accepted == []


@pytest.mark.parametrize('angle,scale', [(0, 1), (20, 1.1)])
def test_keypoint_pipeline_after_jpeg(angle, scale):
    rng = np.random.default_rng(57)
    image = rng.integers(20, 230, (384, 384, 3), dtype=np.uint8)
    # Smoothed texture and shapes supply reproducible multi-scale keypoints.
    image = cv2.GaussianBlur(image, (3, 3), .7)
    for _ in range(50):
        x, y = rng.integers(20, 150, 2)
        cv2.circle(image, (int(x), int(y)), int(rng.integers(2, 10)),
                   tuple(map(int, rng.integers(0, 255, 3))), -1)
    patch = image[16:176, 16:176].copy()
    transform = cv2.getRotationMatrix2D((80, 80), angle, scale)
    transformed = cv2.warpAffine(patch, transform, (160, 160), borderMode=cv2.BORDER_REFLECT)
    forged = image.copy()
    forged[208:368, 208:368] = transformed
    _, encoded = cv2.imencode('.jpg', forged, [cv2.IMWRITE_JPEG_QUALITY, 70])
    compressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    config = dict(feature='KP', threshold_t=6, block_size=16, d_min=40)
    result = detect_copymove_block(compressed, config)
    assert result.is_forged
    assert result.total_matches >= 6
    assert np.any(result.mask[16:176, 16:176])
    assert np.any(result.mask[208:368, 208:368])
    assert not detect_copymove_block(image, config).is_forged


def test_planar_homography_with_outliers():
    from src.geometric import verify_geometric_pairs
    rng = np.random.default_rng(82)
    source = rng.uniform(10, 90, (20, 2)).astype(np.float32)
    h = np.float64([[1.1, .1, 140], [.02, 1, 100], [.001, .0005, 1]])
    target = cv2.perspectiveTransform(source[:, None], h)[:, 0]
    points = np.vstack([source, target, rng.uniform(0, 300, (6, 2))]).astype(np.float32)
    pairs = [(i, i + 20) for i in range(20)] + [(i, i + 40) for i in range(6)]
    cv2.setRNGSeed(42)
    accepted, _, models = verify_geometric_pairs(pairs, points, min_inliers=8,
                                                reprojection_error=.5, model='homography')
    assert set(accepted) == {(i, i + 20) for i in range(20)}
    assert len(models) == 1
