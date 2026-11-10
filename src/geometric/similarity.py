"""Estimate a partial affine similarity transform."""
import cv2
import numpy as np

def estimate(source, target, reprojection_error):
    affine, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC,
        ransacReprojThreshold=reprojection_error, maxIters=4000, confidence=.995, refineIters=10)
    matrix = np.vstack([affine,[0,0,1]]) if affine is not None else None
    return matrix,inliers
