"""KD-tree feature-neighbor search backend."""
import numpy as np
from scipy.spatial import cKDTree

def find_candidates(features, coords, k_neighbors, eps_val, d_min_sq):
    n_blocks = len(features)
    tree = cKDTree(features)
    seen_pairs = set()
    candidates = []
    for i in range(n_blocks):
        yi, xi = map(int, coords[i])
        k = min(k_neighbors + 1, n_blocks)
        while True:
            distances, indices = tree.query(features[i], k=k, distance_upper_bound=eps_val)
            eligible = []
            for distance, j in zip(np.atleast_1d(distances), np.atleast_1d(indices)):
                j = int(j)
                if j == i or j >= n_blocks or not np.isfinite(distance):
                    continue
                yj, xj = map(int, coords[j])
                if (yi-yj)**2 + (xi-xj)**2 >= d_min_sq:
                    eligible.append(j)
            if len(eligible) >= k_neighbors or k == n_blocks or not np.all(np.isfinite(distances)):
                break
            k = min(2*k, n_blocks)
        for j in eligible[:k_neighbors]:
            pair = (min(i,j), max(i,j))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            yj, xj = map(int, coords[j])
            candidates.append((pair, yi-yj, xi-xj))
    return candidates
