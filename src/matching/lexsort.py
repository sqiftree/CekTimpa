"""Lexicographic feature-neighbor search backend."""
import numpy as np

def find_candidates(features, coords, k_neighbors, eps_val, d_min_sq):
    n_blocks = len(features)
    q_feats = np.rint(features).astype(np.int32)
    order = np.lexsort(q_feats.T[::-1])
    eps_sq = eps_val**2
    candidates = []
    for pos in range(n_blocks):
        i = int(order[pos])
        yi, xi = int(coords[i,0]), int(coords[i,1])
        end_pos = min(pos+1+k_neighbors, n_blocks)
        for next_pos in range(pos+1,end_pos):
            j = int(order[next_pos])
            diff = features[i]-features[j]
            if float(np.dot(diff,diff)) > eps_sq:
                continue
            yj,xj = int(coords[j,0]),int(coords[j,1])
            if (yi-yj)**2+(xi-xj)**2 < d_min_sq:
                continue
            candidates.append(((min(i,j),max(i,j)),yi-yj,xi-xj))
    return candidates
