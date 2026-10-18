"""
Modul Pencocokan Fitur & Voting Vektor Pergeseran
Alat bantu komputasi konvensional:
- Kuantisasi leksikografis
- Penapisan jarak spasial minimum (D_min)
- Akumulasi translasi pergeseran (Voting Shift Vectors)
"""

from typing import List, Tuple, Dict
import numpy as np

def match_features(
    features: np.ndarray,
    coords: np.ndarray,
    q_step: float = 1.0,
    eps: float = 2.0,
    window: int = 10,
    d_min: float = 30.0,
    threshold_t: int = 50
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Melakukan pencocokan fitur blok secara leksikografis dan voting vektor pergeseran.
    
    Args:
        features: Matriks fitur shape (N, D)
        coords: Koordinat (y, x) blok shape (N, 2)
        q_step: Langkah kuantisasi fitur Q
        eps: Ambang batas jarak Euclidean kemiripan fitur
        window: Jendela pencarian tetangga terdekat di daftar urut
        d_min: Jarak spasial minimum antar blok pasangan
        threshold_t: Ambang batas minimum akumulasi suara vektor pergeseran (T)

    Returns:
        matched_pairs: List pasangan indeks blok yang terverifikasi konsisten
        winning_shifts: Matriks vektor pergeseran yang lolos ambang T
    """
    n_blocks = len(features)
    if n_blocks == 0:
        return [], np.empty((0, 2), dtype=np.int32)

    # 1. Kuantisasi fitur
    q_feats = np.round(features / max(q_step, 1e-4)).astype(np.int32)

    # 2. Pengurutan leksikografis
    order = np.lexsort(q_feats.T[::-1])
    sorted_feats = features[order]
    sorted_coords = coords[order].astype(np.float32)

    candidates = []
    eps_sq = eps * eps
    d_min_sq = d_min * d_min

    # 3. Pencarian pasangan kandidat dalam jendela w tetangga
    for i in range(n_blocks):
        max_j = min(i + window + 1, n_blocks)
        f_i = sorted_feats[i]
        c_i = sorted_coords[i]

        for j in range(i + 1, max_j):
            # Cek jarak fitur (Euclidean)
            f_diff = f_i - sorted_feats[j]
            if np.dot(f_diff, f_diff) <= eps_sq:
                # Cek jarak spasial (mencegah false alarm blok bertetangga)
                s_diff = c_i - sorted_coords[j]
                if np.dot(s_diff, s_diff) >= d_min_sq:
                    sy, sx = int(s_diff[0]), int(s_diff[1])
                    # Kanonisasi tanda vektor pergeseran
                    if sy < 0 or (sy == 0 and sx < 0):
                        sy, sx = -sy, -sx
                    candidates.append((order[i], order[j], (sy, sx)))

    # 4. Voting vektor pergeseran
    shift_votes: Dict[Tuple[int, int], int] = {}
    for _, _, shift in candidates:
        shift_votes[shift] = shift_votes.get(shift, 0) + 1

    winning_shifts = {s for s, count in shift_votes.items() if count >= threshold_t}

    matched_pairs = []
    for idx_a, idx_b, shift in candidates:
        if shift in winning_shifts:
            matched_pairs.append((idx_a, idx_b))

    winning_arr = np.array(list(winning_shifts), dtype=np.int32) if winning_shifts else np.empty((0, 2), dtype=np.int32)
    return matched_pairs, winning_arr
