"""
Modul Pencocokan Fitur Blok & Penyaringan Vektor Pergeseran (Shift Vector Voting)
Mendukung Pencocokan KD-Tree (Akurat untuk Objek Kecil/Kompleks) & Lexicographical Sort
"""

from typing import List, Tuple
import numpy as np
from scipy.spatial import cKDTree


def _filter_collinear_cluster(
    pair_indices: List[Tuple[int, int]],
    coords: np.ndarray,
    min_minor_axis: float = 2.5
) -> bool:
    """
    Memeriksa apakah kumpulan blok yang cocok membentuk gumpalan objek 2D (seperti tubuh orang)
    atau hanya berjajar membentuk garis lurus 1D (seperti lis atap gedung / zebra cross).
    Menggunakan analisis nilai eigen matriks kovariansi koordinat (PCA Spasial 2D).
    """
    if len(pair_indices) < 4:
        return False

    pts = np.array([coords[i] for i, _ in pair_indices], dtype=np.float32)
    if len(pts) < 4:
        return False

    centered = pts - pts.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    if cov.ndim < 2:
        return False

    eigvals = np.linalg.eigvalsh(cov)
    minor_std = float(np.sqrt(max(eigvals[0], 0.0)))
    # Jika sebaran sumbu pendek (minor axis) terlalu tipis, berarti blok hanya berbaris di garis 1D
    return minor_std >= min_minor_axis


def match_features(
    features: np.ndarray,
    coords: np.ndarray,
    q_step: float = 1.0,
    eps: float = 2.0,
    window: int = 10,
    d_min: float = 30.0,
    threshold_t: int = 50,
    method: str = "kdtree",
    shift_tol: int = 1,
    reject_collinear: bool = False
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Mencocokkan vektor fitur antar blok dan menyaring kandidat menggunakan
    jarak spasial minimum (d_min) serta konsistensi arah pergeseran (threshold_t).

    Parameters
    ----------
    features : np.ndarray
        Matriks fitur berukuran (N, D).
    coords : np.ndarray
        Matriks koordinat (y, x) berukuran (N, 2).
    q_step : float, default=1.0
        Langkah kuantisasi fitur.
    eps : float, default=2.0
        Batas maksimum jarak Euclidean antar vektor fitur untuk dianggap mirip.
    window : int, default=10
        Jumlah tetangga kandidat yang diperiksa pada urutan leksikografis / KD-Tree.
    d_min : float, default=30.0
        Jarak fisik Euclidean minimum (piksel) antara dua blok agar tidak mencocokkan blok berdekatan.
    threshold_t : int, default=50
        Ambang batas frekuensi (voting) minimum untuk vektor translasi (sy, sx).
    method : str, default="kdtree"
        Metode pencarian pasangan: "kdtree" (direkomendasikan untuk objek nyata) atau "lexsort".
    shift_tol : int, default=1
        Toleransi pengelompokan vektor pergeseran (piksel) untuk menangani tepi objek non-persegi.
    reject_collinear : bool, default=False
        Jika True, menolak klaster pergeseran yang hanya membentuk garis lurus 1D (seperti lis gedung).

    Returns
    -------
    matched_pairs : List[Tuple[int, int]]
        Daftar pasangan indeks blok (idx_a, idx_b) yang terverifikasi sebagai area duplikasi.
    winning_shifts : np.ndarray
        Array berukuran (K, 2) berisi vektor translasi (sy, sx) yang melampaui threshold_t.
    """
    if method not in {"kdtree", "lexsort"}:
        raise ValueError("method must be kdtree or lexsort")
    n_blocks = len(features)
    if n_blocks < 2:
        return [], np.empty((0, 2), dtype=np.int32)

    q_val = max(float(q_step), 1e-4)
    scaled_features = features.astype(np.float32) / q_val
    eps_val = float(eps)
    d_min_sq = float(d_min) ** 2
    k_neighbors = min(max(int(window), 2), n_blocks)

    candidate_pairs: List[Tuple[int, int, Tuple[int, int]]] = []
    shift_votes = {}
    shift_groups = {}

    if method == "kdtree":
        # Pencarian tetangga terdekat di ruang fitur berdimensi-D tanpa terpotong batas pembulatan
        tree = cKDTree(scaled_features)
        seen_pairs = set()
        # Expand search until enough spatially eligible neighbors are found.
        # Query one block at a time to avoid a dense N x N allocation.
        for i in range(n_blocks):
            yi, xi = map(int, coords[i])
            k = min(k_neighbors + 1, n_blocks)
            while True:
                distances, indices = tree.query(scaled_features[i], k=k,
                                                 distance_upper_bound=eps_val)
                eligible = []
                for distance, j in zip(np.atleast_1d(distances), np.atleast_1d(indices)):
                    j = int(j)
                    if j == i or j >= n_blocks or not np.isfinite(distance):
                        continue
                    yj, xj = map(int, coords[j])
                    if (yi - yj) ** 2 + (xi - xj) ** 2 >= d_min_sq:
                        eligible.append(j)
                if (len(eligible) >= k_neighbors or k == n_blocks
                        or not np.all(np.isfinite(distances))):
                    break
                k = min(2 * k, n_blocks)

            for j in eligible[:k_neighbors]:
                pair = (min(i, j), max(i, j))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                yj, xj = map(int, coords[j])
                sy, sx = yi - yj, xi - xj
                if sy < 0 or (sy == 0 and sx < 0):
                    sy, sx = -sy, -sx
                if shift_tol > 1:
                    shift_key = (int(round(sy / shift_tol) * shift_tol),
                                 int(round(sx / shift_tol) * shift_tol))
                else:
                    shift_key = (sy, sx)
                candidate_pairs.append((*pair, shift_key))
                shift_votes[shift_key] = shift_votes.get(shift_key, 0) + 1
                shift_groups.setdefault(shift_key, []).append(pair)

    else:
        # Metode Klasik: Pengurutan Leksikografis (Lexsort)
        q_feats = np.rint(scaled_features).astype(np.int32)
        order = np.lexsort(q_feats.T[::-1])
        eps_sq = eps_val ** 2

        for pos in range(n_blocks):
            i = int(order[pos])
            feat_i = scaled_features[i]
            yi, xi = int(coords[i, 0]), int(coords[i, 1])

            end_pos = min(pos + 1 + k_neighbors, n_blocks)
            for next_pos in range(pos + 1, end_pos):
                j_idx = int(order[next_pos])
                diff = feat_i - scaled_features[j_idx]
                if float(np.dot(diff, diff)) > eps_sq:
                    continue

                yj, xj = int(coords[j_idx, 0]), int(coords[j_idx, 1])
                if (yi - yj) ** 2 + (xi - xj) ** 2 < d_min_sq:
                    continue

                sy, sx = yi - yj, xi - xj
                if sy < 0 or (sy == 0 and sx < 0):
                    sy, sx = -sy, -sx

                idx_a, idx_b = (i, j_idx) if i < j_idx else (j_idx, i)
                if shift_tol > 1:
                    bin_y = int(round(sy / shift_tol) * shift_tol)
                    bin_x = int(round(sx / shift_tol) * shift_tol)
                    shift_key = (bin_y, bin_x)
                else:
                    shift_key = (sy, sx)

                candidate_pairs.append((idx_a, idx_b, shift_key))
                shift_votes[shift_key] = shift_votes.get(shift_key, 0) + 1
                shift_groups.setdefault(shift_key, []).append((idx_a, idx_b))

    if not candidate_pairs:
        return [], np.empty((0, 2), dtype=np.int32)

    # Seleksi vektor pergeseran yang melampaui ambang batas suara (threshold_t)
    winning_shift_keys = set()
    for shift_key, count in shift_votes.items():
        if count >= threshold_t:
            if reject_collinear:
                if not _filter_collinear_cluster(shift_groups[shift_key], coords):
                    continue
            winning_shift_keys.add(shift_key)

    if not winning_shift_keys:
        return [], np.empty((0, 2), dtype=np.int32)

    matched_pairs = [
        (i, j) for i, j, shift_key in candidate_pairs
        if shift_key in winning_shift_keys
    ]
    # Hilangkan duplikat pasangan jika ada
    matched_pairs = list(dict.fromkeys(matched_pairs))

    winning_shifts = np.array(sorted(winning_shift_keys,
                                    key=lambda key: (-shift_votes[key], key)), dtype=np.int32)
    return matched_pairs, winning_shifts
