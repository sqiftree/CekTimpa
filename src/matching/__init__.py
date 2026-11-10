"""Feature matching coordinator with selectable search backends."""
from typing import List, Tuple
import numpy as np
from .kdtree import find_candidates as find_kdtree_candidates
from .lexsort import find_candidates as find_lexsort_candidates
from .shift_voting import select_winners

def match_features(features: np.ndarray, coords: np.ndarray, q_step: float=1.0, eps: float=2.0,
                   window: int=10, d_min: float=30.0, threshold_t: int=50, method: str="kdtree",
                   shift_tol: int=1, reject_collinear: bool=False) -> Tuple[List[Tuple[int,int]],np.ndarray]:
    """Match feature vectors, then retain displacement vectors with enough votes."""
    if method not in {"kdtree","lexsort"}:
        raise ValueError("method must be kdtree or lexsort")
    n=len(features)
    if n<2:
        return [],np.empty((0,2),dtype=np.int32)
    q=max(float(q_step),1e-4)
    scaled=features.astype(np.float32)/q
    neighbors=min(max(int(window),2),n)
    backend=find_kdtree_candidates if method=="kdtree" else find_lexsort_candidates
    candidates=backend(scaled,coords,neighbors,float(eps),float(d_min)**2)
    return select_winners(candidates,coords,threshold_t,shift_tol,reject_collinear)

__all__=["match_features"]
