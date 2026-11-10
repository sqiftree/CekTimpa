"""Normalize and vote on matched-pair displacement vectors."""
import numpy as np
from .collinear import _filter_collinear_cluster

def shift_key(sy, sx, shift_tol):
    if sy < 0 or (sy == 0 and sx < 0):
        sy,sx=-sy,-sx
    if shift_tol > 1:
        return (int(round(sy/shift_tol)*shift_tol), int(round(sx/shift_tol)*shift_tol))
    return (int(sy),int(sx))

def select_winners(candidates, coords, threshold_t, shift_tol, reject_collinear):
    votes, groups, pairs_by_key = {}, {}, {}
    for pair,sy,sx in candidates:
        key=shift_key(sy,sx,shift_tol)
        votes[key]=votes.get(key,0)+1
        groups.setdefault(key,[]).append(pair)
        pairs_by_key.setdefault(key,[]).append(pair)
    winners=set()
    for key,count in votes.items():
        if count >= threshold_t and (not reject_collinear or _filter_collinear_cluster(groups[key],coords)):
            winners.add(key)
    if not winners:
        return [],np.empty((0,2),dtype=np.int32)
    pairs=list(dict.fromkeys(pair for key in votes if key in winners for pair in pairs_by_key[key]))
    shifts=np.asarray(sorted(winners,key=lambda key:(-votes[key],key)),dtype=np.int32)
    return pairs,shifts
