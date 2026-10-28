"""Compare deployed pipelines on controlled JPEG patch-support cases.

Each source has a negative and three positive variants. Pixel metrics use
source+target 160x160 support rectangles, including reflected warp borders.
This is a localization proxy, not precise object ground truth or real-data accuracy.
Use separate Python processes for each --root to isolate imported packages.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path('.'))
parser.add_argument('--variant', choices=['main_before', 'alternate', 'integrated'], default='integrated')
parser.add_argument('--output', type=Path, default=Path('experiments/results/branch_comparison.csv'))
args = parser.parse_args()
sys.path.insert(0, str(args.root.resolve()))
import cv2
import numpy as np
if args.variant in {'alternate', 'integrated'}:
    from src.detectors.copymove_sift import detect_copymove_sift as detect
    config = dict(min_inliers=6, ratio_threshold=.78, d_min=40, kp_radius=10, ransac_threshold=3)
else:
    from src.detectors.copymove_block import detect_copymove_block as detect
    config = dict(feature='KP', threshold_t=6, d_min=40, block_size=16)
rows=[]
for seed in range(3):
    rng=np.random.default_rng(57+seed)
    image=rng.integers(20,230,(384,384,3),dtype=np.uint8)
    image=cv2.GaussianBlur(image,(3,3),.7)
    for _ in range(50):
        x,y=rng.integers(20,150,2)
        cv2.circle(image,(int(x),int(y)),int(rng.integers(2,10)),tuple(map(int,rng.integers(0,255,3))),-1)
    for label,angle,scale in [('clean',0,1),('translation',0,1),('rotation',20,1.1),('rotation45',45,.85)]:
        test=image.copy();gt=np.zeros((384,384),np.uint8)
        if label!='clean':
            patch=image[16:176,16:176].copy()
            mat=cv2.getRotationMatrix2D((80,80),angle,scale)
            test[208:368,208:368]=cv2.warpAffine(patch,mat,(160,160),borderMode=cv2.BORDER_REFLECT)
            gt[16:176,16:176]=255;gt[208:368,208:368]=255
        _,enc=cv2.imencode('.jpg',test,[cv2.IMWRITE_JPEG_QUALITY,70]);test=cv2.imdecode(enc,cv2.IMREAD_COLOR)
        cv2.setRNGSeed(42)
        t=time.perf_counter();result=detect(test,config);elapsed=(time.perf_counter()-t)*1000
        pred=result.mask>0; truth=gt>0
        tp=int((pred&truth).sum());fp=int((pred&~truth).sum());fn=int((~pred&truth).sum())
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0
        rows.append(dict(variant=args.variant,seed=seed,case=label,predicted=bool(result.is_forged),f1=f1,runtime_ms=elapsed,tp=tp,fp=fp,fn=fn))
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
for case in ['translation', 'rotation', 'rotation45']:
    subset = [r for r in rows if r['case'] == case]
    print(f"{case}: F1={np.mean([r['f1'] for r in subset]):.4f}")
print(f"Negative alarms: {sum(r['predicted'] for r in rows if r['case'] == 'clean')}/3")
print(f"Mean runtime: {np.mean([r['runtime_ms'] for r in rows]):.1f} ms")
print(f"Saved {args.output}; config={json.dumps(config)}")
