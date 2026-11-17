"""Periksa isi mask CoMoFoD sebelum eksperimen (apakah _B, _M, atau gabungan yang benar).

    python -m experiments.paper.check_masks --data-dir experiments/paper/data
Untuk tiap set: luas _B, luas _M, tumpang tindih keduanya, luas gabungan.
Tafsir: jika _B dan _M hampir tidak bertumpuk (overlap ~0) dan luasnya mirip, kemungkinan
keduanya adalah sumber dan salinan, sehingga pakai --mask-mode union. Jika sama persis,
cukup salah satu. Pastikan juga dengan MELIHAT gambarnya.
"""
import argparse
import re
from pathlib import Path

import cv2
import numpy as np

from experiments.paper.bench_common import _index_files, _read_mask


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=20)
    a = ap.parse_args()
    idx = _index_files(a.data_dir)
    nums = sorted({m.group(1) for k in idx if (m := re.match(r"^(\d+)_f$", k))})
    print("set | luas_B | luas_M | overlap | luas_gabungan | ukuran_F")
    for n in nums[:a.limit]:
        b, m = _read_mask(idx.get(f"{n}_b")), _read_mask(idx.get(f"{n}_m"))
        f = cv2.imread(str(idx[f"{n}_f"]))
        sz = None if f is None else f.shape[:2]
        ab = int(b.sum()) if b is not None else None
        am = int(m.sum()) if m is not None else None
        ov = int((b & m).sum()) if b is not None and m is not None and b.shape == m.shape else None
        un = int((b | m).sum()) if ov is not None else None
        print(f"{n} | {ab} | {am} | {ov} | {un} | {sz}")


if __name__ == "__main__":
    main()
