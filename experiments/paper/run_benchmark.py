"""Eksperimen utama paper: SIFT vs fitur spasial (A, B) vs fitur frekuensi (DCT).

Contoh (uji cepat, data sintetis):
    python -m experiments.paper.run_benchmark --dataset synthetic --n-sources 6 \
        --scenarios e1,e3 --subset all --workers 2 --output experiments/paper/results/smoke.csv

Contoh (CoMoFoD, subset tuning dulu, lalu bekukan parameter dan jalankan subset test):
    python -m experiments.paper.run_benchmark --dataset comofod --data-dir experiments/paper/data \
        --scenarios e2 --subset tune --output experiments/paper/results/e2_tune.csv
    python -m experiments.paper.run_benchmark --dataset comofod --data-dir experiments/paper/data \
        --scenarios e1,e3,e4,e5 --subset test --output experiments/paper/results/main_test.csv

Satu baris CSV = satu (citra, kondisi, metode). Ground truth hanya dipakai setelah
deteksi untuk metrik. Hasil pada data sintetis BUKAN estimasi akurasi foto nyata.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from experiments.paper.bench_common import (DEFAULT_CONFIG, METHODS, Condition, Sample,
                                      build_conditions, load_comofod, load_synthetic, subsample_sets,
                                      method_config, split_by_source, apply_attack, _prefilter)
from src.detectors.copymove_block import detect_copymove_block
from src.detectors.copymove_sift import detect_copymove_sift
from src.metrics import calculate_pixel_metrics, calculate_psnr

FIELDS = ["sample_id", "label", "scenario", "condition", "level", "method",
          "predicted", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "iou",
          "runtime_ms", "psnr_db", "error", "config"]


def run_detector(method: str, image: np.ndarray, cfg: dict):
    if method == "SIFT":
        return detect_copymove_sift(image, cfg)
    return detect_copymove_block(image, cfg)


def process_condition(sample: Sample, ci: int, cond, base_cfg, methods, seed):
    """Satu kondisi untuk satu citra. Hasil dapat ditulis segera saat selesai."""
    cv2.setNumThreads(1)
    rows = []
    attacked = apply_attack(cond, sample.image, seed + ci)
    psnr = calculate_psnr(sample.image, attacked)
    test_img = _prefilter(attacked, cond.prefilter)
    # E2 menguji satu parameter metode tertentu. Menjalankan metode lain yang
    # parameternya tidak berubah hanya memboroskan waktu dan tidak menambah bukti tuning.
    active_methods = (methods if cond.scenario != "e2"
                      else [method for method in methods if method in cond.overrides])
    for method in active_methods:
        cfg = method_config(method, cond, base_cfg)
        row = dict(sample_id=sample.sample_id, label=sample.label,
                   scenario=cond.scenario, condition=cond.name, level=cond.level,
                   method=method, psnr_db=("inf" if np.isinf(psnr) else f"{psnr:.2f}"),
                   config=json.dumps(cfg, sort_keys=True), error="")
        try:
            res = run_detector(method, test_img, cfg)
            m = calculate_pixel_metrics(res.mask, sample.gt_mask)
            row.update(predicted=int(res.is_forged),
                       runtime_ms=f"{res.timings['total_ms']:.1f}",
                       **{k: m[k] for k in ("tp", "fp", "fn", "tn",
                                            "precision", "recall", "f1", "iou")})
        except Exception as exc:  # catat, jangan hentikan eksperimen
            row.update(predicted="", runtime_ms="", error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["synthetic", "comofod"], default="synthetic")
    ap.add_argument("--data-dir", type=Path, help="folder CoMoFoD (untuk --dataset comofod)")
    ap.add_argument("--mask-mode", choices=["B", "M", "union"], default="B",
                    help="mask CoMoFoD: _B, _M, atau gabungan (cek dulu dengan check_masks.py)")
    ap.add_argument("--sources-dir", type=Path,
                    help="folder foto nyata sebagai sumber sintetis (opsional)")
    ap.add_argument("--n-sources", type=int, default=20, help="jumlah sumber sintetis")
    ap.add_argument("--scenarios", default="e1", help="koma: e1,e2,e3,e4,e5")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--subset", choices=["tune", "test", "all"], default="test",
                    help="pembagian 30%% tune / 70%% test berdasarkan sumber")
    ap.add_argument("--levels", choices=["full", "reduced"], default="full",
                    help="reduced = 2 level per gangguan (jauh lebih cepat)")
    ap.add_argument("--positives-only", action="store_true",
                    help="lewati citra asli (negatif); pakai untuk e3-e5, FPR cukup dari e1")
    ap.add_argument("--n-sets", type=int, default=0,
                    help="ambil N set (pasangan F/O) secara acak sebelum dibagi tune/test; 0 = semua")
    ap.add_argument("--max-images", type=int, default=0, help="batasi jumlah citra (debug)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--config-json", type=Path,
                    help="JSON {metode: {param: nilai}} untuk menimpa parameter awal")
    ap.add_argument("--output", type=Path, default=Path("experiments/paper/results/benchmark.csv"))
    args = ap.parse_args()

    if args.dataset == "comofod":
        if not args.data_dir or not args.data_dir.is_dir():
            ap.error("--data-dir wajib dan harus berupa folder untuk comofod")
        samples = load_comofod(args.data_dir, args.mask_mode)
    else:
        samples = load_synthetic(args.n_sources, args.seed, args.sources_dir)
    if not samples:
        sys.exit("Tidak ada citra yang terbaca. Periksa --data-dir / penamaan berkas (NNN_F, NNN_O, NNN_B).")
    samples = subsample_sets(samples, args.n_sets, args.seed)
    samples = split_by_source(samples, args.subset, args.seed)
    if args.positives_only:
        samples = [s for s in samples if s.label == 1]
    if args.max_images:
        samples = samples[:args.max_images]

    base_cfg = {m: dict(c) for m, c in DEFAULT_CONFIG.items()}
    if args.config_json:
        for m, over in json.loads(args.config_json.read_text()).items():
            base_cfg[m].update(over)
    methods = [m.strip() for m in args.methods.split(",")]
    conditions = build_conditions([s.strip() for s in args.scenarios.split(",")], args.levels)

    n_pos = sum(s.label for s in samples)
    set_ids = sorted({s.sample_id.rsplit("_", 1)[0] for s in samples})
    print(f"Citra: {len(samples)} (positif {n_pos}, negatif {len(samples) - n_pos}) "
          f"subset={args.subset} | kondisi={len(conditions)} | metode={methods}")
    detections_per_image = sum(
        len(methods) if condition.scenario != "e2"
        else sum(method in condition.overrides for method in methods)
        for condition in conditions
    )
    total = len(samples) * detections_per_image
    print(f"Total deteksi: {total}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done = 0
    with args.output.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        fh.flush()

        jobs = [(sample, ci, condition) for sample in samples
                for ci, condition in enumerate(conditions)]

        def flush(rows):
            nonlocal done
            writer.writerows(rows)
            fh.flush()
            done += 1
            if done % max(1, len(jobs) // 20) == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"  {done}/{len(jobs)} kondisi-citra | {el:.0f}s berjalan", flush=True)

        if args.workers > 1:
            # OpenCV may deadlock in forked Linux workers after the parent has
            # decoded images. Spawn starts a clean interpreter per worker.
            with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
                futs = [pool.submit(process_condition, sample, ci, condition, base_cfg, methods, args.seed)
                        for sample, ci, condition in jobs]
                for f in as_completed(futs):
                    flush(f.result())
        else:
            for sample, ci, condition in jobs:
                flush(process_condition(sample, ci, condition, base_cfg, methods, args.seed))
    elapsed_s = time.time() - t0
    metadata = {
        "created_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "data_dir": str(args.data_dir) if args.data_dir else None,
        "mask_mode": args.mask_mode if args.dataset == "comofod" else None,
        "scenarios": [s.strip() for s in args.scenarios.split(",")],
        "subset": args.subset,
        "levels": args.levels,
        "positives_only": args.positives_only,
        "n_sets_requested": args.n_sets,
        "seed": args.seed,
        "workers": args.workers,
        "images": len(samples),
        "positive_images": n_pos,
        "negative_images": len(samples) - n_pos,
        "sets": len(set_ids),
        "set_ids": set_ids,
        "elapsed_seconds": round(elapsed_s, 3),
        "python": sys.version.split()[0],
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu_logical_cores": os.cpu_count(),
        "config_json": str(args.config_json) if args.config_json else None,
        "output": str(args.output),
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Selesai dalam {elapsed_s:.0f}s -> {args.output}")
    print(f"Metadata eksekusi -> {metadata_path}")


if __name__ == "__main__":
    main()
