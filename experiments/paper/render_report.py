"""Render hasil paper dari keluaran benchmark nyata, tanpa membuat angka atau kesimpulan.

Contoh:
  python -m experiments.paper.render_report \
    --summary experiments/paper/results/figures/summary.csv \
    --metadata experiments/paper/results/*.meta.json \
    --best-config experiments/paper/results/best_config.json

Secara default semua E1 sampai E5 wajib tersedia. Gunakan --allow-partial hanya untuk
laporan kemajuan; dokumen tersebut secara eksplisit menandai skenario yang belum ada.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ORDER = ["SIFT", "A", "B", "DCT"]
EXPECTED = {"e1", "e2", "e3", "e4", "e5"}


def fmt(value, digits=4):
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def markdown_table(headers, rows):
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    return "\n".join([line, sep] + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def method_table(summary, scenario, condition, levels=None):
    part = summary[(summary.scenario == scenario) & (summary.condition == condition)]
    if levels is not None:
        part = part[part.level.isin([str(level) for level in levels])]
    if part.empty:
        return "Tidak ada data untuk kondisi ini."
    levels = sorted(part.level.unique(), key=lambda value: float(value))
    rows = []
    for level in levels:
        row = [level]
        for method in ORDER:
            cell = part[(part.level == level) & (part.method == method)]
            row.append(fmt(cell.iloc[0].f1_mean) if not cell.empty else "N/A")
        rows.append(row)
    return markdown_table(["Level"] + ORDER, rows)


def e1_table(summary):
    part = summary[summary.scenario == "e1"]
    rows = []
    for method in ORDER:
        cell = part[part.method == method]
        if cell.empty:
            continue
        value = cell.iloc[0]
        rows.append([method, fmt(value.precision), fmt(value.recall), fmt(value.f1_mean),
                     fmt(value.iou), fmt(value.tpr_img), fmt(value.fpr_img),
                     fmt(value.runtime_ms_median, 1)])
    return markdown_table(
        ["Metode", "Precision", "Recall", "F1", "IoU", "TPR citra", "FPR citra", "Waktu median (ms)"],
        rows,
    ) if rows else "Tidak ada data E1."


def e2_table(summary):
    part = summary[summary.scenario == "e2"]
    mapping = [("A", "block_size"), ("B", "block_size"), ("DCT", "block_size"),
               ("A", "threshold_t"), ("B", "threshold_t"), ("DCT", "threshold_t"),
               ("SIFT", "sift_ratio"), ("SIFT", "sift_min_inliers")]
    rows = []
    for method, condition in mapping:
        choices = part[(part.method == method) & (part.condition == condition)].copy()
        if choices.empty:
            continue
        choices["numeric_level"] = choices.level.astype(float)
        choices = choices.sort_values("numeric_level")
        best = choices.sort_values(["f1_mean", "numeric_level"], ascending=[False, True]).iloc[0]
        values = ", ".join(f"{level}: {fmt(score)}" for level, score in zip(choices.level, choices.f1_mean))
        parameter = {"sift_ratio": "ratio_threshold", "sift_min_inliers": "min_inliers"}.get(condition, condition)
        rows.append([method, parameter, values, best.level, fmt(best.f1_mean)])
    return markdown_table(["Metode", "Parameter", "F1 per nilai", "Nilai terpilih", "F1 terpilih"], rows) if rows else "Tidak ada data E2."


def condition_table(summary, scenario, conditions):
    part = summary[(summary.scenario == scenario) & (summary.condition.isin(conditions))]
    if part.empty:
        return "Tidak ada data untuk kondisi ini."
    rows = []
    for condition in conditions:
        row = [condition]
        for method in ORDER:
            cell = part[(part.condition == condition) & (part.method == method)]
            row.append(fmt(cell.iloc[0].f1_mean) if not cell.empty else "N/A")
        rows.append(row)
    return markdown_table(["Kondisi"] + ORDER, rows)


def load_metadata(paths):
    records = [json.loads(path.read_text()) for path in paths]
    non_comofod = [record.get("output", "unknown") for record in records if record.get("dataset") != "comofod"]
    if non_comofod:
        raise ValueError("Menolak laporan karena metadata bukan CoMoFoD: " + ", ".join(non_comofod))
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", required=True, type=Path)
    ap.add_argument("--metadata", required=True, type=Path, nargs="+")
    ap.add_argument("--best-config", type=Path)
    ap.add_argument("--output", type=Path, default=Path("experiments/paper/data/hasil_experiment.md"))
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    summary = pd.read_csv(args.summary)
    required_columns = {"scenario", "condition", "level", "method", "f1_mean", "precision", "recall", "iou", "tpr_img", "fpr_img", "runtime_ms_median"}
    missing_columns = required_columns - set(summary.columns)
    if missing_columns:
        raise ValueError("Kolom summary hilang: " + ", ".join(sorted(missing_columns)))
    scenarios = set(summary.scenario.dropna())
    missing_scenarios = EXPECTED - scenarios
    if missing_scenarios and not args.allow_partial:
        raise ValueError("Skenario belum lengkap: " + ", ".join(sorted(missing_scenarios)))
    metadata = load_metadata(args.metadata)
    if not metadata:
        raise ValueError("Metadata wajib ada.")

    first = metadata[0]
    total_sets = max(record.get("sets", 0) for record in metadata)
    total_images = max(record.get("images", 0) for record in metadata)
    metadata_rows = [[record.get("output"), ", ".join(record.get("scenarios", [])), record.get("subset"),
                      record.get("images"), record.get("positive_images"), record.get("negative_images"),
                      record.get("elapsed_seconds"), record.get("created_at")] for record in metadata]
    config_text = "Tidak tersedia. Jalankan E2 lalu make_figures dengan input E2."
    if args.best_config and args.best_config.is_file():
        config_text = "```json\n" + args.best_config.read_text().strip() + "\n```"

    sections = [
        "# Hasil Eksperimen",
        "",
        "Paper: Performance Evaluation of Spatial and Frequency Features for Block-Based Copy-Move Forgery Detection",
        "Kelompok: Kelas C, Kelompok 7",
        "",
        "> Dokumen ini dirender dari CSV benchmark CoMoFoD dan metadata eksekusi. Tidak ada angka, hasil sintetis, atau kesimpulan yang dibuat otomatis. Interpretasi ilmiah harus ditulis setelah memeriksa data.",
        "",
        "## 1. Jejak eksekusi",
        "",
        markdown_table(["Output CSV", "Skenario", "Subset", "Citra", "Positif", "Negatif", "Durasi (s)", "Waktu"], metadata_rows),
        "",
        markdown_table(["Item", "Nilai"], [["Dataset", "CoMoFoD set kecil 512x512"], ["Mask mode", str(first.get("mask_mode"))], ["Seed", str(first.get("seed"))], ["Workers", str(first.get("workers"))], ["Python", str(first.get("python"))], ["OpenCV", str(first.get("opencv"))], ["NumPy", str(first.get("numpy"))], ["Platform", str(first.get("platform"))], ["CPU logical cores", str(first.get("cpu_logical_cores"))]]),
        "",
        "## 2. Data yang dipakai",
        "",
        markdown_table(["Item", "Nilai"], [["Set pada satu eksekusi terbesar", total_sets], ["Citra pada satu eksekusi terbesar", total_images], ["Sumber data", str(first.get("data_dir"))], ["Jumlah skenario tersedia", len(scenarios)]]),
        "",
        "## 3. Hasil kuantitatif",
        "",
        "### 3.1 Citra bersih (E1)", "", e1_table(summary), "",
        "Gambar: ![F1 citra bersih](../results/figures/e1_clean_f1.png)", "",
        "### 3.2 Tuning parameter (E2)", "", e2_table(summary), "", "Konfigurasi yang dibekukan:", "", config_text, "",
        "### 3.3 Ketahanan terhadap gangguan (E3)",
        "", "#### Noise Gaussian", "", method_table(summary, "e3", "gaussian_noise"), "", "![Noise Gaussian](../results/figures/e3_gaussian_noise.png)",
        "", "#### Salt and pepper", "", method_table(summary, "e3", "salt_pepper"), "", "![Salt and pepper](../results/figures/e3_salt_pepper.png)",
        "", "#### Blur Gaussian", "", method_table(summary, "e3", "gaussian_blur"), "", "![Blur Gaussian](../results/figures/e3_gaussian_blur.png)",
        "", "#### Kompresi JPEG", "", method_table(summary, "e3", "jpeg"), "", "![JPEG](../results/figures/e3_jpeg.png)",
        "", "### 3.4 Perubahan kontras (E4)", "", method_table(summary, "e4", "gamma"), "", "![Gamma](../results/figures/e4_gamma.png)", "", "Histogram equalization:", "", method_table(summary, "e4", "hist_eq"),
        "", "### 3.5 Efek pra-filter (E5)", "", "Salt and pepper:", "", condition_table(summary, "e5", ["salt_pepper+none", "salt_pepper+median", "salt_pepper+gaussian"]), "", "Gaussian noise:", "", condition_table(summary, "e5", ["gaussian_noise+none", "gaussian_noise+median", "gaussian_noise+gaussian"]), "", "![Pra-filter salt and pepper](../results/figures/e5_prefilter_salt_pepper.png)", "", "![Pra-filter noise Gaussian](../results/figures/e5_prefilter_gaussian_noise.png)",
        "", "## 4. Interpretasi dan kesimpulan", "", "Bagian ini sengaja tidak diisi program. Tulis fakta dari tabel atau grafik, interpretasi yang dapat dipertanggungjawabkan, lalu kaitannya dengan H1 sampai H4.",
    ]
    if missing_scenarios:
        sections.extend(["", "## Peringatan kelengkapan", "", "Skenario belum tersedia: " + ", ".join(sorted(missing_scenarios)) + ". Dokumen ini hanya laporan kemajuan."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(map(str, sections)) + "\n")
    print("Laporan dirender ->", args.output)


if __name__ == "__main__":
    main()
