"""Ringkas CSV benchmark -> tabel (CSV + LaTeX-ready) dan grafik untuk paper.

    python -m experiments.paper.make_figures --input experiments/paper/results/main_test.csv \
        --outdir experiments/paper/results/figures
Metrik: F1 piksel (rata-rata pada citra palsu, 95% CI bootstrap atas citra),
TPR/FPR tingkat citra, median waktu (ms). Baris ber-error dilaporkan terpisah.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["SIFT", "A", "B", "DCT"]
NAMES = {"SIFT": "SIFT (keypoint)", "A": "Sub-block (spatial)",
         "B": "Gaussian (spatial)", "DCT": "DCT (frequency)"}
STYLE = {"SIFT": ("#d55e00", "o"), "A": ("#0072b2", "s"), "B": ("#009e73", "^"), "DCT": ("#cc79a7", "D")}


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(axis=1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def summarize(df):
    rows = []
    for key, g in df.groupby(["scenario", "condition", "level", "method"], sort=False):
        pos, neg = g[g.label == 1], g[g.label == 0]
        lo, hi = boot_ci(pos.f1)
        rows.append(dict(scenario=key[0], condition=key[1], level=key[2], method=key[3],
                         n_pos=len(pos), f1_mean=pos.f1.mean(), f1_ci_lo=lo, f1_ci_hi=hi,
                         precision=pos.precision.mean(), recall=pos.recall.mean(),
                         iou=pos.iou.mean(), tpr_img=pos.predicted.mean(),
                         fpr_img=neg.predicted.mean() if len(neg) else np.nan,
                         runtime_ms_median=g.runtime_ms.median()))
    return pd.DataFrame(rows)


def plot_curves(s, outdir):
    for (sc, cond), g in s.groupby(["scenario", "condition"]):
        if g.level.nunique() < 2 or sc in ("e2", "e5"):
            continue
        fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=200)
        for m in ORDER:
            d = g[g.method == m].copy()
            if d.empty:
                continue
            d["lv"] = d.level.astype(float)
            d = d.sort_values("lv")
            c, mk = STYLE[m]
            ax.plot(d.lv, d.f1_mean, marker=mk, color=c, label=NAMES[m], lw=1.2, ms=4)
            ax.fill_between(d.lv, d.f1_ci_lo, d.f1_ci_hi, color=c, alpha=.12, lw=0)
        ax.set_xlabel(f"{cond.replace('_', ' ')} level")
        ax.set_ylabel("Pixel-level F1")
        ax.set_ylim(-.02, 1.02)
        ax.grid(alpha=.3)
        if cond == "jpeg":
            ax.invert_xaxis()
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(outdir / f"{sc}_{cond}.png")
        plt.close(fig)


def plot_prefilter(s, outdir):
    g = s[s.scenario == "e5"]
    if g.empty:
        return
    g = g.assign(noise=g.condition.str.split("+").str[0], pf=g.condition.str.split("+").str[1])
    for noise, h in g.groupby("noise"):
        fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=200)
        w = .2
        for i, m in enumerate(ORDER):
            d = h[h.method == m].set_index("pf").reindex(["none", "median", "gaussian"])
            ax.bar(np.arange(3) + (i - 1.5) * w, d.f1_mean, w, label=NAMES[m], color=STYLE[m][0])
        ax.set_xticks(range(3)); ax.set_xticklabels(["none", "median", "gaussian"])
        ax.set_xlabel(f"pre-filter ({noise.replace('_', ' ')})"); ax.set_ylabel("Pixel-level F1")
        ax.set_ylim(0, 1.02); ax.legend(fontsize=6); ax.grid(alpha=.3, axis="y")
        fig.tight_layout(); fig.savefig(outdir / f"e5_prefilter_{noise}.png"); plt.close(fig)


def plot_clean(s, outdir):
    """Grafik E1 yang dirujuk oleh laporan, dari ringkasan metrik yang sama."""
    g = s[s.scenario == "e1"].set_index("method").reindex(
        [m for m in ORDER if m in set(s[s.scenario == "e1"].method)])
    if g.empty:
        return
    fig, ax = plt.subplots(figsize=(4.0, 2.6), dpi=200)
    colors = [STYLE[m][0] for m in g.index]
    ax.bar(g.index, g.f1_mean, color=colors)
    ax.set_ylim(0, 1.02); ax.set_ylabel("Pixel-level F1"); ax.set_xlabel("Method")
    ax.grid(alpha=.3, axis="y")
    for i, value in enumerate(g.f1_mean):
        ax.text(i, value + .02, f"{value:.3f}", ha="center", fontsize=7)
    fig.tight_layout(); fig.savefig(outdir / "e1_clean_f1.png"); plt.close(fig)


def select_best_config(summary):
    """Pilih nilai E2 dengan rata-rata F1 terbesar, tanpa mengubah data mentah."""
    from experiments.paper.bench_common import DEFAULT_CONFIG

    config = {method: dict(values) for method, values in DEFAULT_CONFIG.items()}
    choices = {
        ("A", "block_size"): ("block_size", int),
        ("B", "block_size"): ("block_size", int),
        ("DCT", "block_size"): ("block_size", int),
        ("A", "threshold_t"): ("threshold_t", int),
        ("B", "threshold_t"): ("threshold_t", int),
        ("DCT", "threshold_t"): ("threshold_t", int),
        ("SIFT", "sift_ratio"): ("ratio_threshold", float),
        ("SIFT", "sift_min_inliers"): ("min_inliers", int),
    }
    selected = []
    tune = summary[summary.scenario == "e2"]
    for (method, condition), (parameter, cast) in choices.items():
        candidates = tune[(tune.method == method) & (tune.condition == condition)].copy()
        if candidates.empty:
            continue
        candidates["level_num"] = candidates.level.astype(float)
        best = candidates.sort_values(["f1_mean", "level_num"], ascending=[False, True]).iloc[0]
        value = cast(best.level)
        config[method][parameter] = value
        selected.append({"method": method, "condition": condition, "parameter": parameter,
                         "value": value, "f1_mean": float(best.f1_mean),
                         "n_pos": int(best.n_pos)})
    return config, selected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True, nargs="+")
    ap.add_argument("--outdir", type=Path, default=Path("experiments/paper/results/figures"))
    ap.add_argument("--best-config-out", type=Path,
                    default=Path("experiments/paper/results/best_config.json"))
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(p) for p in a.input], ignore_index=True)
    a.outdir.mkdir(parents=True, exist_ok=True)
    err = df[df.error.notna()]
    if len(err):
        print(f"PERINGATAN: {len(err)} deteksi gagal (lihat kolom error).")
        err.groupby(["method", "error"]).size().to_csv(a.outdir / "errors.csv")
    ok = df[df.error.isna()]
    s = summarize(ok)
    s.round(4).to_csv(a.outdir / "summary.csv", index=False)
    clean = s[s.scenario == "e1"]
    if len(clean):
        t = clean.set_index("method").loc[[m for m in ORDER if m in set(clean.method)]]
        cols = ["precision", "recall", "f1_mean", "iou", "tpr_img", "fpr_img", "runtime_ms_median"]
        t[cols].round(3).to_csv(a.outdir / "table_e1_clean.csv")
        (a.outdir / "table_e1_clean.tex").write_text(t[cols].round(3).to_latex())
        print(t[cols].round(3).to_string())
    plot_curves(s, a.outdir); plot_prefilter(s, a.outdir)
    plot_clean(s, a.outdir)
    config, selected = select_best_config(s)
    if selected:
        a.best_config_out.parent.mkdir(parents=True, exist_ok=True)
        a.best_config_out.write_text(json.dumps(config, indent=2) + "\n")
        (a.outdir / "e2_selected_parameters.csv").write_text(
            pd.DataFrame(selected).round(4).to_csv(index=False))
        print("Parameter E2 terpilih ->", a.best_config_out)
    else:
        print("Tidak ada E2 pada input: best_config.json tidak ditulis.")
    print("Keluaran:", a.outdir)


if __name__ == "__main__":
    main()
