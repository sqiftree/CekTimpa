# Menjalankan eksperimen paper

Dataset CoMoFoD yang telah disaring berada di `experiments/paper/data/` dengan struktur
`images/forged/`, `images/original/`, dan `masks/`. Loader membaca struktur ini secara rekursif.
Semua perintah dijalankan dari root repo. Jalankan `check_masks` dahulu dan pilih mode
mask berdasarkan bukti yang dicetaknya, bukan asumsi.

## Struktur hasil dan Git

Dataset sumber bersifat lokal dan tidak diunggah ke GitHub: `data/images/`, `data/masks/`,
`data/noise/`, dan `data/screenshots/` sudah tercantum di `.gitignore`. Berkas yang memang
perlu dibagikan untuk paper tetap dapat diunggah:

- `data/hasil_experiment.md`: laporan dan penjelasan hasil;
- `results/e1.csv`, `results/e2_tune.csv`, beserta berkas metadata: data mentah hasil akhir;
- `results/figures/`: tabel turunan dan gambar grafik yang dipakai laporan;
- `results/best_config.json`: konfigurasi yang dibekukan dari E2.

Keluaran `*smoke*.csv` hanya untuk uji cepat dan diabaikan. Untuk menambahkan seluruh kode,
laporan, serta hasil paper yang diizinkan, gunakan:

```bash
git add experiments/paper
git status --short
git diff --cached --stat
```

Pastikan daftar staged tidak memuat `experiments/paper/data/images/` atau
`experiments/paper/data/masks/` sebelum membuat commit.

```bash
python -m experiments.paper.check_masks --data-dir experiments/paper/data --limit 20

# E2, tuning pada 30% set. Ganti B dengan M atau union bila hasil check_masks mengharuskannya.
python -m experiments.paper.run_benchmark --dataset comofod --data-dir experiments/paper/data \
  --mask-mode B --scenarios e2 --subset tune --output experiments/paper/results/e2_tune.csv

# Merangkum E2 dan membekukan parameter yang benar-benar menang pada F1 tuning.
python -m experiments.paper.make_figures --input experiments/paper/results/e2_tune.csv

# E1 pada 70% set uji, termasuk citra asli untuk mengukur FPR.
python -m experiments.paper.run_benchmark --dataset comofod --data-dir experiments/paper/data \
  --mask-mode B --scenarios e1 --subset test --config-json experiments/paper/results/best_config.json \
  --output experiments/paper/results/e1.csv

# E3 sampai E5 pada set uji. Jangan memakai --levels reduced untuk hasil akhir paper.
python -m experiments.paper.run_benchmark --dataset comofod --data-dir experiments/paper/data \
  --mask-mode B --scenarios e3,e4,e5 --subset test --positives-only \
  --config-json experiments/paper/results/best_config.json --output experiments/paper/results/e345.csv

# Grafik, tabel CSV, konfigurasi beku, lalu laporan Markdown berbasis data nyata.
python -m experiments.paper.make_figures --input experiments/paper/results/e2_tune.csv \
  experiments/paper/results/e1.csv experiments/paper/results/e345.csv
python -m experiments.paper.render_report --summary experiments/paper/results/figures/summary.csv \
  --metadata experiments/paper/results/e2_tune.csv.meta.json \
  experiments/paper/results/e1.csv.meta.json experiments/paper/results/e345.csv.meta.json \
  --best-config experiments/paper/results/best_config.json
```

Catatan: stride 1 wajib untuk perbandingan blok yang adil; satu deteksi blok ~5 detik/citra 512x512.

Jangan masukkan `smoke.csv` atau keluaran `--dataset synthetic` ke perintah langkah 6 atau 7.
`render_report` menolak metadata selain `dataset: comofod`, dan hanya merender angka yang
benar-benar ada di `summary.csv`. Narasi ilmiah tetap ditulis manusia setelah memeriksa tabel dan grafik.
