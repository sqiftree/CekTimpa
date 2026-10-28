# Perbandingan alternate dan main

Perbandingan dilakukan pada 28 Oktober 2026. Branch aktif tetap `main`.
`alternate` diperiksa pada commit `2a8d9206425b1b185e6dadd5ec9b8bd65605e419`;
fetch HTTPS mengonfirmasi commit tersebut masih merupakan ujung branch remote.
Baseline main mencakup perubahan lokal DCT, metrik, dan KP dari sesi sebelumnya,
bukan hanya commit Git `0cb8071`. Perubahan lokal tersebut dipertahankan.

## Keputusan integrasi

| Aspek | Main sebelum integrasi | Alternate | Hasil gabungan |
|---|---|---|---|
| Keypoint | SIFT/ORB + BF, parameter terbatas | SIFT terpisah, FLANN, parameter lengkap | SIFT + FLANN sebagai pilihan utama; KP lama tetap baseline |
| Geometri | Similarity RANSAC, pasangan unik, dukungan fisik 2D | Homografi iteratif, pemeriksaan dukungan lebih longgar | Multi-model similarity default; homografi planar opsional, pemeriksaan degenerasi dan lokasi unik |
| Mask | Kotak ukuran tetap di sekitar titik | Lingkaran mengikuti skala, closing | Mask mengikuti skala; filter area diperbaiki |
| Fitur blok | A/B/C/D/H dan DCT | DCT–Bilateral menggantikan fitur lama | DCT–Bilateral ditambahkan; seluruh baseline dipertahankan |
| Bukti visual | Overlay dan jumlah pasangan | Gambar keypoint, kandidat, inlier | Visualisasi pasangan dan transformasi tersedia di dashboard |
| Evaluasi | Validasi input dan konvensi metrik eksplisit | Attacks serta demo sintetis sudah terisi | Metrik main dipertahankan; attacks dan demo diadaptasi |

Perbaikan penting terhadap alternate:

- Reciprocal matches dihitung sekali. Beberapa orientasi SIFT pada titik yang
  sama tidak dianggap banyak lokasi fisik independen.
- Filter area selalu mengembalikan mask bersih; sebelumnya jika seluruh
  komponen terlalu kecil, mask awal justru dipertahankan.
- Tidak menyimpulkan mana sumber/target hanya dari query/train index.
- Homografi adalah model bidang; tidak diklaim invariant terhadap semua
  perubahan sudut pandang objek 3D.
- Epsilon DCT–Bilateral tetap dalam satuan koefisien asli seperti alternate.
  Main mengukur jarak setelah pembagian Q, sehingga diperlukan konversi
  epsilon/Q. Q pada fitur lama tetap memakai konvensi main.
- Vektor blok diurutkan berdasarkan jumlah pasangan pendukung seperti
  alternate, dengan pengurutan deterministik saat suara sama.
- Filter jarak kandidat FLANN divektorisasi untuk mengurangi biaya loop Python.

`detect_copymove(image, method='sift')` memakai mesin utama.
`method='bilateral'`/`'dct_bilateral'` memakai default B=8, Q=16, epsilon=5,
min_std=3 dan lexsort. `method='block'` mempertahankan pipeline A/B/C/D/H/DCT.
Pemilihan `feature='DCT_BILATERAL'` langsung juga didukung di pipeline blok.

## Pengukuran awal

Tiga sumber sintetis (seed 57–59), resolusi 384×384, masing-masing mempunyai
satu negatif dan tiga positif. Seluruh input dikompresi JPEG quality 70.
Positif memakai patch 160×160 dengan translasi, rotasi 20°/skala 1.1, atau
rotasi 45°/skala 0.85. Total 9 positif dan 3 negatif per pipeline.

| Ukuran | Main sebelum integrasi | Alternate | Gabungan |
|---|---:|---:|---:|
| Rata-rata F1 dukungan patch, translasi | 0.9512 | 0.9544 | 0.9563 |
| Rata-rata F1, rotasi 20°/skala 1.1 | 0.8275 | 0.8689 | 0.8829 |
| Rata-rata F1, rotasi 45°/skala 0.85 | 0.6279 | 0.7530 | 0.7875 |
| Rata-rata F1 seluruh positif | 0.8022 | 0.8588 | 0.8755 |
| Alarm palsu pada negatif | 0/3 | 0/3 | 0/3 |
| Rata-rata waktu seluruh kasus | 278.1 ms | 71.4 ms | 138.0 ms |

Hasil ini mendukung adopsi mask mengikuti skala dari alternate. Gabungan
memberi rata-rata cakupan terbaik di percobaan ini; alternate tetap lebih
cepat. Gabungan juga lebih cepat dari baseline main pada pengukuran ini.
Ini **perbandingan pipeline**, bukan bukti bahwa satu perubahan tunggal
menyebabkan kenaikan: jumlah keypoint, ambang kontras, matching, verifikasi,
dan pembentukan mask berbeda. Beberapa sampel individual bisa lebih buruk.

Keterbatasan: mask GT merupakan **kotak dukungan patch**, termasuk border
refleksi saat warp, bukan segmentasi objek presisi. Tekstur sintetis dan tiga
negatif terlalu sedikit untuk menyimpulkan akurasi foto nyata atau FPR umum.
Belum ada kalibrasi independen, interval kepercayaan, evaluasi COVERAGE,
atau pengukuran RAM. Waktu merupakan satu pengulangan per kasus dalam
container yang sama secara berurutan; dipengaruhi perangkat dan beban host.
Tes COVERAGE yang diadaptasi dari alternate dilewati karena data tidak tersedia.

Data setiap kasus dan ringkasannya tersimpan di
[alternate-comparison-results.json](alternate-comparison-results.json).
Raw CSV ada di `experiments/results/compare_*.csv` (hasil eksperimen diabaikan Git).

## Mengulang percobaan

```bash
python experiments/compare_branches.py --root . --variant integrated --output experiments/results/compare_integrated.csv
```

Untuk alternate, ekspor commit yang disebut di atas ke direktori terpisah,
lalu berikan direktori itu melalui `--root` dan pilih `--variant alternate`.
Untuk baseline historis, gunakan snapshot main sebelum integrasi dan pilih
`--variant main_before`; opsi tersebut memanggil fitur KP lama. Jalankan setiap
pipeline di proses Python terpisah agar modul antarbranch tidak tercampur.

Konfigurasi pembanding utama: min_inliers=6, ratio=0.78, d_min=40,
kp_radius=10, reprojection_error=3. Baseline KP memakai threshold_t=6,
d_min=40, block_size=16. Default parameter lain mengikuti pipeline masing-masing.
Dashboard memakai rasio default 0.75, sehingga angka ini tidak otomatis
mewakili setiap kombinasi parameter dashboard.
