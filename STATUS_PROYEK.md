# Status Proyek CekTimpa

Dokumen ini merangkum kemampuan yang sudah tersedia dan keterbatasan yang
masih perlu diperbaiki. Status tes di bawah mencatat pemeriksaan terakhir
pada 10 November 2026; bukan sertifikasi akurasi untuk seluruh jenis foto.

## Ringkasan

CekTimpa adalah aplikasi Streamlit untuk membantu memeriksa indikasi
**copy-move**: satu area citra disalin dan ditempelkan kembali pada citra yang
sama. Aplikasi menggunakan pengolahan citra konvensional, bukan deep learning.
Hasilnya adalah bukti indikatif, bukan keputusan forensik otomatis.

## Yang sudah tersedia

- Antarmuka untuk mengunggah foto dan menjalankan deteksi, serta demo citra
  sintetis bersih dan copy-move.
- Empat pilihan metode pada dashboard:
  - **SIFT + FLANN + RANSAC** untuk pencocokan keypoint dan verifikasi geometri.
  - **A** untuk membandingkan rata-rata sub-blok.
  - **B** untuk membandingkan sampel blok setelah filter Gaussian.
  - **DCT** untuk membandingkan deskripsi koefisien kosinus frekuensi rendah.
- Pengaturan awal adaptif berdasarkan ukuran dan tekstur citra, serta pilihan
  pengaturan manual. Nilai adaptif adalah heuristik, bukan parameter yang
  dijamin optimal.
- Pipeline blok dengan filter tekstur, pencarian kandidat, batas jarak,
  pemungutan suara vektor translasi, dan pembuatan mask prediksi.
- Verifikasi geometri SIFT dengan model similarity sebagai pilihan utama dan
  homography planar sebagai opsi eksperimental.
- Evaluasi pixel-level terhadap ground truth memakai precision, recall, F1,
  IoU, false positive, dan false negative.
- Mask ground truth JPG/JPEG, PNG, dan TIFF dapat diunggah. JPEG diubah menjadi
  grayscale lalu dibinarisasi pada ambang 128; PNG/TIFF harus memiliki nilai
  biner 0/255.
- Implementasi metode pembanding lain masih tersimpan untuk pengujian dan
  eksperimen meski tidak ditampilkan dalam pilihan utama dashboard.
- Tes otomatis mencakup pipeline, metode, geometri, metrik, impor modular, dan
  integrasi dashboard. Pemeriksaan terakhir: **73 passed, 1 skipped**.

## Kekurangan dan batasan yang diketahui

### Akurasi belum dibuktikan untuk foto umum

- Tes sintetis membuktikan perilaku pada skenario terbatas, bukan akurasi
  lapangan. Dataset benchmark nyata belum disertakan di repositori dan
  parameter belum dikalibrasi secara menyeluruh pada data validasi terpisah.
- Pada contoh foto penyeberangan yang diperiksa, dengan preset yang diuji,
  metode A dan B tidak menemukan pasangan yang lolos; DCT menghasilkan banyak
  kecocokan yang tidak relevan; SIFT menemukan kecocokan tetapi ikut
  mencocokkan orang lain yang tampak serupa. Hasil contoh ini tidak
  memprediksi kinerja pada foto lain.
- Copy-move kecil, blur, kompresi, perubahan pencahayaan, tekstur lemah,
  pengulangan alami, atau perubahan skala/perspektif dapat menyebabkan hasil
  terlewat atau kecocokan palsu.
- Metode blok terutama memverifikasi pola translasi. A/B/DCT bukan pengganti
  verifikasi geometri transformasi umum.

### Mask dan evaluasi

- Mask keluaran adalah perkiraan dukungan pasangan blok atau keypoint; mask
  SIFT khususnya tidak menjamin mengikuti siluet objek secara utuh.
- Ground truth **tidak** membantu proses deteksi. Ground truth hanya
  dibandingkan dengan hasil sesudah detektor berjalan untuk menghitung metrik.
- Evaluasi mengharapkan ground truth dan foto memiliki ukuran serta orientasi
  sama. Jika dataset menyimpan area sumber dan target dalam dua mask terpisah,
  mask tersebut perlu digabungkan terlebih dahulu.
- JPEG bersifat lossy. Ambang 128 dapat mengubah batas atau piksel tipis;
  gunakan PNG biner 0/255 bila tersedia untuk evaluasi yang lebih presisi.

### Cakupan fitur dan eksperimen

- Sistem berfokus pada copy-move; tidak ditujukan untuk splicing, inpainting,
  manipulasi generatif, atau menentukan keaslian foto secara keseluruhan.
- Generator umum `make_forgery` di `src/forgery_generator.py` masih berupa
  kerangka dan melempar `NotImplementedError`. Demo sintetis menggunakan
  `generate_synthetic_forgery`, yang merupakan fungsi berbeda.
- Belum ada laporan benchmark nyata lengkap yang membandingkan precision,
  recall, F1, runtime, dan false-positive pada beragam jenis citra.
- Pilihan preset adaptif adalah titik awal praktis, bukan autotuning yang
  menggunakan ground truth atau menjamin hasil terbaik.

## Prioritas pengembangan berikutnya

1. Jalankan evaluasi terpisah pada dataset copy-move nyata, termasuk citra
   negatif dan kasus sulit; simpan konfigurasi, metrik, serta hasil per metode.
2. Kalibrasi parameter hanya pada set validasi, lalu laporkan hasil pada set
   uji yang tidak dipakai saat memilih parameter.
3. Perbaiki pemisahan objek copy dari objek alami yang mirip, terutama pada
   foto kerumunan dan pola berulang.
4. Tingkatkan konstruksi mask agar dapat mengikuti batas objek, dan validasi
   mask terhadap siluet ground truth tanpa mengubah ukuran secara diam-diam.
5. Implementasikan atau hapus kerangka `make_forgery` agar API generator umum
   tidak tampak siap digunakan padahal belum bekerja.
6. Dokumentasikan batas ukuran, format, parameter awal, serta cara interpretasi
   hasil secara konsisten pada antarmuka dan README.

## Interpretasi hasil

**Terindikasi copy-move** berarti sistem menemukan pola kecocokan yang lolos
ambang metode terpilih; pola itu masih dapat berasal dari objek atau tekstur
alami yang mirip. **Tidak terindikasi copy-move** hanya berarti bukti yang
dibutuhkan tidak ditemukan oleh metode dan pengaturan tersebut. Keduanya
perlu ditinjau bersama citra asli, overlay, konfigurasi, dan ground truth bila
tersedia.
