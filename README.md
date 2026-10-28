# <p align="center">🔍 CekTimpa: Detektor Copy-Move</p>
<p align="center">
  <em>Deteksi COPY-MOVE: foto yang ditimpa potongan dari foto itu sendiri</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python Version" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Method-Pure%20PCD%20(Non--Deep%20Learning)-success.svg" alt="Pure PCD" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License" />
</p>

---

## 📖 Ringkasan Proyek

**CekTimpa** adalah aplikasi deteksi pemalsuan citra berbasis blok (*block-based copy-move forgery detection*) yang dibangun menggunakan konsep murni **Pengolahan Citra Digital (PCD)** dan matematika konvensional tanpa *deep learning*.

Aplikasi ini dirancang untuk mendeteksi manipulasi **Copy-Move**, yaitu tindakan menyalin satu area pada foto lalu menempelkannya ke area lain pada foto yang sama untuk menduplikasi atau menyembunyikan objek tertentu.

> [!IMPORTANT]
> **Cakupan & Batasan Klaim Ilmiah:**
> * CekTimpa **hanya** mendeteksi pemalsuan tipe **Copy-Move** (salin-tempel dalam satu citra yang sama).
> * Sistem **tidak** dirancang untuk mendeteksi *splicing* (menempelkan objek dari foto lain), manipulasi teks/struk, hasil *inpainting* / generatif AI, maupun filter artistik.
> * Luaran sistem berupa **"Terindikasi copy-move"** atau **"Tidak terindikasi copy-move"**. Status tidak terindikasi bukan jaminan mutlak bahwa citra 100% asli.

---

## Tim Pengembang

**Departemen Informatika, Universitas Sebelas Maret (UNS) Surakarta**  
*Mata Kuliah: Pengolahan Citra Digital*

* **Yashif Victoriawan**
* **Fairuz**
* **Fadhil**
* **Andre**

*(Catatan: Rincian pembagian peran dan modul akan disepakati dan diperbarui bersama oleh tim pengembang).*

---

## 🧮 Metode & Materi Pengolahan Citra Digital

Sistem bekerja melalui tahapan modular berbasis silabus PCD:
1. **Representasi Blok & Neighborhood:** Citra dipartisi menjadi blok bertumpuk ukuran $B \times B$ ($16 \times 16$) dengan pergeseran *stride* $s$.
2. **Ekstraksi Fitur Blok:**
   - **Fitur H (Spasial):** Statistik histogram blok ($\mu, \sigma^2, \mu_3$).
   - **Fitur A (Spasial):** Rata-rata sub-blok $G \times G$ (Box Filtering).
   - **Fitur B (Spasial):** Gaussian lowpass filter + subsampling.
   - **Fitur C (Frekuensi):** Magnitudo Transformasi Fourier 2D frekuensi rendah $|F(u,v)|$.
   - **Fitur D (Frekuensi Modifikasi):** DFT Radial $R(r)$ memanfaatkan sifat rotasi Fourier.
3. **Pencarian Pasangan & Voting Vektor Pergeseran:** Kuantisasi leksikografis, penapisan jarak spasial $D_{min}$, dan akumulasi vektor translasi $\vec{s} = \vec{p}_b - \vec{p}_a$ dengan ambang batas $T$.
4. **Pasca-Pemrosesan Morfologi:** Pembersihan mask deteksi menggunakan operasi *Closing*, *Opening*, *Hole Filling*, serta *Boundary Extraction* ($\beta(A) = A - (A \ominus SE)$) untuk garis batas di UI.

---

## 📁 Struktur Repositori

```text
CekTimpa/
├── Dockerfile                 # Konfigurasi container Docker multi-platform
├── docker-compose.yml         # Orkestrasi container untuk WSL live-reload
├── .dockerignore              # Pengecualian berkas image Docker
├── requirements.txt           # Dependensi pustaka Python
├── Makefile                   # Shortcut perintah praktis (make up, make test, dll.)
├── README.md                  # Dokumentasi proyek
├── .gitignore                 # Konfigurasi pengabaian berkas git
├── data/
│   ├── sources/               # Citra sumber asli
│   └── benchmark/             # Dataset pemalsuan sintetis (forged, authentic, masks)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Grayscale, konversi RGB ke HSI, penapisan awal
│   ├── blocks.py              # Ekstraksi sliding window blok BxB
│   ├── features.py            # Ekstraksi fitur H, A, B, C, D
│   ├── matching.py            # Kuantisasi, pengurutan leksikografis, voting vektor
│   ├── postprocess.py         # Morfologi citra, hole filling, boundary extraction
│   ├── detectors/
│   │   ├── __init__.py
│   │   └── copymove_block.py  # Pipeline deteksi end-to-end
│   ├── forgery_generator.py   # Generator salin-tempel sintetis + ground truth
│   ├── attacks.py             # Simulasi gangguan (noise, blur, gamma, histeq, JPEG)
│   ├── metrics.py             # Evaluasi kuantitatif (P, R, F1 piksel, TPR, FPR, PSNR)
│   └── utils.py               # Fungsi utilitas & visualisasi
├── experiments/
│   ├── __init__.py
│   ├── make_benchmark.py      # Pembuatan dataset uji terstandarisasi
│   ├── run_features.py        # Eksperimen 1: Komparasi fitur tanpa gangguan
│   ├── run_params.py          # Eksperimen 2: Sensitivitas parameter (B, stride, Q, T)
│   ├── run_noise_blur_jpeg.py # Eksperimen 3: Ketahanan terhadap noise, blur, & JPEG
│   ├── run_intensity.py       # Eksperimen 4: Perubahan kontras & gamma
│   ├── run_geometric.py       # Eksperimen 5: Ketahanan rotasi & skala (C vs D)
│   ├── run_color.py           # Eksperimen 6: Perbandingan Grayscale vs RGB vs HSI
│   ├── run_prefilter.py       # Eksperimen 7: Pengaruh pra-penapisan restorasi
│   ├── make_figures.py        # Pembuatan grafik publikasi dari CSV
│   └── results/               # Berkas luaran evaluasi empiris (*.csv)
├── app/
│   └── main.py                # Dashboard aplikasi interaktif Streamlit
└── tests/                     # Pengujian unit otomatis (pytest)
```

---

## ⚙️ Panduan Instalasi & Penggunaan

### 1. Kloning Repositori
```bash
git clone https://github.com/sqiftree/CekTimpa.git
cd CekTimpa
```

### ⚡ Pintasan Cepat (Makefile)
Jika Anda menggunakan **WSL / Linux**, Anda dapat menggunakan perintah praktis berikut:

| Perintah | Deskripsi |
| :--- | :--- |
| **`make up`** | Menjalankan aplikasi di Docker + live-reload (`http://localhost:8501`) |
| **`make down`** | Menghentikan container Docker |
| **`make test-docker`** | Menjalankan pengujian `pytest` di dalam container Docker |
| **`make run-local`** | Menjalankan Streamlit langsung di Python lokal |
| **`make test`** | Menjalankan `pytest` di Python lokal |
| **`make clean`** | Membersihkan cache Python (`__pycache__`) |
| **`make help`** | Menampilkan daftar seluruh perintah yang tersedia |

---

### OPSI A: Menggunakan Docker & WSL (Sangat Direkomendasikan 🐳)
Jika Anda menggunakan **Docker Desktop / WSL**, Anda tidak perlu memasang pustaka OpenCV secara manual di host:

```bash
# Bangun image dan jalankan container
docker compose up --build
```
Aplikasi Streamlit otomatis aktif dan dapat diakses di peramban pada **`http://localhost:8501`**.  
*Fitur volume mount aktif, sehingga perubahan kode pada editor akan langsung live-reload di dalam container.*

Untuk menghentikan container:
```bash
docker compose down
```

---

### OPSI B: Menjalankan Secara Native (Python Virtual Environment)

1. **Buat dan Aktifkan Virtual Environment:**
   ```bash
   python3 -m venv venv

   # Di Linux / WSL / macOS:
   source venv/bin/activate

   # Di Windows CMD/PowerShell:
   venv\Scripts\activate
   ```

2. **Pasang Dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan Aplikasi Web Streamlit:**
   ```bash
   streamlit run app/main.py
   ```
   Aplikasi akan terbuka otomatis di peramban pada `http://localhost:8501`.

---

### 🧪 Menjalankan Pengujian Unit (Unit Testing)

* **Melalui Docker:**
  ```bash
  docker compose run --rm cektimpa pytest tests/
  ```
* **Melalui Lingkungan Lokal:**
  ```bash
  pytest tests/
  ```

---

## Rencana Eksperimen & Tolok Ukur

Sistem ini dievaluasi secara empiris melalui 8 skenario pengujian dengan pembagian dataset 30% *tuning* dan 70% *testing* untuk memastikan reproduksibilitas ilmiah sesuai standar artikel IEEE.

---

<!-- dataset-reference:start -->

## 📂 Sumber Dataset Pengujian & Acuan Benchmark

Untuk pengujian ilmiah dan evaluasi kinerja deteksi copy-move, tim dapat mengunduh citra dari sumber-sumber standar berikut:

### 1. Dataset Khusus Forensik Copy-Move (Dengan Ground Truth Mask)
* **CoMoFoD Database (Rekomendasi Utama):**
  * Tautan Resmi: [vcl.fer.hr/comofod/](http://vcl.fer.hr/comofod/)
  * Kaggle Mirror: [https://www.kaggle.com/datasets/tusharchauhan1898/comofod](https://www.kaggle.com/datasets/tusharchauhan1898/comofod)
  * Keterangan: 260 set citra: 200 set beresolusi 512×512 dan 60 set beresolusi 3000×2000, dilengkapi mask dan variasi gangguan. [Rincian resmi](https://www.vcl.fer.hr/comofod/comofod.html).
* **CASIA 2.0 Tampered Dataset (Kaggle):**
  * Tautan: [https://www.kaggle.com/datasets/santoshanand10203040/casiav2](https://www.kaggle.com/datasets/santoshanand10203040/casiav2)
  * Keterangan: Berisi ribuan sampel pemalsuan foto realistis (pilih sub-kategori copy-move).
* **COVERAGE Dataset (Kasus Mirip Objek Asli):**
  * Tautan: [https://www.kaggle.com/datasets/ma7moud3tef/coverage](https://www.kaggle.com/datasets/ma7moud3tef/coverage)
  * Keterangan: Pasangan citra asli dan manipulasi dengan objek yang mirip secara alami, untuk menguji alarm palsu akibat kemiripan objek. [Penjelasan penyusun](https://github.com/wenbihan/coverage).

### 2. Dataset Citra Sumber Asli (Laboratorium Pengolahan Citra Digital)
* **USC-SIPI Image Database:**
  * Tautan: [sipi.usc.edu/database/](https://sipi.usc.edu/database/)
  * Keterangan: Citra uji klasik laboratorium PCD (*Lena, Cameraman, Peppers, Baboon/Mandrill, House*).
* **Unsplash & Pexels (Foto Bebas Royalti Resolusi Tinggi):**
  * Tautan: [unsplash.com](https://unsplash.com/) dan [pexels.com](https://www.pexels.com/)
  * Keterangan: Sangat baik untuk mengambil objek tunggal (mobil, kapal, pohon) atau pola berulang (dinding bata, kain batik) untuk pengujian *false positive*.


### 3. Demo Sintetis Internal

`generate_synthetic_forgery` membuat objek lingkaran bertekstur 48×48 pada
latar gradien, dengan pilihan salinan identik atau objek tunggal. Mask demo
menandai area sumber dan salinan; demo ini belum mewakili foto nyata.

<!-- dataset-reference:end -->

---

## Lisensi
Proyek ini didistribusikan di bawah lisensi MIT untuk keperluan akademik dan penelitian Pengolahan Citra Digital.

## Benchmark sintetis awal (tanpa deep learning)

Pilihan **DCT** tersedia pada dashboard: DCT-II ortonormal, 16 koefisien
frekuensi rendah dalam urutan zigzag, tanpa DC, dibagi panjang sisi blok.
Pembuangan DC mengurangi pengaruh offset kecerahan tanpa clipping; bukan
jaminan ketahanan terhadap rotasi, skala, atau kompresi.

Jalankan pemeriksaan translasi yang reproducible:

```bash
python -m experiments.run_features --samples 5 --seed 1000 --output experiments/results/synthetic.csv
```

Melalui Docker:

```bash
docker compose run --rm cektimpa python -m experiments.run_features --samples 5
```

Setiap seed menghasilkan citra RGB acak 128×128 dan salinannya dengan patch
32×32 dipindahkan dari (16,16) ke (80,80). Mask menandai **sumber dan target**.
CSV menyimpan konfigurasi, seed, confusion counts, metrik, dan runtime per citra.
Ringkasan pixel F1 merupakan rata-rata pada citra positif; FPR dihitung pada
citra negatif. Pembagian nol pada precision/recall/F1/IoU menghasilkan 0,
termasuk mask kosong vs kosong. Balanced accuracy tidak terdefinisi (`NaN`)
jika hanya satu kelas tersedia. PSNR default memakai rentang intensitas 255.

Percobaan awal (5 sumber, seed 1000–1004):

| Fitur | Pixel F1 | Image recall | Image FPR |
|---|---:|---:|---:|
| A | 1.000 | 1.000 | 0.000 |
| B | 0.860 | 1.000 | 0.000 |
| C | 1.000 | 1.000 | 0.000 |
| D | 1.000 | 1.000 | 0.000 |
| H | 1.000 | 1.000 | 0.000 |
| DCT | 1.000 | 1.000 | 0.000 |

**Ini pemeriksaan dasar, bukan estimasi akurasi foto nyata atau peringkat
fitur.** Semua metode memakai epsilon 0.01 untuk menguji salinan identik;
parameter belum dikalibrasi per fitur. Random texture tidak mewakili pola
berulang alami. Tidak ada noise, JPEG, blur, rotasi, atau skala dalam percobaan
ini. Generator umum `make_forgery` dan sebagian skrip eksperimen masih
berupa kerangka; modul attacks serta helper demo sintetis sudah diambil dari alternate.

Tahap evaluasi berikutnya: pisahkan data berdasarkan citra sumber, pilih
parameter pada set validasi saja, bekukan konfigurasi, kemudian uji CoMoFoD
serta negatif sulit dari COVERAGE. Laporkan metrik piksel dan citra secara
terpisah, waktu/RAM, dan interval kepercayaan berdasarkan citra sumber.
Mode KP kini memakai SIFT/ORB + RANSAC similarity (translasi, rotasi, dan
skala seragam). Kandidat diperluas menjadi hingga 64 tetangga sebelum filter
jarak spasial. `threshold_t` pada KP berarti jumlah minimum pasangan inlier
unik (default 6), bukan voting translasi yang dibagi tiga. `ransac_error`
(default 3 piksel) mengatur toleransi reproyeksi. Model harus memiliki sebaran
2D dan lokasi fisik yang berbeda; maksimal 8 model dengan rasio luas
dukungan target/sumber 0.25–4.0.
`shifts` pada hasil KP hanya merangkum median perpindahan tiap model, bukan
parameter transformasi. Mask KP tetap lingkungan kotak keypoint, sehingga
belum merupakan segmentasi objek penuh. Perluasan mask dengan korelasi lokal
belum diimplementasikan.

Tes mencakup outlier, korespondensi kollinear, salinan berotasi/berskala
setelah JPEG quality 70, negatif sintetis, serta pemilihan DCT dan KP melalui
Streamlit AppTest. Tes ini belum menggantikan evaluasi foto nyata.

Untuk CoMoFoD, `_O_` berarti original (negatif), `_F_` forged (positif), dan
`_B` mask evaluasi. UI menampilkan petunjuk jika nama berkas berpola `_O_`;
nama berkas tidak pernah digunakan untuk menentukan hasil detektor.
Jika UI sudah menampilkan DCT tetapi proses lama menolak fitur tersebut,
jalankan `docker compose restart cektimpa` untuk memuat ulang modul Python.


## Integrasi alternate ke main

Detektor utama dashboard kini **SIFT + FLANN + RANSAC** dengan mask lingkaran
mengikuti skala keypoint. Fitur DCT–Bilateral, helper demo sintetis, visualisasi
pasangan, dan kontrak hasil bersama diadaptasi dari alternate `2a8d920`.
Baseline A/B/C/D/H/DCT dan KP lama tetap tersedia untuk perbandingan.
Similarity merupakan model default; homografi planar tersedia sebagai opsi
eksperimental. Tidak ada deep learning.

Perbandingan terkontrol, keputusan integrasi, keterbatasan, dan cara mengulang
ada di [docs/alternate-comparison.md](docs/alternate-comparison.md).

### Pengaturan dan evaluasi pada dashboard

Pengaturan menampilkan parameter yang berlaku untuk metode terpilih. SIFT
menyediakan rasio deskriptor, batas keypoint, ambang kontras, minimum inlier,
toleransi reproyeksi, radius dukungan mask, dan luas minimum komponen.
Parameter default belum merupakan hasil kalibrasi pada dataset nyata; memilih
metode utama tidak berarti metode itu selalu terbaik untuk setiap citra.
Radius mask memperluas area visual tanpa menambah pasangan yang cocok.

Unggah mask ground truth biner 0/255 melalui bagian evaluasi opsional untuk
melihat precision, recall, F1, IoU, FN, dan FP pada satu citra. Mask harus
menandai sumber **dan** salinan dengan ukuran/orientasi sama; gabungkan dua
mask bila dataset memisahkannya. Mask berwarna dan ukuran berbeda ditolak
untuk evaluasi. Demo sintetis menyertakan mask secara otomatis.

Tab dataset membaca bagian README di antara komentar `dataset-reference:start`
dan `dataset-reference:end` setiap kali aplikasi dijalankan ulang. Perbarui
referensi dataset di bagian tersebut agar dokumentasi dan aplikasi tetap sama.

### Penyesuaian metode pembanding

| Metode | Parameter khusus | Catatan |
|---|---|---|
| A | `g_sub` | Grid harus membagi ukuran blok; dimensi fitur G². |
| B | `gaussian_sigma`, `g_samples`, `gaussian_context` | Default backend kini `block`: Gaussian per blok menghindari pengaruh konteks luar. Pilih `image` untuk mereproduksi baseline lama. |
| C | `low_freq_l`, `fft_normalize` | Ukuran koefisien L²; pembagian B² mengurangi perubahan skala koefisien akibat ukuran blok. |
| D | `radial_bins`, `fft_normalize` | Jumlah cincin dapat diubah; voting tetap translasi sehingga bukan detektor rotasi penuh. |
| H | `feature_scaling` | Mean, variance, skewness berbeda satuan; standardisasi opsional menyeimbangkan komponennya, tanpa memulihkan informasi susunan spasial. |
| DCT | `n_coeffs`, `include_dc` | Default 16 koefisien tanpa DC; menyertakan DC menambah informasi kecerahan. |
| DCT–Bilateral | `p_ratio`, `bilateral_d`, `bilateral_sigma_s`, `bilateral_sigma_r` | Detail bilateral dan jumlah koefisien dapat diuji terpisah. |
| KP | `ratio_threshold`, `n_features`, `contrast_threshold` | Kontrol matcher BF dan ekstraksi SIFT; saat fallback ORB, ambang kontras SIFT tidak digunakan. |

Seluruh fitur blok mendukung `feature_scaling='native'` (default API) atau
`'standardized'`: z-score setiap komponen dari blok citra yang sama, kemudian
dibagi √jumlah dimensi. Komponen konstan tetap berkontribusi nol. Tidak ada
fit pada label atau data uji lintas citra. Standardisasi memakai Q efektif=1
agar Q native, khususnya Q=16 pada bilateral, tidak memperbesar toleransi
standardisasi tanpa sengaja. Epsilon perlu dipilih ulang saat skala berubah.
Di dashboard, H memulai dengan standardisasi; C/D memulai dengan normalisasi
FFT per piksel. API tetap mempertahankan koefisien FFT native secara default
untuk mengulang eksperimen lama.

`window` dan `shift_tol` kini juga tersedia di dashboard. Hasil blok menyimpan
konfigurasi, jumlah blok, dimensi fitur, skala, serta Q efektif pada metadata.
Ukuran blok/grid/koefisien dan parameter numerik yang tidak valid ditolak.

Sesudah perbaikan B, pengulangan lima sumber sintetis seed 1000–1004 dengan
konfigurasi native sebelumnya menghasilkan F1 mask B **1.000**, dibanding
**0.860** dengan Gaussian seluruh citra. A/C/D/H/DCT tetap 1.000 pada uji
translasi identik ini. Hasil per kasus ada di
`experiments/results/synthetic_adjusted.csv`; angka tersebut belum mengukur
noise, JPEG, rotasi, atau foto nyata, dan bukan kalibrasi parameter default UI.
