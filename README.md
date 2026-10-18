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

## 👥 Tim Pengembang

**Departemen Informatika, Universitas Sebelas Maret (UNS) Surakarta**  
*Mata Kuliah: Pengolahan Citra Digital*

| Nama | NIM | Peran Utama |
| :--- | :--- | :--- |
| **Yashif Victoriawan** | - | Pengembangan Frontend Streamlit |
| **Fairuz** | - | Anggota Tim Pengembang |
| **Fadhil** | - | Anggota Tim Pengembang |
| **Andre** | - | Anggota Tim Pengembang |

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

## 🔬 Rencana Eksperimen & Tolok Ukur

Sistem ini dievaluasi secara empiris melalui 8 skenario pengujian dengan pembagian dataset 30% *tuning* dan 70% *testing* untuk memastikan reproduksibilitas ilmiah sesuai standar artikel IEEE.

---

## 📜 Lisensi
Proyek ini didistribusikan di bawah lisensi MIT untuk keperluan akademik dan penelitian Pengolahan Citra Digital.
