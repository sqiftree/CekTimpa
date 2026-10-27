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

## 🧮 Metode & Mesin Deteksi

Sistem CekTimpa menyediakan dua algoritma deteksi dengan abstraksi terpadu (`BaseDetector`):

### 1. 🔬 SIFT + FLANN + RANSAC (*Primary / Recommended Detector*)
*Metode Rujukan: Amerini, Ballan, Caldelli, Del Bimbo, & Serra (IEEE TIFS 2011)*
* **Scale-Space & DoG Extrema:** Mendeteksi titik kunci lokal yang invariant terhadap skala (DoG pyramid).
* **128-Dim Gradient Descriptor:** Memberikan ketahanan kuat terhadap variasi pencahayaan dan derau.
* **FLANN Approximate Matching:** Pencarian pasangan dengan kompleksitas $O(N \log N)$ + *Lowe's Ratio Test*.
* **RANSAC Geometric Verification:** Mengestimasi matriks homografi $H$ ($3 \times 3$) dan memverifikasi konsistensi geometri. Tahan terhadap **rotasi penuh 0°–360°**, **perubahan skala**, dan **perspektif 3D**.

### 2. 🧱 DCT + Bilateral Filter (*Block-based Baseline*)
*Metode Rujukan: Kaplan, Aksakalli, Kilic, & Erer (BAJECE 2020)*
* **Dekomposisi Multiskala Bilateral:** Memisahkan lapisan detail $D^2(I)$ untuk mempertahankan tepi batas.
* **Hibrida DCT 2D:** Transformasi ortogonal frekuensi rendah dengan pemindaian zig-zag ($P=0.25$).
* **Kuantisasi Leksikografis & Voting Vektor:** Mengidentifikasi pola pergeseran translasi murni $(\Delta x, \Delta y)$.
* **Morfologi Matematis:** Operasi *Closing*, *Opening*, *Hole Filling*, dan *Boundary Extraction* ($\beta(A) = A - (A \ominus SE)$).

---

## 📁 Struktur Repositori

```text
CekTimpa/
├── Dockerfile                 # Konfigurasi container Docker multi-platform
├── docker-compose.yml         # Orkestrasi container untuk WSL live-reload
├── requirements.txt           # Dependensi pustaka Python
├── Makefile                   # Shortcut perintah praktis (make test, make up, make run-local)
├── pytest.ini                 # Konfigurasi test runner pytest
├── README.md                  # Dokumentasi proyek
├── data/                      # Dataset & sampel uji
├── src/
│   ├── __init__.py            # Core package exports
│   ├── detectors/             # Clean Architecture Detectors
│   │   ├── __init__.py        # Factory get_detector & detect_copymove
│   │   ├── base.py            # BaseDetector ABC & BaseDetectionResult dataclass
│   │   ├── copymove_sift.py   # SIFT + FLANN + RANSAC detector (Amerini 2011)
│   │   └── copymove_block.py  # DCT + Bilateral block detector (Kaplan 2020)
│   ├── preprocessing.py       # Grayscale luminance & multiscale bilateral filter
│   ├── blocks.py              # Ekstraksi sliding window blok BxB
│   ├── features.py            # Ekstraksi fitur hibrida DCT-Bilateral
│   ├── matching.py            # Kuantisasi leksikografis & voting vektor pergeseran
│   ├── postprocess.py         # Morfologi citra, hole filling, boundary extraction
│   ├── forgery_generator.py   # Generator manipulasi sintetis (tampered, clean_single, clean_multi)
│   ├── attacks.py             # Simulasi degradasi citra (noise, blur, gamma, histeq, JPEG)
│   ├── metrics.py             # Evaluasi kuantitatif (Precision, Recall, F1, IoU, PSNR)
│   └── utils.py               # Overlay warna & penandaan visual
├── tests/                     # Unit test otomatis (20 tests passed)
│   ├── test_bilateral_detection.py
│   ├── test_sift_detection.py
│   └── test_metrics.py
└── app/
    └── main.py                # Dashboard Web Streamlit interaktif
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

## 📂 Sumber Dataset Pengujian & Acuan Benchmark

Untuk pengujian ilmiah dan evaluasi kinerja deteksi copy-move, tim dapat mengunduh citra dari sumber-sumber standar berikut:

### 1. Dataset Khusus Forensik Copy-Move (Dengan Ground Truth Mask)
* **CoMoFoD Database (Rekomendasi Utama):**
  * Tautan Resmi: [vcl.fer.hr/comofod/](http://vcl.fer.hr/comofod/)
  * Kaggle Mirror: [kaggle.com/datasets/shreyaswankhede/comofod-dataset](https://www.kaggle.com/datasets/shreyaswankhede/comofod-dataset)
  * Keterangan: 260 set citra (resolusi $512 \times 512$) dilengkapi mask biner dan variasi serangan (JPEG, noise, blur, kontras, rotasi).
* **CASIA 2.0 Tampered Dataset (Kaggle):**
  * Tautan: [kaggle.com/datasets/sophatvathana/casiav2](https://www.kaggle.com/datasets/sophatvathana/casiav2)
  * Keterangan: Berisi ribuan sampel pemalsuan foto realistis (pilih sub-kategori copy-move).
* **COVERAGE Dataset (Kasus Mirip Objek Asli):**
  * Tautan: [kaggle.com/datasets/divg23/coverage-dataset](https://www.kaggle.com/datasets/divg23/coverage-dataset)
  * Keterangan: 100 foto dengan objek salinan yang ditempel pada area bertekstur serupa.

### 2. Dataset Citra Sumber Asli (Laboratorium Pengolahan Citra Digital)
* **USC-SIPI Image Database:**
  * Tautan: [sipi.usc.edu/database/](https://sipi.usc.edu/database/)
  * Keterangan: Citra uji klasik laboratorium PCD (*Lena, Cameraman, Peppers, Baboon/Mandrill, House*).
* **Unsplash & Pexels (Foto Bebas Royalti Resolusi Tinggi):**
  * Tautan: [unsplash.com](https://unsplash.com/) dan [pexels.com](https://www.pexels.com/)
  * Keterangan: Sangat baik untuk mengambil objek tunggal (mobil, kapal, pohon) atau pola berulang (dinding bata, kain batik) untuk pengujian *false positive*.

---

## Lisensi
Proyek ini didistribusikan di bawah lisensi MIT untuk keperluan akademik dan penelitian Pengolahan Citra Digital.
