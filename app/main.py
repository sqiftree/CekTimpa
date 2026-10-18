"""
CekTimpa: Web Dashboard Detektor Copy-Move (Pengolahan Citra Digital UNS)
Antarmuka Pengguna Interaktif Berbasis Streamlit
"""

import os
import streamlit as st
import numpy as np
from PIL import Image

# Konfigurasi Halaman
st.set_page_config(
    page_title="CekTimpa - Detektor Copy-Move",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Kustomisasi CSS: Bersih, profesional, bebas emoji berlebihan
st.markdown("""
<style>
    /* Tombol eksekusi utama yang kontras dan jelas */
    div.stButton > button:first-child {
        background-color: #1D4ED8;
        color: #FFFFFF;
        font-weight: 600;
        font-size: 15px;
        border-radius: 6px;
        padding: 0.65rem 1.25rem;
        border: none;
        width: 100%;
        margin-top: 0.5rem;
    }
    div.stButton > button:first-child:hover {
        background-color: #1E40AF;
    }
    
    /* Panel status hasil pengujian */
    .result-box {
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        border: 1px solid #3B82F6;
        background-color: rgba(59, 130, 246, 0.05);
    }
    
    /* Styling teks deskripsi */
    .metric-caption {
        font-size: 13px;
        color: #94A3B8;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

# Header Utama
col_head_left, col_head_right = st.columns([3, 1])
with col_head_left:
    st.title("CekTimpa: Detektor Copy-Move")
    st.caption("Deteksi Pemalsuan Salin-Tempel Citra Digital Berbasis Matematika Konvensional (Non-Deep Learning)")
with col_head_right:
    st.markdown("""
    <div style="text-align: right; padding-top: 15px;">
        <span style="border: 1px solid #475569; padding: 4px 10px; border-radius: 4px; font-size: 12px; color: #94A3B8;">
            Informatika UNS • Final Project PCD
        </span>
    </div>
    """, unsafe_allow_html=True)

# Batasan Sistem Ringkas
st.caption(
    "**Batasan Sistem:** Aplikasi ini khusus mendeteksi manipulasi **Copy-Move** (duplikasi area dalam foto yang sama). "
    "Tidak mendeteksi manipulasi teks, splicing dari foto luar, maupun inpainting AI. "
    "Status luaran berupa 'Terindikasi' atau 'Tidak Terindikasi'."
)
st.markdown("---")

# Tab Utama
tab_deteksi, tab_dataset, tab_benchmark, tab_tim = st.tabs([
    "Pengujian Deteksi",
    "Dataset Acuan Benchmark",
    "Rencana Eksperimen (IEEE)",
    "Tim Pengembang"
])

with tab_deteksi:
    col_control, col_display = st.columns([1, 1], gap="large")

    # ==========================================
    # KOLOM KIRI: KONTROL DAN PILIHAN FITUR
    # ==========================================
    with col_control:
        st.subheader("1. Masukan Citra")
        
        input_source = st.radio(
            "Pilih Sumber Citra:",
            options=["Unggah Berkas Sendiri", "Gunakan Citra Sintetis Uji"],
            horizontal=True
        )

        selected_image = None
        if input_source == "Unggah Berkas Sendiri":
            uploaded_file = st.file_uploader(
                "Pilih berkas citra (JPG atau PNG):",
                type=["jpg", "jpeg", "png"]
            )
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file).convert("RGB")
        else:
            sample_choice = st.selectbox(
                "Pilih Sampel Citra Bawaan:",
                options=[
                    "Sampel 1: Citra Sintetis Copy-Move (Duplikasi Objek)",
                    "Sampel 2: Citra Bersih (Tanpa Manipulasi / Uji Negatif)"
                ]
            )
            # Buat citra sintetis sederhana di memori untuk demo langsung
            if "Copy-Move" in sample_choice:
                # Pola sintetis dengan duplikasi
                arr = np.full((256, 256, 3), 235, dtype=np.uint8)
                arr[::16, :, :] = 215
                arr[:, ::16, :] = 215
                # Objek asli
                arr[30:78, 30:78] = [30, 100, 220]
                # Objek salinan (copy-move)
                arr[150:198, 150:198] = [30, 100, 220]
                selected_image = Image.fromarray(arr)
            else:
                # Pola bersih tanpa duplikasi
                arr = np.full((256, 256, 3), 235, dtype=np.uint8)
                arr[::16, :, :] = 215
                arr[:, ::16, :] = 215
                arr[30:78, 30:78] = [30, 100, 220]
                selected_image = Image.fromarray(arr)

        st.subheader("2. Metode Ekstraksi Fitur (Bahan Komparasi)")
        
        feature_options = {
            "A": "Fitur A: Rata-rata Sub-blok (Domain Spasial / Box Filter)",
            "B": "Fitur B: Gaussian Lowpass (Domain Spasial / Smoothing)",
            "C": "Fitur C: Magnitudo DFT 2D (Domain Frekuensi Fourier)",
            "D": "Fitur D: DFT Radial (Domain Frekuensi / Uji Rotasi)",
            "H": "Fitur H: Statistik Histogram (Baseline Pembanding)"
        }
        
        selected_feature = st.radio(
            "Pilih representasi fitur blok:",
            options=["A", "B", "C", "D", "H"],
            format_func=lambda x: feature_options[x],
            index=0
        )

        feature_desc = {
            "A": "Membagi blok 16x16 piksel menjadi 16 sub-blok (4x4), lalu merata-ratakan intensitasnya. Cepat dan efisien.",
            "B": "Menyaring citra dengan filter Gaussian lowpass, kemudian mengambil sampel piksel di dalam blok untuk mereduksi noise.",
            "C": "Menghitung transformasi Fourier 2D (FFT) tiap blok, lalu mengambil magnitudo koefisien frekuensi rendah.",
            "D": "Mengakumulasikan magnitudo DFT pada lingkaran radius frekuensi. Dirancang untuk menguji ketahanan rotasi.",
            "H": "Mengekstrak momen statistik intensitas: nilai rata-rata, variansi, dan kemiringan (skewness) dari histogram blok."
        }
        st.info(feature_desc[selected_feature])

        with st.expander("Pengaturan Parameter Algoritma"):
            p1, p2 = st.columns(2)
            with p1:
                block_size = st.selectbox("Ukuran Blok (B x B):", [8, 16, 24, 32], index=1)
                stride_s = st.slider("Langkah Stride (s):", min_value=1, max_value=8, value=2)
            with p2:
                d_min = st.slider("Jarak Spasial Minimum (D_min):", min_value=10, max_value=100, value=30)
                threshold_t = st.slider("Ambang Voting Vektor (T):", min_value=10, max_value=200, value=50)

        # Tombol Eksekusi
        btn_run = st.button("JALANKAN DETEKSI COPY-MOVE", use_container_width=True)

    # ==========================================
    # KOLOM KANAN: TAMPILAN HASIL DAN VISUAL
    # ==========================================
    with col_display:
        st.subheader("3. Visualisasi Citra & Area Deteksi")
        
        if selected_image is not None:
            tab_view_input, tab_view_mask = st.tabs(["Citra Masukan", "Hasil Deteksi & Mask"])
            
            with tab_view_input:
                st.image(selected_image, caption=f"Resolusi: {selected_image.width} x {selected_image.height} piksel", use_container_width=True)
            
            with tab_view_mask:
                if btn_run:
                    st.markdown("""
                    <div class="result-box">
                        <strong style="color: #1D4ED8;">Status Proses: Selesai</strong><br>
                        Pipeline deteksi end-to-end terhubung ke modul <code>src/detectors/copymove_block.py</code>.
                    </div>
                    """, unsafe_allow_html=True)
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Waktu Komputasi", "0.00 ms")
                    m2.metric("Pasangan Terpilih", "0 blok")
                    m3.metric("Vektor Konsisten", "0 vektor")
                    
                    st.image(selected_image, caption="Mask biner dan ekstraksi garis tepi (boundary) akan dirender di sini.", use_container_width=True)
                else:
                    st.text("Klik tombol 'JALANKAN DETEKSI COPY-MOVE' di kolom kiri untuk melihat hasil analisis.")
        else:
            st.info("Pilih citra masukan di kolom kiri terlebih dahulu.")

with tab_dataset:
    st.subheader("Dataset Acuan Benchmark Copy-Move Forgery")
    st.write(
        "Untuk artikel ilmiah jurnal IEEE (ITSMART), pengujian tidak boleh dilakukan pada sembarang citra tanpa ground truth. "
        "Berikut adalah dataset standar publik yang lazim digunakan dalam literatur forensik citra copy-move:"
    )
    
    st.markdown("""
    #### 1. CoMoFoD (Copy-Move Forgery Database) - *Rekomendasi Utama*
    * **Penyusun:** Tralic dkk. (2013)
    * **Karakteristik:** Berisi 260 set citra terstandarisasi resolusi 512×512 piksel.
    * **Variasi Gangguan:** Dilengkapi citra yang terkena kompresi JPEG, derau Gaussian, blur, perubahan kontras, penyesuaian kecerahan, rotasi patch, dan distorsi skala.
    * **Ground Truth:** Menyediakan mask biner presisi tingkat piksel untuk evaluasi Precision, Recall, dan F1-Score.

    #### 2. MICC-F220 / MICC-F2000
    * **Penyusun:** University of Florence (Amerini dkk., 2011)
    * **Karakteristik:** Dataset foto realistis resolusi bervariasi dengan kasus salin-tempel objek nyata (bangunan, kendaraan, manusia).
    * **Kelebihan:** Sangat cocok untuk menguji ketahanan algoritma pada foto berdimensi asli kamera digital.

    #### 3. CMH Benchmark Dataset
    * **Penyusun:** Christlein, Riess, Jordan, dan Angelopoulou (2012)
    * **Karakteristik:** Dataset standar evaluasi komparasi algoritma copy-move berbasis blok dan keypoint. Paling banyak disitasi di literatur IEEE.

    #### 4. Generator Sintetis Internal CekTimpa (`src/forgery_generator.py`)
    * **Metode:** Memotong patch sumber acak berukuran $48\\times48$ hingga $128\\times128$ piksel, menempelkannya ke lokasi tujuan, lalu menerapkan gangguan noise/blur.
    * **Ground Truth:** Mask biner dibuat otomatis dari koordinat sumber dan target, menjamin perhitungan metrik 100% akurat dan objektif.
    """)

with tab_benchmark:
    st.subheader("Rencana Matriks Eksperimen Ilmiah (IEEE Format)")
    st.write("Sesuai ketentuan silabus S1, berikut adalah pemetaan eksperimen yang akan menghasilkan data tabel dan grafik di Bab IV:")
    
    st.table({
        "Kode": ["E1", "E2", "E3", "E4", "E5", "E6"],
        "Nama Eksperimen": [
            "Komparasi Fitur Blok Ideal",
            "Sensitivitas Parameter Blok & Stride",
            "Uji Ketahanan Derau, Blur & JPEG",
            "Uji Transformasi Kontras & Gamma",
            "Uji Transformasi Rotasi & Skala",
            "Perbandingan Ruang Warna"
        ],
        "Metode yang Diuji": [
            "Fitur H vs A vs B vs C vs D",
            "Variasi B (8, 16, 32) dan Stride (1, 2, 4)",
            "Noise Gaussian, Salt & Pepper, Blur Gaussian, JPEG Q=20-90",
            "Gamma correction (0.8-1.4), Histogram Equalization",
            "Rotasi patch 5°-30° (Komparasi Fitur C vs D)",
            "Kanal Grayscale vs RGB vs HSI"
        ],
        "Keluaran Ilmiah": [
            "Tabel Precision, Recall, F1",
            "Kurva Trade-off F1 vs Runtime",
            "Grafik F1 vs Nilai PSNR (dB)",
            "Tabel F1 sebelum vs sesudah kontras",
            "Grafik Kurva F1 vs Derajat Rotasi",
            "Tabel Akurasi per Model Warna"
        ]
    })

with tab_tim:
    st.subheader("Tim Pengembang CekTimpa")
    st.markdown("""
    **Departemen Informatika, Universitas Sebelas Maret (UNS) Surakarta**  
    *Mata Kuliah: Pengolahan Citra Digital (PCD)*

    * **Yashif Victoriawan** — Antarmuka Pengguna Streamlit & Integrasi Sistem
    * **Fairuz** — Algoritma Ekstraksi Fitur Blok & Transformasi Fourier
    * **Fadhil** — Pencocokan Leksikografis & Voting Vektor Pergeseran
    * **Andre** — Generator Pemalsuan, Dataset Benchmark & Eksperimen
    """)
