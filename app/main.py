"""
CekTimpa: Web Dashboard Detektor Copy-Move (Pengolahan Citra Digital UNS)
Antarmuka Pengguna Interaktif Berbasis Streamlit dengan Filter Area Polos & Fokus Objek 2D
"""

import sys
import os
import streamlit as st
import numpy as np
from PIL import Image

# Pastikan path modul src/ dapat diimpor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detectors.copymove_block import detect_copymove_block

# Konfigurasi Halaman
st.set_page_config(
    page_title="CekTimpa - Detektor Copy-Move",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Kustomisasi CSS: Bersih, profesional, bebas emoji berlebihan
st.markdown("""
<style>
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
    .result-box {
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        border: 1px solid;
    }
    .result-alert {
        border-color: #EF4444;
        background-color: rgba(239, 68, 68, 0.08);
    }
    .result-safe {
        border-color: #10B981;
        background-color: rgba(16, 185, 129, 0.08);
    }
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

st.caption(
    "**Batasan Sistem:** Aplikasi ini khusus mendeteksi manipulasi **Copy-Move** (duplikasi area dalam foto yang sama). "
    "Dilengkapi penyaringan variansi blok dan gradien 2D untuk mengabaikan latar polos (seperti langit/tembok) serta memfokuskan deteksi pada objek bertekstur."
)
st.markdown("---")

tab_deteksi, tab_dataset, tab_benchmark, tab_tim = st.tabs([
    "Pengujian Deteksi",
    "Dataset Acuan Benchmark",
    "Rencana Eksperimen (IEEE)",
    "Tim Pengembang"
])

with tab_deteksi:
    col_control, col_display = st.columns([1, 1], gap="large")

    with col_control:
        st.subheader("1. Masukan Citra")

        input_source = st.radio(
            "Pilih Sumber Citra:",
            options=["Unggah Berkas Sendiri", "Gunakan Citra Sintetis Uji"],
            horizontal=True
        )

        selected_image = None
        is_synthetic_demo = (input_source == "Gunakan Citra Sintetis Uji")

        if not is_synthetic_demo:
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
                    "Sampel 1: Citra Sintetis Copy-Move (Duplikasi Objek Bertekstur)",
                    "Sampel 2: Citra Bersih (Tanpa Manipulasi / Uji Negatif)"
                ]
            )
            arr = np.full((256, 256, 3), 235, dtype=np.uint8)
            arr[::16, :, :] = 215
            arr[:, ::16, :] = 215

            # Buat pola objek sintetis bertekstur 2D di (30, 30)
            yy, xx = np.meshgrid(np.arange(48), np.arange(48), indexing="ij")
            pattern = ((np.sin(yy / 3.0) * np.cos(xx / 3.0) + 1.0) * 80).astype(np.uint8)
            obj_patch = np.stack([30 + pattern // 2, 80 + pattern, np.full_like(pattern, 220)], axis=-1)
            arr[30:78, 30:78] = obj_patch

            if "Copy-Move" in sample_choice:
                arr[150:198, 150:198] = obj_patch

            selected_image = Image.fromarray(arr)

        st.subheader("2. Metode Ekstraksi Fitur (Bahan Komparasi)")

        feature_options = {
            "A": "Fitur A: Rata-rata Sub-blok (Domain Spasial / Box Filter)",
            "B": "Fitur B: Gaussian Lowpass (Domain Spasial / Smoothing)",
            "C": "Fitur C: Magnitudo DFT 2D (Domain Frekuensi Fourier)",
            "D": "Fitur D: DFT Radial (Domain Frekuensi / Uji Rotasi)",
            "H": "Fitur H: Statistik Histogram (Baseline Pembanding)",
            "KP": "Fitur KP: Keypoint SIFT/ORB Konvensional (Khusus Objek Kompleks/Manusia)"
        }

        selected_feature = st.radio(
            "Pilih representasi fitur blok:",
            options=["A", "B", "C", "D", "H", "KP"],
            format_func=lambda x: feature_options[x],
            index=0
        )

        feature_desc = {
            "A": "Membagi blok menjadi 16 sub-blok (4x4), lalu merata-ratakan intensitasnya. Cepat dan efektif untuk duplikasi objek.",
            "B": "Menyaring citra dengan filter Gaussian lowpass, kemudian mengambil sampel piksel di dalam blok untuk meredam derau halus.",
            "C": "Menghitung transformasi Fourier 2D (FFT) tiap blok, lalu mengambil magnitudo koefisien frekuensi untuk ketahanan geser.",
            "D": "Mengakumulasikan magnitudo DFT pada lingkaran radius frekuensi. Dirancang untuk menguji ketahanan awal terhadap rotasi.",
            "H": "Mengekstrak momen statistik intensitas: nilai rata-rata, variansi, dan kemiringan (skewness) dari histogram blok.",
            "KP": "Mendeteksi titik sudut/tekstur objek secara adaptif (SIFT/ORB) tanpa terikat kotak persegi. Sangat disarankan untuk mendeteksi duplikasi objek manusia atau objek yang mengalami sedikit perubahan skala."
        }
        st.info(feature_desc[selected_feature])

        with st.expander("Pengaturan Parameter Algoritma & Filter Objek"):
            p1, p2 = st.columns(2)
            with p1:
                block_size = st.selectbox("Ukuran Blok (B x B):", [8, 12, 16, 24, 32], index=0 if not is_synthetic_demo else 2)
                stride_s = st.slider("Langkah Stride (s):", min_value=1, max_value=8, value=2)
                q_step = st.number_input("Kuantisasi Fitur (Q):", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
                min_std = st.slider(
                    "Filter Area Polos / Langit (Min Std):",
                    min_value=0.0, max_value=40.0,
                    value=0.0 if is_synthetic_demo else 10.0,
                    step=1.0,
                    help="Membuang blok latar belakang polos seperti langit atau tembok rata agar tidak terdeteksi sebagai false positive."
                )
                min_grad_2d = st.slider(
                    "Filter Fokus Objek 2D (Min Gradien 2D):",
                    min_value=0.0, max_value=20.0,
                    value=0.0 if is_synthetic_demo else 3.0,
                    step=0.5,
                    help="Menekan false positive pada garis lurus 1D (seperti lis gedung atau zebra cross) dan memfokuskan pencarian pada objek bertekstur 2D."
                )
            with p2:
                d_min = st.slider("Jarak Spasial Minimum (D_min):", min_value=10, max_value=100, value=30)
                threshold_t = st.slider("Ambang Voting Vektor (T):", min_value=3, max_value=200, value=15 if not is_synthetic_demo else 25)
                eps = st.number_input("Toleransi Jarak Fitur (Epsilon):", min_value=0.1, max_value=25.0, value=4.0, step=0.5)
                match_method = st.selectbox(
                    "Metode Pencocokan Fitur:",
                    options=["kdtree", "lexsort"],
                    format_func=lambda m: "KD-Tree Nearest Neighbor (Akurat)" if m == "kdtree" else "Lexicographical Sort (Klasik)"
                )
                reject_collinear = st.checkbox(
                    "Tolak Klaster Garis Lurus 1D (Anti-Garis Gedung/Zebra Cross)",
                    value=(not is_synthetic_demo)
                )

        btn_run = st.button("JALANKAN DETEKSI COPY-MOVE", use_container_width=True)

    with col_display:
        st.subheader("3. Visualisasi Citra & Area Deteksi")

        if selected_image is not None:
            image_np = np.array(selected_image)

            if btn_run:
                config = {
                    "feature": selected_feature,
                    "block_size": int(block_size),
                    "stride": int(stride_s),
                    "d_min": float(d_min),
                    "threshold_t": int(threshold_t),
                    "q_step": float(q_step),
                    "eps": float(eps),
                    "area_min": 60,
                    "prefilter": "none",
                    "min_std": float(min_std),
                    "min_grad_2d": float(min_grad_2d),
                    "match_method": match_method,
                    "shift_tol": 2,
                    "reject_collinear": bool(reject_collinear)
                }

                with st.spinner("Memproses ekstraksi blok bertekstur, fitur, dan pencocokan..."):
                    result = detect_copymove_block(image_np, config)

                if result.is_forged:
                    st.markdown(f"""
                    <div class="result-box result-alert">
                        <strong style="color: #DC2626;">Status: Terindikasi Copy-Move</strong><br>
                        Ditemukan <strong>{len(result.shifts)} pola translasi pergeseran</strong> dengan <strong>{result.total_matches} pasangan blok cocok</strong>.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box result-safe">
                        <strong style="color: #059669;">Status: Tidak Terindikasi Copy-Move</strong><br>
                        Tidak ditemukan pergeseran blok kembar yang konsisten melebihi ambang batas voting T ({threshold_t}).
                    </div>
                    """, unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("Waktu Total", f"{result.timings['total_ms']:.1f} ms")
                m2.metric("Pasangan Terpilih", f"{len(result.pairs)} blok")
                m3.metric("Vektor Konsisten", f"{len(result.shifts)} vektor")

                tab_v1, tab_v2, tab_v3 = st.tabs(["Overlay Deteksi", "Mask Biner & Boundary", "Citra Masukan"])
                with tab_v1:
                    st.image(result.overlay, caption="Area merah transparan menandai manipulasi copy-move yang terdeteksi.")
                with tab_v2:
                    st.image(result.mask, caption="Mask biner ground truth prediksi (hasil morfologi closing, opening, dan penutupan lubang).")
                with tab_v3:
                    st.image(selected_image, caption=f"Citra asli ({selected_image.width} x {selected_image.height} px)")
            else:
                st.image(selected_image, caption=f"Citra Uji ({selected_image.width} x {selected_image.height} px)")
                st.caption("Pilih metode fitur di kolom kiri, lalu klik tombol **'JALANKAN DETEKSI COPY-MOVE'** untuk melihat hasil analisis.")
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
    * **Tautan Resmi:** [vcl.fer.hr/comofod/](http://vcl.fer.hr/comofod/) | [Kaggle Mirror](https://www.kaggle.com/datasets/sophatvathana/casiav2)
    * **Karakteristik:** Berisi 260 set citra terstandarisasi resolusi 512×512 piksel.
    * **Variasi Gangguan:** Dilengkapi citra yang terkena kompresi JPEG, derau Gaussian, blur, perubahan kontras, dan rotasi.
    * **Ground Truth:** Menyediakan mask biner presisi tingkat piksel untuk evaluasi Precision, Recall, dan F1-Score.

    #### 2. CASIA 2.0 Tampered Dataset (Kaggle)
    * **Tautan:** [kaggle.com/datasets/sophatvathana/casiav2](https://www.kaggle.com/datasets/sophatvathana/casiav2)
    * **Karakteristik:** Berisi ribuan sampel pemalsuan foto realistis (pilih sub-kategori copy-move).

    #### 3. COVERAGE Dataset (Kasus Mirip Objek Asli)
    * **Tautan:** [kaggle.com/datasets/divg23/coverage-dataset](https://www.kaggle.com/datasets/divg23/coverage-dataset)
    * **Karakteristik:** 100 foto dengan objek salinan yang ditempel pada area bertekstur serupa.

    #### 4. USC-SIPI Image Database (Citra Standar Pengolahan Citra Digital)
    * **Tautan:** [sipi.usc.edu/database/](https://sipi.usc.edu/database/)
    * **Karakteristik:** Citra laboratorium standar (*Lena, Cameraman, Peppers, Baboon/Mandrill, House*).

    #### 5. Generator Sintetis Internal CekTimpa (`src/forgery_generator.py`)
    * **Metode:** Memotong patch sumber acak berukuran $48\\times48$ hingga $128\\times128$ piksel, menempelkannya ke lokasi acak lain dengan pergeseran minimal $D_{min}$.
    * **Ground Truth:** Mask biner dibuat otomatis dari koordinat sumber dan target, menjamin perhitungan metrik 100% presisi.
    """)

with tab_benchmark:
    st.subheader("Rencana Matriks Eksperimen Ilmiah (IEEE Format)")
    st.write("Sesuai ketentuan silabus S1, berikut adalah pemetaan eksperimen yang akan menghasilkan data tabel dan grafik analisis:")

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

    * **Yashif Victoriawan**
    * **Fairuz**
    * **Fadhil**
    * **Andre**

    *(Catatan: Rincian pembagian peran dan modul pengerjaan akan disepakati dan diperbarui bersama oleh tim pengembang)*
    """)
