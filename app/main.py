"""
CekTimpa: Web Dashboard Detektor Copy-Move (Pengolahan Citra Digital UNS)
Antarmuka Pengguna Interaktif Berbasis Streamlit
"""

import streamlit as st
import numpy as np
from PIL import Image

# Konfigurasi Halaman
st.set_page_config(
    page_title="CekTimpa - Detektor Copy-Move",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Kustomisasi CSS: Tampilan bersih, kontras jelas, non-slop
st.markdown("""
<style>
    /* Styling tombol utama agar menonjol */
    div.stButton > button:first-child {
        background-color: #2563EB;
        color: white;
        font-weight: 600;
        font-size: 16px;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        border: none;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background-color: #1D4ED8;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    
    /* Box status keputusan yang jelas */
    .status-card {
        padding: 1rem 1.25rem;
        border-radius: 8px;
        margin-top: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid;
    }
    .status-neutral {
        background-color: rgba(59, 130, 246, 0.08);
        border-color: #3B82F6;
    }
    .status-warning {
        background-color: rgba(239, 68, 68, 0.08);
        border-color: #EF4444;
    }
    
    /* Panel parameter ringkas */
    .param-box {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header Aplikasi
col_title, col_meta = st.columns([3, 1])
with col_title:
    st.title("CekTimpa: Detektor Copy-Move")
    st.caption("Deteksi manipulasi salin-tempel pada satu foto yang sama menggunakan matematika murni Pengolahan Citra Digital (PCD).")
with col_meta:
    st.markdown("""
    <div style="text-align: right; padding-top: 10px;">
        <span style="background: #1E293B; color: #94A3B8; padding: 4px 10px; border-radius: 6px; font-size: 13px; font-weight: 500;">
            Informatika UNS • Non-Deep Learning
        </span>
    </div>
    """, unsafe_allow_html=True)

# Catatan Batasan Ilmiah (Ringkas & Tegas)
st.info(
    "**Catatan Batasan Sistem:** CekTimpa khusus mendeteksi pemalsuan **Copy-Move** (duplikasi area di foto yang sama). "
    "Sistem tidak mendeteksi objek dari foto lain (*splicing*), teks dokumen, maupun rekayasa AI. "
    "Hasil bersifat indikasi teknis, bukan sertifikasi keaslian mutlak.",
    icon="ℹ️"
)

# Tab Navigasi Utama
tab_deteksi, tab_benchmark, tab_tim = st.tabs(["Uji Deteksi Citra", "Hasil Benchmark Riset", "Tim Pengembang"])

with tab_deteksi:
    st.subheader("1. Pilih Citra Uji")
    uploaded_file = st.file_uploader(
        "Pilih file foto yang dicurigai:",
        type=["jpg", "jpeg", "png"],
        help="Unggah foto berformat JPG atau PNG"
    )

    st.subheader("2. Pilih Metode Ekstraksi Fitur (Bahan Komparasi)")
    
    # Tombol pemilihan fitur horizontal yang sangat jelas
    feature_labels = {
        "A": "Fitur A: Rata-rata Sub-blok (Domain Spasial / Box Filter)",
        "B": "Fitur B: Gaussian Lowpass (Domain Spasial / Smoothing)",
        "C": "Fitur C: Magnitudo DFT (Domain Frekuensi 2D Fourier)",
        "D": "Fitur D: DFT Radial (Domain Frekuensi / Uji Rotasi)",
        "H": "Fitur H: Statistik Histogram (Baseline Pembanding)"
    }
    
    selected_feature = st.radio(
        "Metode representasi fitur blok:",
        options=["A", "B", "C", "D", "H"],
        format_func=lambda x: feature_labels[x],
        horizontal=False,
        index=0
    )

    # Penjelasan singkat fitur yang dipilih
    feature_explanations = {
        "A": "Membagi blok 16×16 menjadi 16 sub-blok berukuran 4×4, lalu mengambil nilai intensitas rata-rata tiap sub-blok. Cepat dan sederhana.",
        "B": "Menyaring citra dengan kernel Gaussian lowpass, lalu mengambil sampel teratur di dalam blok untuk meredam noise frekuensi tinggi.",
        "C": "Mentransformasikan blok ke spektrum Fourier 2D dan mengambil magnitudo frekuensi rendah untuk menangkap struktur dominan.",
        "D": "Mengintegrasikan magnitudo DFT pada lingkaran radius frekuensi. Dirancang untuk menguji ketahanan terhadap rotasi patch.",
        "H": "Baseline sederhana: mengekstrak nilai mean, variansi, dan skewness dari histogram intensitas blok."
    }
    st.caption(f"💡 **Prinsip Kerja:** {feature_explanations[selected_feature]}")

    # Parameter Lanjutan (Bisa dilipat agar tidak memenuhi layar)
    with st.expander("⚙️ Pengaturan Parameter Lanjutan (Tuning & Ruang Warna)"):
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            color_mode = st.selectbox("Ruang Warna Citra:", ["Grayscale (Intensitas)", "RGB (Per-kanal)", "HSI"], index=0)
            block_size = st.select_slider("Ukuran Blok (B × B):", options=[8, 16, 24, 32], value=16)
        with p_col2:
            stride = st.slider("Langkah Stride (s):", min_value=1, max_value=8, value=2, help="Langkah pergeseran blok. Stride 2 lebih cepat dibanding stride 1.")
            d_min = st.slider("Jarak Spasial Minimum (D_min):", min_value=10, max_value=100, value=30, help="Mencegah blok yang bertetangga dekat saling cocok sendiri.")
        with p_col3:
            threshold_t = st.slider("Ambang Voting Vektor (T):", min_value=10, max_value=200, value=50, help="Jumlah minimum pasangan blok dengan translasi pergeseran yang sama.")
            prefilter = st.selectbox("Pra-penapisan Restorasi:", ["Tidak ada", "Median Filter (Hapus Salt & Pepper)", "Gaussian Filter"], index=0)

    # Tombol Eksekusi Utama (Besar & Jelas)
    btn_detect = st.button("Jalankan Deteksi Copy-Move", type="primary", use_container_width=True)

    st.markdown("---")
    
    # Area Hasil Analisis
    st.subheader("3. Hasil & Analisis Visual")
    
    col_input, col_result = st.columns(2)
    
    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file).convert("RGB")
        with col_input:
            st.markdown("**Citra Masukan:**")
            st.image(raw_image, use_container_width=True)
            st.caption(f"Resolusi: {raw_image.width} × {raw_image.height} piksel")

        with col_result:
            st.markdown("**Area Deteksi & Garis Tepi (Boundary):**")
            if btn_detect:
                # Placeholder simulasi status deteksi
                st.markdown("""
                <div class="status-card status-neutral">
                    <h4 style="margin: 0; color: #1E3A8A;">Status Analisis Selesai</h4>
                    <p style="margin: 5px 0 0 0; font-size: 14px;">
                        Algoritma detektor inti di <code>src/detectors/copymove_block.py</code> sedang disiapkan untuk integrasi.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Mockup metrik eksekusi
                m1, m2, m3 = st.columns(3)
                m1.metric("Waktu Ekstraksi", "0.00 ms")
                m2.metric("Pasangan Cocok", "0 pasang")
                m3.metric("Vektor Lolos Ambang", "0 vektor")
                
                st.info("Setelah modul algoritma diimplementasikan, area hasil salin-tempel dan garis tepi (boundary extraction) akan dirender langsung di atas citra ini.")
            else:
                st.info("Pilih metode fitur di atas, lalu klik tombol **'Jalankan Deteksi Copy-Move'** untuk melihat hasil analisis.")
    else:
        with col_input:
            st.info("Silakan unggah foto di langkah 1 terlebih dahulu.")
        with col_result:
            st.text("Menunggu unggahan citra...")

with tab_benchmark:
    st.subheader("Hasil Eksperimen & Analisis Ilmiah (Jurnal ITSMART)")
    st.write(
        "Halaman ini digunakan untuk menampilkan data kuantitatif dari skrip eksperimen di folder `experiments/results/` "
        "seperti perbandingan F1-Score antar fitur dan kurva ketahanan terhadap derau Gaussian, blur, dan JPEG."
    )
    st.table({
        "ID Eksperimen": ["E1", "E2", "E3", "E5", "E6"],
        "Fokus Pengujian": [
            "Komparasi Fitur Blok Tanpa Gangguan (H, A, B, C, D)",
            "Sensitivitas Parameter Ukuran Blok (B) & Stride (s)",
            "Ketahanan terhadap Derau Gaussian, Blur, dan Kompresi JPEG",
            "Uji Kelemahan Transformasi Geometrik (Rotasi & Skala)",
            "Perbandingan Ruang Warna (Grayscale vs RGB vs HSI)"
        ],
        "Metrik Utama": ["Precision, Recall, F1", "F1 vs Waktu Komputasi", "F1 vs PSNR (dB)", "F1 pada sudut rotasi 5°-30°", "F1 per model warna"]
    })

with tab_tim:
    st.subheader("Tim Pengembang CekTimpa")
    st.markdown("""
    **Departemen Informatika, Universitas Sebelas Maret (UNS) Surakarta**  
    *Final Project: Pengolahan Citra Digital (PCD)*

    * **Yashif Victoriawan** — Frontend Streamlit, Integrasi UI & Visualisasi
    * **Fairuz** — Algoritma Inti Ekstraksi Fitur & Matriks Blok
    * **Fadhil** — Pencocokan Leksikografis & Voting Vektor Pergeseran
    * **Andre** — Generator Pemalsuan Sintetis, Dataset & Eksperimen Benchmark
    """)
