"""
Aplikasi Antarmuka Interaktif CekTimpa
Framework: Streamlit
"""

import streamlit as st
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="CekTimpa: Detektor Copy-Move",
    page_icon="🔍",
    layout="wide"
)

# Header Utama
st.title("🔍 CekTimpa: Detektor COPY-MOVE")
st.caption("Deteksi Pemalsuan Citra: Foto yang ditimpa potongan dari foto itu sendiri (Murni Konvensional PCD)")

# Panel Wajib: Cakupan & Batasan Ilmiah
with st.expander("ℹ️ Cakupan & Batasan Sistem (Penting Dibaca)", expanded=True):
    st.markdown("""
    * **Cakupan Deteksi:** Sistem ini khusus mendeteksi manipulasi **Copy-Move** (area yang disalin lalu ditempel ke bagian lain pada **foto yang sama**).
    * **Di Luar Cakupan:** Sistem **tidak mendeteksi** objek dari foto lain (*splicing*), manipulasi teks struk/dokumen, hasil *inpainting* / generasi AI, maupun efek filter retouch.
    * **Prinsip Keputusan:** Hasil berupa **'Terindikasi copy-move'** atau **'Tidak terindikasi copy-move'**. Status *tidak terindikasi* bukan sertifikasi bahwa foto 100% asli.
    """)

# Sidebar Parameter Konfigurasi
st.sidebar.header("⚙️ Parameter Algoritma")
feature_type = st.sidebar.selectbox("Pilih Ekstraksi Fitur:", ["A (Rata-rata Sub-blok)", "B (Gaussian Lowpass)", "C (Magnitudo DFT)", "D (DFT Radial)", "H (Statistik Histogram)"])
color_mode = st.sidebar.selectbox("Model Warna:", ["Grayscale (Intensitas)", "RGB (Per-kanal)", "HSI"])
block_size = st.sidebar.slider("Ukuran Blok (B x B):", min_value=8, max_value=32, value=16, step=4)
stride = st.sidebar.slider("Langkah Stride (s):", min_value=1, max_value=8, value=2, step=1)
threshold_t = st.sidebar.slider("Ambang Voting Vektor (T):", min_value=10, max_value=200, value=50, step=10)

# Area Utama: Upload & Tab
tab_deteksi, tab_benchmark, tab_tentang = st.tabs(["🖼️ Uji Deteksi Citra", "📊 Benchmark & Hasil Riset", "👥 Tentang Pengembang"])

with tab_deteksi:
    uploaded_file = st.file_uploader("Unggah citra yang dicurigai (JPG / PNG):", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Citra Masukan")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("Hasil Deteksi & Mask")
            st.info("Algoritma detektor sedang dalam tahap implementasi oleh tim pengembang.")
            
with tab_benchmark:
    st.subheader("Evaluasi Kinerja Model (Hasil Eksperimen CSV)")
    st.write("Tabel metrik kuantitatif (Precision, Recall, F1-Score) dan grafik ketahanan terhadap gangguan (noise, blur, JPEG) akan ditampilkan di sini.")

with tab_tentang:
    st.subheader("Tim Pengembang CekTimpa (Informatika UNS)")
    st.markdown("""
    * **Yashif Victoriawan** — Pengembang Utama & Frontend
    * **Fairuz** — Anggota Tim Pengembang
    * **Fadhil** — Anggota Tim Pengembang
    * **Andre** — Anggota Tim Pengembang
    """)
