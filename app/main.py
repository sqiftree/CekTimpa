"""
CekTimpa: Web Dashboard Detektor Copy-Move (Pengolahan Citra Digital UNS)
Antarmuka Pengguna Interaktif Berbasis Streamlit dengan Filter Area Polos & Fokus Objek 2D
"""

import sys
import os
import re
import streamlit as st
import numpy as np
from PIL import Image

# Pastikan path modul src/ dapat diimpor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detectors import detect_copymove_block, detect_copymove_sift
from src.forgery_generator import generate_synthetic_forgery
from src.documentation import dataset_reference_markdown
from src.metrics import calculate_pixel_metrics

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
        gt_mask = None
        is_synthetic_demo = (input_source == "Gunakan Citra Sintetis Uji")

        if not is_synthetic_demo:
            uploaded_file = st.file_uploader(
                "Pilih berkas citra (JPG, PNG, atau TIFF):",
                type=["jpg", "jpeg", "png", "tif", "tiff"]
            )
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file).convert("RGB")
                if re.match(r"^\d{2,3}_O(?:_|\.)", uploaded_file.name, re.IGNORECASE):
                    st.info("Nama berkas mengikuti pola citra ASLI CoMoFoD (_O_). "
                            "Jika berkas belum diubah, hasil negatif adalah yang diharapkan. "
                            "Untuk uji positif gunakan pasangan _F_ dan mask _B_.")
            with st.expander("Evaluasi dengan Ground Truth (Opsional)"):
                mask_file = st.file_uploader("Mask sumber + salinan (PNG/TIFF):",
                                             type=["png", "tif", "tiff"], key="gt_mask")
                st.caption("Gunakan mask biner putih untuk sumber dan salinan, hitam untuk latar, "
                           "dengan ukuran dan orientasi yang sama. Gabungkan mask sumber/target jika dataset memisahkannya.")
                if mask_file is not None:
                    gt_mask = np.array(Image.open(mask_file).convert("L"))
                    if not np.isin(gt_mask, [0, 255]).all():
                        st.warning("Mask harus biner 0/255; mask berwarna atau antialias belum dapat dievaluasi.")
                        gt_mask = None
        else:
            sample_choice = st.selectbox(
                "Pilih Sampel Citra Bawaan:",
                options=[
                    "Sampel 1: Citra Sintetis Copy-Move (Duplikasi Objek Bertekstur)",
                    "Sampel 2: Citra Bersih (Tanpa Manipulasi / Uji Negatif)"
                ]
            )
            arr, gt_mask, _ = generate_synthetic_forgery(
                "tampered" if "Copy-Move" in sample_choice else "clean_single")

            selected_image = Image.fromarray(arr)

        st.subheader("2. Metode Deteksi")

        feature_options = {
            "SIFT": "SIFT + FLANN + RANSAC — Metode Utama",
            "DCT_BILATERAL": "DCT–Bilateral: Fitur Detail Tepi",
            "DCT": "Fitur DCT: Koefisien Kosinus Frekuensi Rendah (Tanpa DC)",
            "A": "Fitur A: Rata-rata Sub-blok (Domain Spasial / Box Filter)",
            "B": "Fitur B: Gaussian Lowpass (Domain Spasial / Smoothing)",
            "C": "Fitur C: Magnitudo DFT 2D (Domain Frekuensi Fourier)",
            "D": "Fitur D: DFT Radial (Domain Frekuensi / Uji Rotasi)",
            "H": "Fitur H: Statistik Histogram (Baseline Pembanding)",
            "KP": "Fitur KP: SIFT/ORB + Verifikasi Geometri RANSAC"
        }

        selected_feature = st.radio(
            "Pilih metode deteksi:",
            options=["SIFT", "A", "B", "C", "D", "H", "DCT", "DCT_BILATERAL", "KP"],
            format_func=lambda x: feature_options[x],
            index=0
        )

        feature_desc = {
            "SIFT": "Pencocokan FLANN, verifikasi geometri pada pasangan unik, dan mask yang mengikuti skala titik. Mendukung translasi, rotasi, dan skala seragam; mask menunjukkan dukungan titik, bukan segmentasi objek penuh.",
            "DCT_BILATERAL": "Menggabungkan koefisien DCT dengan detail filter bilateral. Baseline translasi dengan filter tekstur dan pencocokan blok. Nilai Q/epsilon perlu dikalibrasi terpisah dari DCT biasa.",
            "DCT": "DCT ortonormal dengan 16 koefisien zigzag frekuensi rendah tanpa DC. Mengurangi pengaruh perubahan kecerahan aditif; ketahanan noise, rotasi, dan skala perlu diuji melalui benchmark.",
            "A": "Membagi blok menjadi 16 sub-blok (4x4), lalu merata-ratakan intensitasnya. Cepat dan efektif untuk duplikasi objek.",
            "B": "Menyaring citra dengan filter Gaussian lowpass, kemudian mengambil sampel piksel di dalam blok untuk meredam derau halus.",
            "C": "Menghitung transformasi Fourier 2D (FFT) tiap blok, lalu mengambil magnitudo koefisien frekuensi untuk ketahanan geser.",
            "D": "Mengakumulasikan magnitudo DFT pada lingkaran radius frekuensi. Dirancang untuk menguji ketahanan awal terhadap rotasi.",
            "H": "Mengekstrak momen statistik intensitas: nilai rata-rata, variansi, dan kemiringan (skewness) dari histogram blok.",
            "KP": "Mencocokkan SIFT/ORB lalu memverifikasi translasi, rotasi, dan skala seragam dengan RANSAC. Area merah merupakan lingkungan titik yang cocok; objek halus atau salinan kecil masih bisa terlewat."
        }
        st.info(feature_desc[selected_feature])

        # Defaults for parameters not applicable to the selected method.
        block_size, stride_s, q_step = 16, 2, 1.0
        min_std, min_grad_2d, eps = 0., 0., 4.
        match_method, reject_collinear = "kdtree", False
        ratio_threshold, kp_radius, geometry_model = .75, 10, "similarity"
        ransac_error, n_features, contrast_threshold, area_min = 3., 4000, .015, 60
        feature_args = {}
        with st.expander("Pengaturan Metode"):
            p1, p2 = st.columns(2)
            if selected_feature in {"KP", "SIFT"}:
                with p1:
                    if selected_feature == "SIFT":
                        ratio_threshold = st.slider("Rasio Deskriptor SIFT:", .5, .9, .75, .01,
                            help="Nilai lebih rendah lebih selektif; nilai lebih tinggi menambah kandidat dan risiko alarm palsu.")
                        n_features = st.number_input("Batas Titik Kunci:", 1000, 12000, 4000, 1000)
                        contrast_threshold = st.slider("Ambang Kontras SIFT:", .005, .08, .015, .005,
                            help="Nilai lebih kecil menerima lebih banyak titik berkontras rendah; belum tentu meningkatkan akurasi.")
                        kp_radius = st.slider("Radius Dukungan Mask (piksel):", 3, 30, 10,
                            help="Mengubah cakupan mask, bukan menambah bukti kecocokan objek.")
                    else:
                        block_size = st.selectbox("Ukuran Area Mask Keypoint:", [8, 12, 16, 24, 32], index=2)
                        ratio_threshold = st.slider("Rasio Deskriptor KP:", .5, .9, .78, .01)
                        n_features = st.number_input("Batas Titik Kunci KP:", 1000, 12000, 4000, 1000)
                        contrast_threshold = st.slider("Ambang Kontras KP (SIFT):", .005, .08, .015, .005)
                        feature_args.update(ratio_threshold=ratio_threshold, n_features=int(n_features),
                                            contrast_threshold=contrast_threshold)
                with p2:
                    d_min = st.slider("Jarak Spasial Minimum (D_min):", 10, 100, 30)
                    threshold_t = st.slider("Minimum Pasangan Inlier RANSAC:", 4, 100, 6)
                    ransac_error = st.slider("Toleransi Reproyeksi RANSAC (piksel):", 1., 8., 3., .5)
                    if selected_feature == "SIFT":
                        geometry_model = st.selectbox("Model Geometri:", ["similarity", "homography"],
                            help="Similarity untuk translasi/rotasi/skala. Homografi planar bersifat eksperimental dan lebih mudah menerima kecocokan kebetulan.")
                st.caption("Mulai dengan similarity dan parameter default. Naikkan selektivitas bila citra asli berpola berulang ikut ditandai. "
                           "Menambah radius mask tidak memperbaiki recall pasangan.")
            else:
                feature_args['feature_scaling'] = st.selectbox("Skala Fitur:", ["native", "standardized"],
                    index=1 if selected_feature == "H" else 0,
                    help="Standardized memakai z-score per citra dibagi √dimensi. Q ditetapkan 1; epsilon mengukur jarak RMS terstandardisasi dan perlu dituning ulang.")
                standardized = feature_args['feature_scaling'] == "standardized"
                with p1:
                    block_size = st.selectbox("Ukuran Blok (B x B):", [8, 12, 16, 24, 32],
                                             index=0 if not is_synthetic_demo or selected_feature == "DCT_BILATERAL" else 2)
                    stride_s = st.slider("Langkah Stride (s):", 1, 8, 2)
                    if selected_feature == "A":
                        grids = [g for g in [2, 4, 8] if block_size % g == 0]
                        feature_args['g_sub'] = st.selectbox("Grid Sub-blok (G × G):", grids, index=grids.index(4))
                    elif selected_feature == "B":
                        feature_args['gaussian_sigma'] = st.slider("Sigma Gaussian Fitur B:", .2, 3., 1., .1)
                        feature_args['g_samples'] = st.selectbox("Sampel per Sisi Fitur B:", [2, 4, 8], index=1)
                        feature_args['gaussian_context'] = st.selectbox("Konteks Gaussian:", ["block", "image"],
                            help="Block menghaluskan tiap blok secara mandiri. Image mempertahankan baseline lama yang dipengaruhi konteks di luar patch.")
                    elif selected_feature == "C":
                        feature_args['low_freq_l'] = st.slider("Koefisien FFT per Sisi (L):", 2, block_size, 4)
                        feature_args['fft_normalize'] = st.checkbox("Bagi Magnitudo FFT dengan B²", value=True,
                            help="Mengurangi pengaruh ukuran blok pada skala koefisien; bukan jaminan tahan rotasi.")
                    elif selected_feature == "D":
                        feature_args['radial_bins'] = st.slider("Jumlah Cincin Frekuensi:", 1, block_size // 2, block_size // 2)
                        feature_args['fft_normalize'] = st.checkbox("Bagi Magnitudo FFT dengan B²", value=True)
                        st.caption("Perataan radial mengurangi informasi arah, tetapi voting translasi tetap membatasi deteksi rotasi.")
                    elif selected_feature == "DCT":
                        feature_args['n_coeffs'] = st.slider("Jumlah Koefisien DCT:", 4, min(64, block_size ** 2 - 1), 16)
                        feature_args['include_dc'] = st.checkbox("Sertakan Koefisien DC", value=False,
                            help="DC membawa rata-rata kecerahan; menghapusnya meningkatkan toleransi offset tetapi mengurangi pembeda area.")
                    elif selected_feature == "DCT_BILATERAL":
                        feature_args['p_ratio'] = st.slider("Proporsi Koefisien DCT–Bilateral:", .1, 1., .25, .05)
                        feature_args['bilateral_d'] = st.selectbox("Diameter Filter Bilateral:", [3, 5, 7], index=0)
                        feature_args['bilateral_sigma_s'] = st.slider("Sigma Spasial Bilateral:", .5, 5., 1., .5)
                        feature_args['bilateral_sigma_r'] = st.slider("Sigma Intensitas Bilateral:", 5., 75., 25.5, .5)
                    elif selected_feature == "H":
                        st.caption("Mean, variance, dan skewness memiliki satuan berbeda. Standardisasi menyeimbangkan pengaruhnya; histogram tetap lemah membedakan susunan spasial.")
                    q_step = st.number_input("Kuantisasi Fitur (Q):", .1, 32.,
                                             16. if selected_feature == "DCT_BILATERAL" and not standardized else 1., .1,
                                             disabled=standardized)
                    min_std = st.slider("Filter Area Polos (Min Std):", 0., 40.,
                                        0. if is_synthetic_demo else 10., 1.)
                    min_grad_2d = st.slider("Filter Gradien 2D:", 0., 20.,
                                            0. if is_synthetic_demo else 3., .5)
                with p2:
                    d_min = st.slider("Jarak Spasial Minimum (D_min):", 10, 100, 30)
                    threshold_t = st.slider("Ambang Voting Vektor (T):", 3, 200, 25 if is_synthetic_demo else 15)
                    eps = st.number_input("Toleransi Jarak Fitur (Epsilon):", .01, 25.,
                                          .5 if standardized else (5. if selected_feature == "DCT_BILATERAL" else 4.), .1)
                    match_method = st.selectbox("Metode Pencocokan Fitur:", ["kdtree", "lexsort"])
                    feature_args['window'] = st.slider("Jumlah Tetangga Kandidat:", 2, 40, 10)
                    feature_args['shift_tol'] = st.slider("Lebar Bin Vektor Pergeseran (piksel):", 1, 8, 2)
                    reject_collinear = st.checkbox("Tolak Klaster Garis Lurus 1D", value=not is_synthetic_demo)
            area_min = st.slider("Luas Minimum Komponen Mask (piksel):", 20, 2000, 60, 20)


        btn_run = st.button("JALANKAN DETEKSI COPY-MOVE", use_container_width=True)

    with col_display:
        st.subheader("3. Visualisasi Citra & Area Deteksi")

        if selected_image is not None:
            image_np = np.array(selected_image)

            if btn_run:
                config = {
                    "feature": selected_feature,
                    "ransac_error": ransac_error,
                    "block_size": int(block_size),
                    "stride": int(stride_s),
                    "d_min": float(d_min),
                    "threshold_t": int(threshold_t),
                    "q_step": float(q_step),
                    "eps": float(eps),
                    "area_min": area_min,
                    "prefilter": "none",
                    "min_std": float(min_std),
                    "min_grad_2d": float(min_grad_2d),
                    "match_method": match_method,
                    "shift_tol": 2,
                    "reject_collinear": bool(reject_collinear),
                    **feature_args
                }

                with st.spinner("Memproses fitur, pasangan kandidat, dan verifikasi kecocokan..."):
                    if selected_feature == "SIFT":
                        result = detect_copymove_sift(image_np, {
                            "min_inliers": threshold_t, "d_min": d_min,
                            "ransac_threshold": ransac_error, "area_min": area_min,
                            "ratio_threshold": ratio_threshold, "kp_radius": kp_radius,
                            "geometry_model": geometry_model,
                            "n_features": int(n_features), "contrast_threshold": contrast_threshold})
                    else:
                        result = detect_copymove_block(image_np, config)

                group_name = "model geometri" if selected_feature in {"KP", "SIFT"} else "pola translasi"
                pair_name = "pasangan keypoint" if selected_feature in {"KP", "SIFT"} else "pasangan blok"
                no_match_reason = (
                    f"Tidak ditemukan model dengan minimal {threshold_t} pasangan inlier unik dan mask yang lolos penapisan."
                    if selected_feature in {"KP", "SIFT"} else
                    f"Tidak ditemukan pasangan blok dengan dukungan voting minimal {threshold_t} dan mask yang lolos penapisan."
                )
                if result.is_forged:
                    st.markdown(f"""
                    <div class="result-box result-alert">
                        <strong style="color: #DC2626;">Status: Terindikasi Copy-Move</strong><br>
                        Ditemukan <strong>{len(result.shifts)} {group_name}</strong> dengan <strong>{result.total_matches} {pair_name} cocok</strong>.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-box result-safe">
                        <strong style="color: #059669;">Status: Tidak Terindikasi Copy-Move</strong><br>
                        {no_match_reason}<br>Hasil ini bukan jaminan bahwa citra asli.
                    </div>
                    """, unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("Waktu Total", f"{result.timings['total_ms']:.1f} ms")
                m2.metric("Pasangan Terpilih", f"{len(result.pairs)} pasangan")
                m3.metric("Model Geometri" if selected_feature in {"KP", "SIFT"} else "Vektor Konsisten", len(result.shifts))

                if selected_feature in {"KP", "SIFT"}:
                    d = (dict(keypoints=result.n_keypoints, candidate_pairs=result.n_spatial,
                              verified_pairs=result.n_inliers)
                         if selected_feature == "SIFT" else result.diagnostics)
                    st.caption(
                        f"Titik kunci: {d.get('keypoints', 0)} → "
                        f"kandidat unik setelah ratio test: {d.get('candidate_pairs', 0)} → "
                        f"pasangan lolos geometri: {d.get('verified_pairs', 0)}. "
                        "Kecocokan geometris belum membuktikan pemalsuan pada objek yang memang mirip."
                    )

                if selected_feature not in {"SIFT", "KP"}:
                    st.caption(f"Blok lolos filter: {result.metadata.get('n_blocks', 0)} · "
                               f"Dimensi fitur: {result.metadata.get('feature_dimensions', 0)} · "
                               f"Skala: {feature_args.get('feature_scaling', 'native')}. "
                               "Voting blok memverifikasi translasi; kemiripan fitur belum membuktikan kesamaan objek.")
                if selected_feature == "SIFT":
                    with st.expander("Periksa Bukti Kecocokan"):
                        st.image(result.step1_keypoints_img, caption="Titik kunci dan skala SIFT")
                        st.image(result.step3_spatial_img, caption="Pasangan kandidat setelah filter jarak dan rasio")
                        st.image(result.step4_ransac_img, caption="Pasangan yang lolos verifikasi geometri")
                        st.write(result.transform_type)
                        if result.homography is not None:
                            st.write(result.homography)

                tab_v1, tab_v2, tab_v3 = st.tabs(["Overlay Deteksi", "Mask Biner & Boundary", "Citra Masukan"])
                with tab_v1:
                    st.image(result.overlay, caption="Area merah menunjukkan dukungan kecocokan yang terdeteksi; batasnya masih merupakan perkiraan.")
                with tab_v2:
                    st.image(result.mask, caption="Mask prediksi setelah morfologi; bukan ground truth.")
                with tab_v3:
                    st.image(selected_image, caption=f"Citra asli ({selected_image.width} x {selected_image.height} px)")
                if gt_mask is not None:
                    if gt_mask.shape != result.mask.shape:
                        st.warning("Ukuran ground truth berbeda dari citra. Evaluasi dilewati; mask tidak diubah ukurannya otomatis.")
                    else:
                        scores = calculate_pixel_metrics(result.mask, gt_mask)
                        st.subheader("Evaluasi Mask pada Citra Ini")
                        st.write({name: round(scores[name], 4) for name in ("precision", "recall", "f1", "iou")})
                        st.caption(f"Piksel area GT terlewat (FN): {scores['fn']} · "
                                   f"Piksel tambahan di luar GT (FP): {scores['fp']}. "
                                   "Ini metrik satu citra, bukan akurasi seluruh dataset.")
            else:
                st.image(selected_image, caption=f"Citra Uji ({selected_image.width} x {selected_image.height} px)")
                st.caption("Pilih metode fitur di kolom kiri, lalu klik tombol **'JALANKAN DETEKSI COPY-MOVE'** untuk melihat hasil analisis.")
        else:
            st.info("Pilih citra masukan di kolom kiri terlebih dahulu.")

with tab_dataset:
    st.subheader("Dataset Acuan Benchmark Copy-Move Forgery")
    try:
        st.markdown(dataset_reference_markdown())
    except (OSError, ValueError) as error:
        st.error(f"Referensi dataset belum dapat dibaca: {error}")

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
