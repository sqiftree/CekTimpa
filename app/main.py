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
from src.detectors.presets import image_adaptive_preset
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
            "A": "Fitur A: Rata-rata Sub-blok (Domain Spasial / Box Filter)",
            "B": "Fitur B: Gaussian Lowpass (Domain Spasial / Smoothing)",
            "C": "Fitur C: SIFT + FLANN + RANSAC (Domain Spasial / Keypoint)",
            "D": "Fitur D: DCT ortonormal dengan koefisien zigzag frekuensi rendah tanpa DC",
        }

        selected_feature = st.radio(
            "Pilih metode deteksi:",
            options=["A", "B", "C", "D"],
            format_func=lambda x: feature_options[x],
            index=0
        )

        settings_mode = st.selectbox(
            "Preset Pengaturan:",
            ["Adaptif per citra", "Manual"],
            help="Preset adaptif memilih nilai awal dari ukuran dan tekstur citra. Semua parameter tetap dapat disesuaikan."
        )
        image_profile = (image_adaptive_preset(np.array(selected_image), selected_feature)
                         if selected_image is not None else None)
        if settings_mode == "Adaptif per citra" and image_profile:
            st.caption(
                f"Preset adaptif: sisi pendek {image_profile['short_side']} px · "
                f"tekstur {image_profile['texture']:.1f} · nilai awal menyesuaikan citra."
            )
        image_token = ("no-image" if image_profile is None else
                       "_".join(str(value) for name, value in sorted(image_profile.items())
                                if name not in {"texture", "short_side"}))

        def setting_key(name):
            return f"setting_{selected_feature}_{settings_mode}_{image_token}_{name}"

        feature_desc = {
            "A": "Membagi blok menjadi 16 sub-blok (4x4), lalu merata-ratakan intensitasnya. Cepat dan efektif untuk duplikasi objek.",
            "B": "Menyaring citra dengan filter Gaussian lowpass, kemudian mengambil sampel piksel di dalam blok untuk meredam derau halus.",
            "C": "Pencocokan FLANN dan verifikasi geometri RANSAC. Mendukung translasi, rotasi, dan skala seragam; mask menunjukkan dukungan titik, bukan segmentasi objek penuh.",
            "D": "DCT ortonormal dengan koefisien zigzag frekuensi rendah tanpa DC. Mengurangi pengaruh offset kecerahan; tidak memverifikasi rotasi atau skala.",
        }
        st.info(feature_desc[selected_feature])

        # Defaults for parameters not applicable to the selected method.
        use_adaptive = settings_mode == "Adaptif per citra" and image_profile is not None
        block_size = image_profile["block_size"] if use_adaptive else 16
        stride_s = image_profile["stride"] if use_adaptive else 2
        q_step = 1.0
        min_std = image_profile["min_std"] if use_adaptive else 0.
        min_grad_2d = image_profile["min_grad_2d"] if use_adaptive else 0.
        eps = 4.
        match_method, reject_collinear = "kdtree", False
        ratio_threshold, kp_radius, geometry_model = .75, 10, "similarity"
        ransac_error = image_profile["ransac_error"] if use_adaptive else 3.
        n_features = image_profile["n_features"] if use_adaptive else 4000
        contrast_threshold = image_profile["contrast_threshold"] if use_adaptive else .015
        area_min = image_profile["area_min"] if use_adaptive else 60
        feature_args = {}
        with st.expander("Pengaturan Metode"):
            p1, p2 = st.columns(2)
            if selected_feature == "SIFT":
                with p1:
                    ratio_threshold = st.slider("Rasio Deskriptor SIFT:", .5, .9, .75, .01,
                        help="Nilai lebih rendah lebih selektif; nilai lebih tinggi menambah kandidat dan risiko alarm palsu.",
                        key=setting_key("ratio_threshold"))
                    n_features = st.number_input("Batas Titik Kunci:", 1000, 12000, n_features, 1000,
                        key=setting_key("n_features"))
                    contrast_threshold = st.slider("Ambang Kontras SIFT:", .005, .08, contrast_threshold, .005,
                        help="Nilai lebih kecil menerima lebih banyak titik berkontras rendah; belum tentu meningkatkan akurasi.",
                        key=setting_key("contrast_threshold"))
                    kp_radius = st.slider("Radius Dukungan Mask (piksel):", 3, 30, 10,
                        help="Mengubah cakupan mask, bukan menambah bukti kecocokan objek.",
                        key=setting_key("kp_radius"))
                with p2:
                    d_min = st.slider("Jarak Spasial Minimum (D_min):", 10, 100,
                        image_profile["d_min"] if use_adaptive else 30, key=setting_key("d_min"))
                    threshold_t = st.slider("Minimum Pasangan Inlier RANSAC:", 4, 100,
                        image_profile["threshold_t"] if use_adaptive else 6,
                        key=setting_key("threshold_t"))
                    ransac_error = st.slider("Toleransi Reproyeksi RANSAC (piksel):", 1., 8.,
                        ransac_error, .5, key=setting_key("ransac_error"))
                    geometry_model = st.selectbox("Model Geometri:", ["similarity", "homography"],
                        help="Similarity untuk translasi/rotasi/skala. Homografi planar bersifat eksperimental dan lebih mudah menerima kecocokan kebetulan.",
                        key=setting_key("geometry_model"))
                st.caption("Mulai dengan similarity dan parameter default. Naikkan selektivitas bila citra asli berpola berulang ikut ditandai. "
                           "Menambah radius mask tidak memperbaiki recall pasangan.")
            else:
                feature_args['feature_scaling'] = st.selectbox("Skala Fitur:", ["native", "standardized"],
                    index=0,
                    help="Standardized memakai z-score per citra dibagi √dimensi. Q ditetapkan 1; epsilon mengukur jarak RMS terstandardisasi dan perlu dituning ulang.",
                    key=setting_key("feature_scaling"))
                standardized = feature_args['feature_scaling'] == "standardized"
                with p1:
                    block_options = [8, 12, 16, 24, 32]
                    block_size = st.selectbox("Ukuran Blok (B x B):", block_options,
                        index=block_options.index(block_size) if use_adaptive else
                              (0 if not is_synthetic_demo else 2),
                        key=setting_key("block_size"))
                    stride_s = st.slider("Langkah Stride (s):", 1, 8, stride_s,
                        key=setting_key("stride"))
                    if selected_feature == "A":
                        grids = [g for g in [2, 4, 8] if block_size % g == 0]
                        feature_args['g_sub'] = st.selectbox("Grid Sub-blok (G × G):", grids, index=grids.index(4))
                    elif selected_feature == "B":
                        feature_args['gaussian_sigma'] = st.slider("Sigma Gaussian Fitur B:", .2, 3., 1., .1)
                        feature_args['g_samples'] = st.selectbox("Sampel per Sisi Fitur B:", [2, 4, 8], index=1)
                        feature_args['gaussian_context'] = st.selectbox("Konteks Gaussian:", ["block", "image"],
                            help="Block menghaluskan tiap blok secara mandiri. Image mempertahankan baseline lama yang dipengaruhi konteks di luar patch.")
                    elif selected_feature == "DCT":
                        feature_args['n_coeffs'] = st.slider("Jumlah Koefisien DCT:", 4, min(64, block_size ** 2 - 1), 16)
                        feature_args['include_dc'] = st.checkbox("Sertakan Koefisien DC", value=False,
                            help="DC membawa rata-rata kecerahan; menghapusnya meningkatkan toleransi offset tetapi mengurangi pembeda area.")
                    q_step = st.number_input("Kuantisasi Fitur (Q):", .1, 32.,
                                             1., .1,
                                             disabled=standardized, key=setting_key("q_step"))
                    min_std = st.slider("Filter Area Polos (Min Std):", 0., 40.,
                                        min_std if use_adaptive else (0. if is_synthetic_demo else 10.), 1.,
                                        key=setting_key("min_std"))
                    min_grad_2d = st.slider("Filter Gradien 2D:", 0., 20.,
                                            min_grad_2d if use_adaptive else (0. if is_synthetic_demo else 3.), .5,
                                            key=setting_key("min_grad_2d"))
                with p2:
                    d_min = st.slider("Jarak Spasial Minimum (D_min):", 10, 100,
                        image_profile["d_min"] if use_adaptive else 30, key=setting_key("d_min"))
                    threshold_t = st.slider("Ambang Voting Vektor (T):", 3, 200,
                        image_profile["threshold_t"] if use_adaptive else (25 if is_synthetic_demo else 15),
                        key=setting_key("threshold_t"))
                    eps = st.number_input("Toleransi Jarak Fitur (Epsilon):", .01, 25.,
                                          .5 if standardized else 4., .1,
                                          key=setting_key("eps"))
                    match_method = st.selectbox("Metode Pencocokan Fitur:", ["kdtree", "lexsort"],
                        key=setting_key("match_method"))
                    feature_args['window'] = st.slider("Jumlah Tetangga Kandidat:", 2, 40, 10,
                        key=setting_key("window"))
                    feature_args['shift_tol'] = st.slider("Lebar Bin Vektor Pergeseran (piksel):", 1, 8, 2,
                        key=setting_key("shift_tol"))
                    reject_collinear = st.checkbox("Tolak Klaster Garis Lurus 1D", value=not is_synthetic_demo,
                        key=setting_key("reject_collinear"))
            area_min = st.slider("Luas Minimum Komponen Mask (piksel):", 20, 2000,
                image_profile["area_min"] if use_adaptive else 60, 20, key=setting_key("area_min"))


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

                group_name = "model geometri" if selected_feature == "SIFT" else "pola translasi"
                pair_name = "pasangan keypoint" if selected_feature == "SIFT" else "pasangan blok"
                no_match_reason = (
                    f"Tidak ditemukan model dengan minimal {threshold_t} pasangan inlier unik dan mask yang lolos penapisan."
                    if selected_feature == "SIFT" else
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
                m3.metric("Model Geometri" if selected_feature == "SIFT" else "Vektor Konsisten", len(result.shifts))

                if selected_feature == "SIFT":
                    d = dict(keypoints=result.n_keypoints, candidate_pairs=result.n_spatial,
                             verified_pairs=result.n_inliers)
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
