"""
CekTimpa: Web Dashboard Detektor Copy-Move
Metode: SIFT + FLANN + RANSAC (Amerini dkk., TIFS IEEE 2011)
"""

import sys
import os
import streamlit as st
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.detectors import (
    detect_copymove_sift,
    detect_copymove_block,
    SIFTDetectionResult,
    BlockDetectionResult,
)
from src.forgery_generator import generate_synthetic_forgery
from src.metrics import calculate_pixel_metrics

# ─────────────────────────────────────────────────────────────────────────────
# Konfigurasi Halaman
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CekTimpa – SIFT Copy-Move Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
code, pre { font-family: 'JetBrains Mono', monospace !important; }

/* ── Tombol Utama ── */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #1D4ED8 0%, #7C3AED 100%);
    color: #fff;
    font-weight: 700;
    font-size: 15px;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    border: none;
    width: 100%;
    margin-top: 0.5rem;
    letter-spacing: 0.5px;
    box-shadow: 0 4px 15px rgba(124,58,237,0.35);
    transition: all .2s;
}
div.stButton > button:first-child:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(124,58,237,0.5);
}

/* ── Status Box ── */
.result-box { padding:1rem 1.25rem; border-radius:8px; margin-bottom:1rem; border-left:4px solid; }
.result-alert { border-color:#EF4444; background:rgba(239,68,68,.07); }
.result-safe  { border-color:#10B981; background:rgba(16,185,129,.07); }

/* ── Pipeline Step Card ── */
.step-card {
    background: rgba(30,41,59,.6);
    border: 1px solid rgba(100,116,139,.3);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 8px;
    transition: border-color .2s;
}
.step-card:hover { border-color: rgba(124,58,237,.6); }
.step-num {
    display:inline-block;
    background: linear-gradient(135deg,#1D4ED8,#7C3AED);
    color:#fff;
    font-weight:700;
    font-size:11px;
    width:22px; height:22px;
    border-radius:50%;
    text-align:center;
    line-height:22px;
    margin-right:8px;
    vertical-align:middle;
}
.step-title { font-weight:600; font-size:14px; color:#E2E8F0; display:inline; }
.step-formula {
    font-family:'JetBrains Mono',monospace;
    font-size:12px;
    color:#A5B4FC;
    background:rgba(99,102,241,.12);
    padding:6px 10px;
    border-radius:6px;
    margin:6px 0 4px 30px;
    border-left:2px solid #6366F1;
    display:block;
}
.step-desc { font-size:12px; color:#94A3B8; margin-left:30px; line-height:1.5; }
.step-val  { font-size:12px; color:#34D399; margin-left:30px; font-weight:600; margin-top:4px; }

.arrow-down { text-align:center; color:#475569; font-size:18px; margin:2px 0; }

/* ── Legend dot ── */
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; vertical-align:middle; }
.dot-red   { background:#FF3C00; }
.dot-green { background:#00DC5A; }
.dot-cyan  { background:#00C8FF; }
.dot-lime  { background:#32FF64; }

/* ── Metric box ── */
.metric-row { display:flex; gap:12px; margin:12px 0; }
.metric-box {
    flex:1;
    background:rgba(30,41,59,.6);
    border:1px solid rgba(100,116,139,.25);
    border-radius:8px;
    padding:12px;
    text-align:center;
}
.metric-val { font-size:22px; font-weight:700; color:#E2E8F0; }
.metric-lbl { font-size:11px; color:#64748B; margin-top:2px; }

/* ── Homography matrix ── */
.h-matrix {
    font-family:'JetBrains Mono',monospace;
    font-size:12px;
    background:rgba(15,23,42,.7);
    border:1px solid rgba(99,102,241,.3);
    border-radius:8px;
    padding:12px 16px;
    color:#A5B4FC;
    line-height:2;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([4, 1])
with c1:
    st.title("🔬 CekTimpa — SIFT Copy-Move Detector")
    st.caption(
        "Deteksi Pemalsuan Salin-Tempel Berbasis **SIFT + FLANN + RANSAC** · "
        "Amerini dkk., *IEEE TIFS 2011* · Invariant terhadap Rotasi, Skala, dan Perspektif 3D"
    )
with c2:
    st.markdown("""
    <div style="text-align:right;padding-top:18px;">
      <span style="border:1px solid #334155;padding:4px 10px;border-radius:6px;font-size:11px;color:#64748B;">
        Informatika UNS · PCD
      </span>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Tabs Utama
# ─────────────────────────────────────────────────────────────────────────────
tab_detect, tab_algo, tab_dataset, tab_experiment, tab_tim = st.tabs([
    "🔍 Deteksi",
    "📐 Cara Kerja Algoritma",
    "📦 Dataset",
    "📊 Eksperimen",
    "👥 Tim",
])

# =============================================================================
# TAB 1: DETEKSI
# =============================================================================
with tab_detect:
    col_ctrl, col_disp = st.columns([4, 7], gap="large")

    # ── Kolom Kiri: Input & Parameter ────────────────────────────────────────
    with col_ctrl:
        st.subheader("1. Masukan Citra")

        base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_dir = os.path.join(base_dir, "dataset")
        has_dataset = False
        coverage_files = []
        if os.path.isdir(dataset_dir):
            for root, _, files in os.walk(dataset_dir):
                for f in sorted(files):
                    if f.lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg")):
                        coverage_files.append(os.path.relpath(os.path.join(root, f), dataset_dir))
            has_dataset = len(coverage_files) > 0

        opts = ["Unggah Berkas Sendiri", "Gunakan Citra Sintetis Uji"]
        if has_dataset:
            opts.insert(0, "Sampel Dataset COVERAGE")

        input_source = st.radio("Sumber citra:", options=opts, horizontal=True)

        selected_image = None
        st.session_state["synthetic_gt_mask"] = None

        if input_source == "Sampel Dataset COVERAGE" and has_dataset:
            selected_rel = st.selectbox(
                "Pilih citra dari dataset/:",
                options=coverage_files,
                format_func=lambda x: f"{x} (🔴 Tampered)" if "t.tif" in x.lower() else f"{x} (🟢 Original)",
            )
            selected_image = Image.open(os.path.join(dataset_dir, selected_rel)).convert("RGB")

        elif input_source == "Unggah Berkas Sendiri":
            uploaded = st.file_uploader("Pilih berkas (JPG/PNG/TIF):", type=["jpg","jpeg","png","tif","tiff"])
            if uploaded:
                selected_image = Image.open(uploaded).convert("RGB")
        else:
            sample_choice = st.selectbox(
                "Pilih sampel sintetis:",
                options=[
                    "Sampel 1: Ada Manipulasi (Duplikasi Objek Bertekstur)",
                    "Sampel 2: Tidak Ada Manipulasi (Citra Asli Objek Tunggal)",
                    "Sampel 3: Tidak Ada Manipulasi (Citra Asli Multi-Objek)",
                ],
            )
            arr, gt_mask, meta = generate_synthetic_forgery(sample_choice)
            selected_image = Image.fromarray(arr)
            st.session_state["synthetic_gt_mask"] = gt_mask
            st.caption(f"💡 {meta['description']}")

        st.divider()

        st.divider()

        # ── Pilihan Algoritma ────────────────────────────────────────────────
        st.subheader("2. Pilihan Algoritma Deteksi")
        selected_algo = st.radio(
            "Pilih Metode:",
            options=[
                "🔬 SIFT (Amerini dkk., 2011) — Direkomendasikan",
                "🧱 DCT-Bilateral (Kaplan dkk., 2020) — Blok Spasial",
            ],
            index=0,
            help="SIFT tahan rotasi, skala, dan kompresi (cocok untuk foto riil). DCT-Bilateral memeriksa blok piksel murni.",
        )

        is_sift_mode = "SIFT" in selected_algo

        if is_sift_mode:
            # ── Preset Profil 1-Klik ──────────────────────────────────────────
            preset_choice = st.selectbox(
                "⚡ Preset Cepat (1-Klik):",
                options=[
                    "📷 Objek 3D & Ruangan Nyata (Telepon, Meja, Interior)",
                    "🎯 Standar / Default (Foto Umum 2D)",
                    "🔍 Objek Kecil / Halus (Stempel, Koin, Logo)",
                    "⚙️ Kustom Manual",
                ],
                index=0,
                help="Pilih preset agar sistem otomatis mengatur takaran parameter terbaik sesuai jenis objek.",
            )

            if "Objek 3D" in preset_choice:
                def_ratio = 0.70
                def_dmin = 60
                def_ransac = 3.5
                def_inliers = 8
                def_radius = 10
                def_area = 50
                def_contrast = 0.03
                def_edge = 10
                st.caption("✨ *Preset 3D aktif:* Rasio selektif (0.70) untuk mengabaikan keyboard/speaker kembar, jarak d_min 60px, dan radius presisi 10px.")
            elif "Objek Kecil" in preset_choice:
                def_ratio = 0.78
                def_dmin = 20
                def_ransac = 5.0
                def_inliers = 5
                def_radius = 8
                def_area = 20
                def_contrast = 0.03
                def_edge = 15
                st.caption("✨ *Preset Objek Kecil aktif:* Ambang inlier diturunkan ke 5 dan radius ramping 8px.")
            else:
                def_ratio = 0.75
                def_dmin = 30
                def_ransac = 5.0
                def_inliers = 10
                def_radius = 18
                def_area = 100
                def_contrast = 0.04
                def_edge = 10

            with st.expander("🔩 Rincian Parameter SIFT (Opsional / Lanjutan)", expanded=False):
                c1p, c2p = st.columns(2)
                with c1p:
                    n_features = st.number_input(
                        "Maks. Keypoints (0=∞):", min_value=0, max_value=10000, value=0, step=100,
                        help="Batasi jumlah keypoint terdeteksi. 0 = tidak terbatas (semua keypoint digunakan).",
                    )
                    contrast_threshold = st.slider(
                        "Contrast Threshold:", min_value=0.01, max_value=0.10, value=float(def_contrast), step=0.005,
                        help="Keypoint dengan kontras < threshold dibuang.",
                    )
                with c2p:
                    edge_threshold = st.slider(
                        "Edge Threshold:", min_value=5, max_value=30, value=int(def_edge),
                        help="Filter keypoint di tepi (edge) yang tidak stabil.",
                    )
                    sigma = st.number_input(
                        "Gaussian σ:", min_value=1.0, max_value=3.0, value=1.6, step=0.1,
                        help="Simpangan baku Gaussian awal untuk scale-space (Lowe, 2004).",
                    )

                c1m, c2m = st.columns(2)
                with c1m:
                    ratio_threshold = st.slider(
                        "Lowe's Ratio (r):", min_value=0.50, max_value=0.95, value=float(def_ratio), step=0.01,
                        help="d(m₁)/d(m₂) < r → match diterima.",
                    )
                    d_min_sift = st.slider(
                        "Jarak Spasial Minimum (px):", min_value=10, max_value=150, value=int(def_dmin),
                        help="Pasangan keypoint dengan jarak < d_min dibuang.",
                    )
                with c2m:
                    ransac_threshold = st.slider(
                        "RANSAC Error (px):", min_value=1.0, max_value=20.0, value=float(def_ransac), step=0.5,
                        help="Threshold reprojection error untuk inlier RANSAC.",
                    )
                    min_inliers = st.slider(
                        "Min. Inlier RANSAC:", min_value=4, max_value=50, value=int(def_inliers),
                        help="Jumlah minimum inlier RANSAC agar terindikasi copy-move.",
                    )

                c1k, c2k = st.columns(2)
                with c1k:
                    kp_radius = st.slider(
                        "Radius Keypoint (px):", min_value=5, max_value=50, value=int(def_radius),
                        help="Radius lingkaran mask di setiap keypoint inlier.",
                    )
                with c2k:
                    area_min_sift = st.number_input(
                        "Min. Luas Komponen (px²):", min_value=10, max_value=2000, value=int(def_area), step=10,
                        help="Komponen mask lebih kecil dari nilai ini dihapus.",
                    )

            btn_run = st.button("▶  JALANKAN DETEKSI SIFT", use_container_width=True)

        else:
            with st.expander("🧱 Parameter Blok & DCT", expanded=True):
                c1b, c2b = st.columns(2)
                with c1b:
                    block_size = st.selectbox("Ukuran Blok (B):", options=[8, 16], index=0, help="Ukuran blok piksel persegi.")
                    stride = st.slider("Langkah Pergeseran (s):", min_value=1, max_value=4, value=2, help="Langkah pergeseran jendela.")
                with c2b:
                    q_step = st.slider("Kuantisasi Q:", min_value=4.0, max_value=32.0, value=16.0, step=2.0)
                    eps = st.slider("Ambang Epsilon (ε):", min_value=1.0, max_value=10.0, value=5.0, step=0.5)

            with st.expander("🌊 Filter Bilateral & Voting", expanded=True):
                c1v, c2v = st.columns(2)
                with c1v:
                    d_min_block = st.slider("Jarak Spasial d_min:", min_value=10.0, max_value=80.0, value=30.0, step=5.0)
                    threshold_t = st.slider("Ambang Voting Vektor (T):", min_value=5, max_value=80, value=25, step=5)
                with c2v:
                    bilateral_sigma_s = st.slider("Bilateral Sigma Space:", min_value=0.5, max_value=3.0, value=1.0, step=0.1)
                    bilateral_sigma_r = st.slider("Bilateral Sigma Color:", min_value=10.0, max_value=50.0, value=25.5, step=2.5)

            with st.expander("🗺️ Mask Morfologi Blok"):
                area_min_block = st.number_input("Min. Luas Komponen (px²):", min_value=10, max_value=1000, value=60, step=10)

            btn_run = st.button("▶  JALANKAN DETEKSI DCT-BILATERAL", use_container_width=True)

    # ── Kolom Kanan: Hasil & Visualisasi ─────────────────────────────────────
    with col_disp:
        st.subheader("3. Hasil Deteksi")

        if selected_image is None:
            st.info("Pilih citra masukan di kolom kiri terlebih dahulu.")
        else:
            image_np = np.array(selected_image)

            if not btn_run:
                st.image(
                    selected_image,
                    caption=f"Citra siap dianalisis ({selected_image.width}×{selected_image.height} px). "
                            "Klik tombol deteksi untuk memulai.",
                    use_container_width=True,
                )
            else:
                if is_sift_mode:
                    config = {
                        "n_features": int(n_features),
                        "contrast_threshold": float(contrast_threshold),
                        "edge_threshold": float(edge_threshold),
                        "sigma": float(sigma),
                        "ratio_threshold": float(ratio_threshold),
                        "d_min": float(d_min_sift),
                        "ransac_threshold": float(ransac_threshold),
                        "min_inliers": int(min_inliers),
                        "kp_radius": int(kp_radius),
                        "area_min": int(area_min_sift),
                    }
                    with st.spinner("Menjalankan pipeline SIFT → FLANN → Ratio Test → RANSAC → Mask…"):
                        result = detect_copymove_sift(image_np, config)

                    # ── Status Banner SIFT ─────────────────────────────────────────
                    if result.is_forged:
                        st.markdown(f"""
                        <div class="result-box result-alert">
                            <strong style="color:#EF4444;font-size:15px;">🔴 Terindikasi Copy-Move Forgery (SIFT)</strong><br>
                            Ditemukan <strong>{result.n_inliers} inlier keypoint</strong> yang geometrically consistent
                            setelah verifikasi RANSAC.<br>
                            <span style="color:#94A3B8;font-size:13px;">Transformasi: <code>{result.transform_type}</code></span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-box result-safe">
                            <strong style="color:#10B981;font-size:15px;">🟢 Tidak Terindikasi Copy-Move (SIFT)</strong><br>
                            Tidak ditemukan cukup keypoint inlier yang konsisten secara geometri.
                            ({result.n_inliers} inlier &lt; ambang batas {min_inliers})<br>
                            <span style="color:#94A3B8;font-size:13px;">Status: <code>{result.transform_type}</code></span>
                        </div>
                        """, unsafe_allow_html=True)

                    # ── Pipeline Ringkasan SIFT ─────────────────────────────────
                    st.markdown("**📊 Ringkasan Pipeline SIFT:**")
                    metrics_html = f"""
                    <div class="metric-row">
                      <div class="metric-box">
                        <div class="metric-val">{result.n_keypoints}</div>
                        <div class="metric-lbl">Keypoints L2</div>
                      </div>
                      <div class="metric-box">
                        <div class="metric-val">{result.n_candidates}</div>
                        <div class="metric-lbl">Kandidat L3–L4a</div>
                      </div>
                      <div class="metric-box">
                        <div class="metric-val">{result.n_spatial}</div>
                        <div class="metric-lbl">Filter Spasial L4b</div>
                      </div>
                      <div class="metric-box">
                        <div class="metric-val" style="color:{'#34D399' if result.is_forged else '#F87171'}">
                          {result.n_inliers}
                        </div>
                        <div class="metric-lbl">Inlier RANSAC L5</div>
                      </div>
                      <div class="metric-box">
                        <div class="metric-val">{result.timings.get('total_ms',0):.0f} ms</div>
                        <div class="metric-lbl">Waktu Total</div>
                      </div>
                    </div>
                    """
                    st.markdown(metrics_html, unsafe_allow_html=True)

                    # ── Timing per Langkah ────────────────────────────────────
                    with st.expander("⏱️ Timing Per Langkah (SIFT)", expanded=False):
                        t = result.timings
                        tcols = st.columns(5)
                        labels_t = [("L1 Grayscale","L1_grayscale_ms"),("L2 SIFT","L2_sift_ms"),
                                    ("L3 Matching","L3_matching_ms"),("L4 Filter","L4_filter_ms"),
                                    ("L5 RANSAC","L5_ransac_ms")]
                        for col, (lbl, key) in zip(tcols, labels_t):
                            col.metric(lbl, f"{t.get(key,0):.1f} ms")

                    # ── Homography Matrix ─────────────────────────────────────
                    if result.homography is not None:
                        with st.expander("🔢 Matriks Homografi H (3×3)", expanded=False):
                            H = result.homography
                            st.markdown(f"""
                            <div class="h-matrix">
                            H &nbsp;=&nbsp;
                            ⎡ {H[0,0]:+8.4f} &nbsp; {H[0,1]:+8.4f} &nbsp; {H[0,2]:+8.2f} ⎤<br>
                            &nbsp;&nbsp;&nbsp;&nbsp;⎢ {H[1,0]:+8.4f} &nbsp; {H[1,1]:+8.4f} &nbsp; {H[1,2]:+8.2f} ⎥<br>
                            &nbsp;&nbsp;&nbsp;&nbsp;⎣ {H[2,0]:+8.6f} &nbsp; {H[2,1]:+8.6f} &nbsp; {H[2,2]:+8.4f} ⎦<br><br>
                            <span style="color:#64748B">
                            Interpretasi: {result.transform_type}
                            </span>
                            </div>
                            """, unsafe_allow_html=True)

                    # ── Visualisasi Tab SIFT ──────────────────────────────────
                    gt_mask = st.session_state.get("synthetic_gt_mask")
                    has_gt = gt_mask is not None and np.any(gt_mask > 0)

                    tab_labels = [
                        "🎯 Overlay Final",
                        "L2 · Keypoints",
                        "L3–4a · Kandidat Match",
                        "L4b · Filter Spasial",
                        "L5 · RANSAC Inlier",
                        "🗺️ Mask Biner",
                    ]
                    if has_gt:
                        tab_labels.append("✅ Ground Truth")
                        tab_labels.append("📈 Evaluasi Metrik")
                    tab_labels.append("📷 Citra Asli")

                    tabs_vis = st.tabs(tab_labels)
                    idx = 0

                    with tabs_vis[idx]:
                        st.image(result.overlay, use_container_width=True, caption="Area merah = manipulasi copy-move terdeteksi.")
                    idx += 1

                    with tabs_vis[idx]:
                        st.image(result.step1_keypoints_img, use_container_width=True,
                                 caption=f"L2 SIFT: {result.n_keypoints} keypoints terdeteksi.")
                    idx += 1

                    with tabs_vis[idx]:
                        st.image(result.step2_candidates_img, use_container_width=True,
                                 caption=f"L3–L4a: {result.n_candidates} kandidat match setelah Lowe's Ratio Test.")
                    idx += 1

                    with tabs_vis[idx]:
                        st.image(result.step3_spatial_img, use_container_width=True,
                                 caption=f"L4b: {result.n_spatial} match valid setelah filter spasial (d_min={d_min_sift}px).")
                    idx += 1

                    with tabs_vis[idx]:
                        st.image(result.step4_ransac_img, use_container_width=True,
                                 caption=f"L5 RANSAC: {result.n_inliers} inlier geometrically consistent.")
                    idx += 1

                    with tabs_vis[idx]:
                        if np.any(result.mask > 0):
                            st.success(f"✓ Mask biner mendeteksi {result.forged_pixels_count:,} piksel manipulasi ({result.forged_area_percentage:.2f}% dari citra).")
                            st.image(result.mask, use_container_width=True, caption=f"L6 Mask Biner — Putih (255) = Copy-Move · Hitam (0) = Asli")
                        else:
                            st.warning(
                                f"⚠️ **Mask biner kosong (hitam)** karena citra ini dinyatakan **Bersih / Tidak Terindikasi Copy-Move**.\n\n"
                                f"• Inlier RANSAC yang ditemukan: **{result.n_inliers}** (syarat minimum: **{min_inliers}** inlier).\n"
                                f"• Jika objek ini seharusnya terdeteksi copy-move, coba turunkan slider **'Min. Inlier RANSAC'** ke `{max(4, result.n_inliers)}` "
                                f"atau naikkan **'Lowe\\'s Ratio'** di panel parameter sebelah kiri."
                            )
                            st.image(result.mask, use_container_width=True, caption="Mask Biner Kosong (0 px)")
                    idx += 1

                    if has_gt:
                        with tabs_vis[idx]:
                            st.image(gt_mask, use_container_width=True, caption="Ground Truth Mask.")
                        idx += 1

                        with tabs_vis[idx]:
                            pm = calculate_pixel_metrics(result.mask, gt_mask)
                            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                            m_c1.metric("Precision (P)", f"{pm['precision'] * 100:.1f}%")
                            m_c2.metric("Recall (R)", f"{pm['recall'] * 100:.1f}%")
                            m_c3.metric("F1-Score", f"{pm['f1'] * 100:.1f}%")
                            m_c4.metric("IoU (Jaccard)", f"{pm['iou'] * 100:.1f}%")
                            st.caption(f"Piksel True Positive: {pm['tp']} · False Positive: {pm['fp']} · False Negative: {pm['fn']}")
                        idx += 1

                    with tabs_vis[idx]:
                        st.image(selected_image, use_container_width=True, caption=f"Citra masukan ({selected_image.width}×{selected_image.height} px).")

                else:
                    # ── DCT-Bilateral Execution ──────────────────────────────
                    config_b = {
                        "block_size": int(block_size),
                        "stride": int(stride),
                        "d_min": float(d_min_block),
                        "threshold_t": int(threshold_t),
                        "q_step": float(q_step),
                        "eps": float(eps),
                        "area_min": int(area_min_block),
                        "bilateral_d": 3,
                        "bilateral_sigma_s": float(bilateral_sigma_s),
                        "bilateral_sigma_r": float(bilateral_sigma_r),
                    }
                    with st.spinner("Menjalankan pipeline DCT-Bilateral → Kuantisasi → Shift Voting → Mask…"):
                        result_b = detect_copymove_block(image_np, config_b)

                    # ── Status Banner Bilateral ───────────────────────────────
                    if result_b.is_forged:
                        st.markdown(f"""
                        <div class="result-box result-alert">
                            <strong style="color:#EF4444;font-size:15px;">🔴 Terindikasi Copy-Move Forgery (DCT-Bilateral)</strong><br>
                            Ditemukan <strong>{len(result_b.pairs)} pasangan blok kembar</strong> dengan 
                            <strong>{len(result_b.shifts)} vektor translasi dominan</strong>.<br>
                            <span style="color:#94A3B8;font-size:13px;">Vektor Pergeseran Terkuat: <code>{result_b.shifts[0].tolist() if len(result_b.shifts) > 0 else '—'}</code></span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-box result-safe">
                            <strong style="color:#10B981;font-size:15px;">🟢 Tidak Terindikasi Copy-Move (DCT-Bilateral)</strong><br>
                            Tidak ada klaster vektor pergeseran yang melampaui ambang batas voting T={threshold_t}.
                        </div>
                        """, unsafe_allow_html=True)

                    # ── Ringkasan Bilateral ───────────────────────────────────
                    metrics_html = f"""
                    <div class="metric-row">
                      <div class="metric-box">
                        <div class="metric-val">{result_b.metadata.get('n_blocks', 0)}</div>
                        <div class="metric-lbl">Total Blok</div>
                      </div>
                      <div class="metric-box">
                        <div class="metric-val">{len(result_b.pairs)}</div>
                        <div class="metric-lbl">Pasangan Blok</div>
                      </div>
                      <div class="metric-box">
                        <div class="metric-val">{len(result_b.shifts)}</div>
                        <div class="metric-lbl">Vektor Translasi</div>
                      </div>
                      <div class="metric-box">
                        <div class="metric-val">{result_b.timings.get('total_ms', 0):.0f} ms</div>
                        <div class="metric-lbl">Waktu Total</div>
                      </div>
                    </div>
                    """
                    st.markdown(metrics_html, unsafe_allow_html=True)

                    # ── Visualisasi Tab Bilateral ─────────────────────────────
                    gt_mask = st.session_state.get("synthetic_gt_mask")
                    has_gt = gt_mask is not None and np.any(gt_mask > 0)

                    tab_b_labels = [
                        "🎯 Overlay Final",
                        "📍 Garis Pergeseran (Shift)",
                        "🗺️ Mask Biner",
                        "🔲 Boundary (Garis Tepi)",
                    ]
                    if has_gt:
                        tab_b_labels.append("✅ Ground Truth")
                        tab_b_labels.append("📈 Evaluasi Metrik")
                    tab_b_labels.append("📷 Citra Asli")

                    tabs_b = st.tabs(tab_b_labels)
                    idx = 0

                    with tabs_b[idx]:
                        st.image(result_b.overlay, use_container_width=True, caption="Area merah = manipulasi terdeteksi.")
                    idx += 1

                    with tabs_b[idx]:
                        st.image(result_b.shift_overlay, use_container_width=True, caption="Garis kuning menghubungkan pasangan blok kembar.")
                    idx += 1

                    with tabs_b[idx]:
                        if np.any(result_b.mask > 0):
                            st.success(f"✓ Mask biner mendeteksi {result_b.forged_pixels_count:,} piksel manipulasi ({result_b.forged_area_percentage:.2f}% dari citra).")
                            st.image(result_b.mask, use_container_width=True, caption="Mask Biner — Putih (255) = Copy-Move · Hitam (0) = Asli")
                        else:
                            st.warning(
                                f"⚠️ **Mask biner kosong (hitam)** karena tidak ada klaster vektor translasi yang mencapai ambang voting T={threshold_t}.\n\n"
                                f"• Coba turunkan **'Ambang Voting Vektor (T)'** ke 10–15, atau gunakan metode **SIFT** jika objek mengalami rotasi/skala."
                            )
                            st.image(result_b.mask, use_container_width=True, caption="Mask Biner Kosong (0 px)")
                    idx += 1

                    with tabs_b[idx]:
                        st.image(result_b.boundary, use_container_width=True, caption="Boundary: β(A) = A - (A ⊖ SE).")
                    idx += 1

                    if has_gt:
                        with tabs_b[idx]:
                            st.image(gt_mask, use_container_width=True, caption="Ground Truth Mask.")
                        idx += 1

                        with tabs_b[idx]:
                            pm = calculate_pixel_metrics(result_b.mask, gt_mask)
                            m_c1, m_c2, m_c3, m_c4 = st.columns(4)
                            m_c1.metric("Precision (P)", f"{pm['precision'] * 100:.1f}%")
                            m_c2.metric("Recall (R)", f"{pm['recall'] * 100:.1f}%")
                            m_c3.metric("F1-Score", f"{pm['f1'] * 100:.1f}%")
                            m_c4.metric("IoU (Jaccard)", f"{pm['iou'] * 100:.1f}%")
                            st.caption(f"Piksel True Positive: {pm['tp']} · False Positive: {pm['fp']} · False Negative: {pm['fn']}")
                        idx += 1

                    with tabs_b[idx]:
                        st.image(selected_image, use_container_width=True, caption=f"Citra masukan ({selected_image.width}×{selected_image.height} px).")


# =============================================================================
# TAB 2: CARA KERJA ALGORITMA (Explainer Edukatif)
# =============================================================================
with tab_algo:
    st.subheader("Cara Kerja: Pipeline SIFT Copy-Move Detection")
    st.caption("Amerini, Ballan, Caldelli, Del Bimbo, Serra — IEEE TIFS 2011")

    st.markdown("""
    <div style="background:rgba(99,102,241,.08);border:1px solid rgba(99,102,241,.3);border-radius:10px;padding:14px 18px;margin-bottom:16px;">
    <strong style="color:#A5B4FC;">Mengapa SIFT?</strong>
    <span style="color:#94A3B8;font-size:13px;">
    DCT-Bilateral (Kaplan 2020) hanya mendeteksi translasi murni (copy-paste piksel eksak tanpa transformasi).
    SIFT mampu mendeteksi duplikasi yang telah mengalami <strong>rotasi, perubahan skala, perubahan pencahayaan,
    dan distorsi perspektif 3D</strong> — cocok untuk foto kamera dunia nyata.
    </span>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline Diagram
    steps_info = [
        {
            "num": "1", "title": "Konversi Grayscale",
            "formula": "Y = 0.299·R + 0.587·G + 0.114·B  (Luminance ITU-R 601)",
            "desc": "Reduksi 3 saluran warna menjadi 1 saluran intensitas. SIFT bekerja pada intensitas "
                    "bukan warna. Bobot luminance memprioritaskan sensitivitas mata manusia terhadap hijau.",
            "icon": "🎨",
        },
        {
            "num": "2", "title": "SIFT Keypoint Detection & Description",
            "formula": (
                "Scale Space:  L(x,y,σ) = G(x,y,σ) ∗ I(x,y)\n"
                "DoG:          D(x,y,σ) = L(x,y,kσ) − L(x,y,σ)\n"
                "Descriptor:   4×4 patch × 8-bin HOG → 128-dim L2-normalized vector"
            ),
            "desc": (
                "① Bangun Gaussian Scale Space (multi-oktaf, multi-σ)\n"
                "② Hitung Difference of Gaussian (DoG) antar level\n"
                "③ Cari local extrema 3D (x, y, σ) pada DoG\n"
                "④ Lokalisasi sub-piksel via Taylor expansion\n"
                "⑤ Filter low-contrast & edge keypoints\n"
                "⑥ Tentukan orientasi dominan dari gradient histogram (36 bins)\n"
                "⑦ Buat descriptor 128-dim dari 4×4 sub-patch × 8-bin HOG"
            ),
            "icon": "🔑",
        },
        {
            "num": "3", "title": "KNN Self-Matching (FLANN, k=3)",
            "formula": (
                "Match deskriptor dᵢ terhadap semua dⱼ (j≠i) dalam citra yang sama\n"
                "FLANN KD-Tree: O(N·log N)  [vs Brute-force O(N²)]"
            ),
            "desc": (
                "FLANN (Fast Library for Approximate Nearest Neighbors) membangun KD-Tree "
                "dari seluruh deskriptor, lalu mencari 3 tetangga terdekat untuk tiap deskriptor. "
                "k=3 karena indeks 0 adalah self-match (d=0, dibuang). "
                "Hasil: pasangan kandidat (dᵢ, dⱼ) yang mirip secara fitur 128-dim."
            ),
            "icon": "🔗",
        },
        {
            "num": "4", "title": "Lowe's Ratio Test + Spatial Distance Filter",
            "formula": (
                "Ratio Test:   d(m₁) / d(m₂) < r  (Lowe 2004, default r=0.75)\n"
                "Spatial:      ‖pᵢ − pⱼ‖₂ ≥ d_min  (default 30 px)"
            ),
            "desc": (
                "Ratio Test (David Lowe, 2004): match diterima hanya jika jarak ke tetangga terbaik "
                "jauh lebih kecil dari tetangga kedua. Ini menjamin disambiguasi — fitur yang mirip dengan "
                "banyak kandidat lain (ambiguous) dibuang.\n\n"
                "Spatial Filter: pasangan keypoint yang terlalu berdekatan dalam citra bukan indikasi "
                "copy-move, melainkan kebetulan tekstur lokal serupa."
            ),
            "icon": "⚗️",
        },
        {
            "num": "5", "title": "RANSAC Geometric Verification",
            "formula": (
                "H = argmin_{H} Σᵢ ‖pⱼ − H·pᵢ‖²\n"
                "DLT (Direct Linear Transform): Ah=0, solusi via SVD\n"
                "Inlier: ‖pⱼ − H·pᵢ‖ < ε_RANSAC  (default 5 px)"
            ),
            "desc": (
                "RANSAC (Random Sample Consensus) mengestimasi matriks homografi H 3×3 "
                "yang mendeskripsikan transformasi geometri antara region sumber dan region salinan.\n\n"
                "Iterasi: pilih 4 pasangan acak → estimasi H dengan DLT → hitung reprojection error "
                "untuk semua pasangan → hitung jumlah inlier → ulangi N kali → H dengan inlier terbanyak menang.\n\n"
                "Tipe H yang dapat dideteksi:\n"
                "  • Translasi:  H≈[[1,0,tx],[0,1,ty],[0,0,1]]\n"
                "  • Affine:     H[2]≈[0,0,1] (rotasi+skala tanpa perspektif)\n"
                "  • Proyektif:  H bebas (perspektif 3D penuh)"
            ),
            "icon": "📐",
        },
        {
            "num": "6", "title": "Mask Generation & Morphological Post-processing",
            "formula": (
                "① Gambar lingkaran r∝σ_kp di setiap keypoint inlier (sumber & tujuan)\n"
                "② Closing: A⊕SE ⊖ SE  (kernel ellipse 15×15)\n"
                "③ Area Threshold: buang komponen < area_min px²\n"
                "④ Boundary: β(A) = A − (A⊖SE)"
            ),
            "desc": (
                "Konversi pasangan keypoint inlier menjadi mask biner piksel-level:\n"
                "① Untuk setiap inlier match, gambar lingkaran terisi di posisi queryIdx (merah) dan trainIdx (hijau)\n"
                "② Radius ∝ ukuran keypoint (skala DoG) agar proporsional dengan objek\n"
                "③ Morphological Closing untuk menyambung area terfragmentasi\n"
                "④ Area thresholding membuang deteksi palsu kecil\n"
                "⑤ Garis batas (boundary) diekstrak untuk visualisasi tepi"
            ),
            "icon": "🗺️",
        },
    ]

    for i, step in enumerate(steps_info):
        st.markdown(f"""
        <div class="step-card">
            <span class="step-num">{step['num']}</span>
            <span class="step-title">{step['icon']} {step['title']}</span>
            <code class="step-formula">{step['formula']}</code>
            <pre class="step-desc" style="font-family:Inter,sans-serif;white-space:pre-wrap;">{step['desc']}</pre>
        </div>
        """, unsafe_allow_html=True)
        if i < len(steps_info) - 1:
            st.markdown('<div class="arrow-down">↓</div>', unsafe_allow_html=True)

    # ── Perbandingan Metode ────────────────────────────────────────────────────
    st.divider()
    st.subheader("Perbandingan: SIFT vs DCT-Bilateral")
    st.markdown("""
    | Properti | DCT-Bilateral (Kaplan 2020) | **SIFT (Amerini 2011)** |
    |----------|:--------------------------:|:----------------------:|
    | Invariant Translasi | ✅ | ✅ |
    | Invariant Rotasi | ❌ | ✅ |
    | Invariant Skala | ❌ | ✅ |
    | Toleran Perspektif 3D | ❌ | ✅ |
    | Toleran JPEG Noise | ⚠️ Terbatas | ✅ |
    | Kecepatan (256×256) | ~50–200 ms | ~10–60 ms |
    | Cocok untuk | Forensik dokumen digital | **Foto kamera dunia nyata** |
    | Dataset Benchmark | CoMoFoD, COVERAGE | MICC-F220, COVERAGE |
    """)

    st.info(
        "💡 **Catatan Ilmiah:** Untuk copy-move yang dikombinasikan dengan JPEG compression, "
        "SIFT masih lebih robust karena deskriptor 128-dim lebih toleran terhadap perubahan kecil "
        "antar blok 8×8 DCT dibandingkan koefisien DCT bilateral yang sangat sensitif terhadap noise kompresi.",
        icon="ℹ️",
    )

    # ── Pseudocode ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Pseudocode Lengkap")
    st.code("""
ALGORITHM: SIFT Copy-Move Detection (Amerini et al., 2011)
INPUT:  I  — citra RGB (H × W × 3)
OUTPUT: M  — mask biner (H × W), H — homografi 3×3

1:  Y ← luminance(I)                              # Grayscale konversi
2:  K, D ← SIFT_detect_describe(Y)                # K = keypoints, D = descriptor N×128
3:  M_raw ← FLANN_knnMatch(D, D, k=3)             # Self-matching k=3

4:  C ← ∅
5:  FOR each group g in M_raw:
6:    diff ← [m ∈ g : m.queryIdx ≠ m.trainIdx]
7:    IF len(diff) ≥ 2:
8:      m₁, m₂ ← diff[0], diff[1]
9:      IF m₁.distance < r · m₂.distance:         # Lowe's Ratio Test
10:       IF ‖K[m₁.queryIdx].pt − K[m₁.trainIdx].pt‖ ≥ d_min:  # Spatial Filter
11:         C ← C ∪ {m₁}

12: src ← [K[m.queryIdx].pt  for m in C]
13: dst ← [K[m.trainIdx].pt  for m in C]
14: H, mask_inlier ← RANSAC_findHomography(src, dst, ε=5px)
15: inliers ← [m for (m, flag) in zip(C, mask_inlier) if flag]

16: IF len(inliers) ≥ min_inliers:
17:   M ← draw_circles_at_keypoints(inliers)
18:   M ← morphological_close(M)
19:   M ← area_threshold(M)
20:   RETURN M, H, FORGED
21: ELSE:
22:   RETURN zeros(H,W), None, AUTHENTIC
""", language="python")


# =============================================================================
# TAB 3: DATASET
# =============================================================================
with tab_dataset:
    st.subheader("Dataset Acuan Benchmark Copy-Move Forgery")
    st.markdown("""
    #### 1. MICC-F220 — *Benchmark Utama SIFT Copy-Move*
    * **Penyusun:** Amerini dkk. (2011) — sama dengan penulis paper metode ini
    * **Karakteristik:** 220 citra (110 original + 110 tampered), resolusi 2048×1536 px
    * **Kinerja SIFT:** Precision ≈ 95%, Recall ≈ 97%

    #### 2. CoMoFoD — *Comprehensive Copy-Move Forgery Database*
    * **Penyusun:** Tralic dkk. (2013)
    * **Tautan:** [vcl.fer.hr/comofod](http://vcl.fer.hr/comofod/)
    * **Karakteristik:** 260 set, 512×512 px, dengan 5 kategori transformasi post-processing

    #### 3. COVERAGE Dataset
    * **Tautan:** [kaggle.com/coverage-dataset](https://www.kaggle.com/datasets/divg23/coverage-dataset)
    * **Karakteristik:** 100 foto objek fisik dengan copy-move mirip (challenging: same-object)

    #### 4. CASIA 2.0
    * **Tautan:** [kaggle.com/casiav2](https://www.kaggle.com/datasets/sophatvathana/casiav2)
    * **Karakteristik:** Manipulasi realistic, copy-move + splicing

    #### 5. Generator Sintetis Internal (`src/forgery_generator.py`)
    * Gradien non-periodik + objek bertekstur kaya
    * Ground truth mask biner presisi piksel
    * Mendukung 3 skenario: tampered, clean-single, clean-multi
    """)


# =============================================================================
# TAB 4: EKSPERIMEN
# =============================================================================
with tab_experiment:
    st.subheader("Matriks Rencana Eksperimen (IEEE Format)")
    st.table({
        "Kode": ["E1", "E2", "E3", "E4", "E5", "E6"],
        "Nama Eksperimen": [
            "SIFT vs DCT-Bilateral vs Baseline",
            "Sensitivitas Parameter SIFT",
            "Uji Ketahanan JPEG & Noise",
            "Uji Rotasi & Skala Patch",
            "Uji Perspektif 3D",
            "Evaluasi Dataset Real-World",
        ],
        "Variabel yang Diuji": [
            "SIFT vs DCT-Bilateral vs DCT Standar",
            "ratio r ∈ {0.6,0.7,0.75,0.8}, d_min ∈ {20–80px}",
            "JPEG Q=20–90, Gaussian σ=1–10",
            "Rotasi 5°–45°, Skala 0.7×–1.4×",
            "Perspektif affine & projective transform",
            "COVERAGE, MICC-F220, CASIA 2.0",
        ],
        "Metrik Keluaran": [
            "Precision, Recall, F1, AUC-ROC",
            "Kurva F1 vs Parameter",
            "Grafik F1 vs PSNR (dB)",
            "Kurva F1 vs Sudut / Skala",
            "F1 per tipe transformasi",
            "TP Rate, FP Rate, Runtime",
        ],
    })


# =============================================================================
# TAB 5: TIM
# =============================================================================
with tab_tim:
    st.subheader("Tim Pengembang CekTimpa")
    st.markdown("""
    **Departemen Informatika, Universitas Sebelas Maret (UNS) Surakarta**
    *Mata Kuliah: Pengolahan Citra Digital (PCD)*

    * **Yashif Victoriawan**
    * **Fairuz**
    * **Fadhil**
    * **Andre**

    ---
    **Metode:** SIFT Copy-Move Detection · Amerini dkk., IEEE TIFS 2011  
    **Stack:** Python 3.11 · OpenCV 5.0 · Streamlit · NumPy · Docker
    """)
