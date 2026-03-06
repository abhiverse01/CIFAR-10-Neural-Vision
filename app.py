"""
app.py  —  CIFAR-10 Neural Vision · Streamlit Edition
Rewrite: ResNet-style CNN + Grad-CAM + rich analytics dashboard.
Run: streamlit run app.py
"""

import os
import sys
import io
import numpy as np
import streamlit as st
from PIL import Image

# ─── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="CIFAR-10 Neural Vision",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from utils.predictor import (
    predict_multiclass, predict_binary,
    get_gradcam_heatmap, overlay_gradcam,
    CLASS_NAMES_CLEAN, CLASS_EMOJIS,
)


# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Import fonts */
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

  /* Root variables */
  :root {
    --bg:        #05070f;
    --surface:   #0d1117;
    --surface2:  #161b27;
    --border:    #1f2937;
    --accent:    #00f5a0;
    --accent2:   #00d4ff;
    --accent3:   #ff6b6b;
    --text:      #e2e8f0;
    --muted:     #64748b;
  }

  /* Global reset */
  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
  }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }
  .block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1400px !important; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stSidebar"] * { color: var(--text) !important; }

  /* ── Hero header ── */
  .hero {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
    position: relative;
  }
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 60%, var(--accent3) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    letter-spacing: -0.03em;
    margin: 0;
  }
  .hero-sub {
    color: var(--muted);
    font-size: 0.95rem;
    margin-top: 0.6rem;
    font-family: 'Space Mono', monospace;
  }
  .hero-line {
    display: block;
    width: 80px;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    margin: 1.2rem auto 0;
    border-radius: 2px;
  }

  /* ── Cards ── */
  .glass-card {
    background: var(--surface) !important;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
  }
  .glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent3));
  }

  /* ── Result badges ── */
  .badge-multiclass {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: linear-gradient(135deg, #003d26, #004d30);
    border: 1px solid var(--accent);
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-size: 1.05rem;
    font-weight: 700;
    padding: 0.5rem 1.2rem;
    border-radius: 50px;
    letter-spacing: 0.04em;
  }
  .badge-binary-pos {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: linear-gradient(135deg, #1a0533, #2d0a5e);
    border: 1px solid var(--accent2);
    color: var(--accent2);
    font-family: 'Space Mono', monospace;
    font-size: 1rem; font-weight: 700;
    padding: 0.5rem 1.2rem; border-radius: 50px;
  }
  .badge-binary-neg {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: linear-gradient(135deg, #2d0a0a, #4a1010);
    border: 1px solid var(--accent3);
    color: var(--accent3);
    font-family: 'Space Mono', monospace;
    font-size: 1rem; font-weight: 700;
    padding: 0.5rem 1.2rem; border-radius: 50px;
  }

  /* ── Confidence bar ── */
  .conf-bar-wrap { margin: 0.3rem 0; }
  .conf-label {
    display: flex; justify-content: space-between;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem; color: var(--muted);
    margin-bottom: 3px;
  }
  .conf-label .cls { color: var(--text); font-weight: 700; }
  .conf-bar-bg {
    background: var(--surface2);
    border-radius: 4px; height: 8px;
    overflow: hidden; border: 1px solid var(--border);
  }
  .conf-bar-fill {
    height: 100%; border-radius: 4px;
    transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
  }
  .conf-bar-fill.top { background: linear-gradient(90deg, var(--accent), var(--accent2)); }
  .conf-bar-fill.mid { background: linear-gradient(90deg, #7c3aed, #a78bfa); }
  .conf-bar-fill.low { background: var(--surface2); }

  /* ── Metric cards ── */
  .metric-row { display: flex; gap: 1rem; margin: 1rem 0; flex-wrap: wrap; }
  .metric-card {
    flex: 1; min-width: 130px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
  }
  .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem; font-weight: 700;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }
  .metric-label {
    font-size: 0.7rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.1em; margin-top: 3px;
  }

  /* ── Upload zone ── */
  [data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 14px !important;
    transition: border-color 0.2s;
  }
  [data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: var(--surface2) !important;
    border-radius: 10px;
    gap: 4px;
    padding: 4px;
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    padding: 0.5rem 1rem !important;
  }
  .stTabs [aria-selected="true"] {
    background: var(--surface) !important;
    color: var(--accent) !important;
    border: 1px solid var(--border) !important;
  }

  /* ── Section labels ── */
  .section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem; color: var(--accent);
    text-transform: uppercase; letter-spacing: 0.15em;
    margin-bottom: 0.6rem; display: flex; align-items: center; gap: 6px;
  }
  .section-label::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
  }

  /* ── Buttons ── */
  .stButton button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #05070f !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.6rem 2rem !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    transition: opacity 0.2s !important;
  }
  .stButton button:hover { opacity: 0.85 !important; }

  /* ── Sidebar info ── */
  .info-pill {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.5rem 0.8rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
    margin-bottom: 0.4rem;
  }
  .info-pill span { color: var(--accent); font-weight: 700; }

  /* ── Spinner ── */
  .stSpinner > div { border-color: var(--accent) transparent transparent !important; }

  /* ── Alert ── */
  .stAlert { border-radius: 10px !important; }

  /* ── Image display ── */
  [data-testid="stImage"] img {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
  }
  
  /* ── Progress bars (Streamlit native) ── */
  .stProgress > div > div > div { background: var(--accent) !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: var(--surface); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  /* ── Divider ── */
  hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─── Model loading (cached) ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    import tensorflow as tf
    mc_path = os.path.join(ROOT, "models", "model_multiclass.keras")
    bin_path = os.path.join(ROOT, "models", "model_binary.keras")

    # Fall back to legacy .h5 if new format not found
    if not os.path.exists(mc_path):
        mc_path = os.path.join(ROOT, "models", "model_multiclass.h5")
    if not os.path.exists(bin_path):
        bin_path = os.path.join(ROOT, "models", "model_binary.h5")

    mc_model = tf.keras.models.load_model(mc_path)   if os.path.exists(mc_path)  else None
    bin_model = tf.keras.models.load_model(bin_path) if os.path.exists(bin_path) else None
    return mc_model, bin_model


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
      <div style='font-family:Space Mono,monospace; font-size:1.4rem; 
                  background:linear-gradient(135deg,#00f5a0,#00d4ff);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                  font-weight:700;'>🧠 NeuralVision</div>
      <div style='color:#64748b; font-size:0.7rem; font-family:Space Mono,monospace;
                  letter-spacing:0.1em; margin-top:4px;'>CIFAR-10 · CNN CLASSIFIER</div>
    </div>
    <hr style='border-color:#1f2937; margin:0.8rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown("**⚙️ Settings**")
    show_gradcam = st.toggle("Show Grad-CAM heatmap", value=True)
    show_all_probs = st.toggle("Show all class probabilities", value=True)
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05,
                                     help="Flag low-confidence predictions")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**📐 Architecture**")

    arch_info = [
        ("Model type", "ResNet-style CNN"),
        ("Input size", "32 × 32 × 3"),
        ("Augmentation", "Flip / Rotate / Zoom"),
        ("Regularization", "L2 + Dropout"),
        ("Optimizer", "AdamW + Cosine LR"),
        ("Loss (MC)", "SparseCatXE + Smoothing"),
        ("Loss (Bin)", "BinaryCrossEntropy"),
        ("Classes", "10 (CIFAR-10)"),
    ]
    for k, v in arch_info:
        st.markdown(f"""
        <div class='info-pill'>{k}: <span>{v}</span></div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**🎯 CIFAR-10 Classes**")
    classes_html = "".join(
        f"<div class='info-pill'>{CLASS_EMOJIS[i]} <span>{CLASS_NAMES_CLEAN[i]}</span></div>"
        for i in range(10)
    )
    st.markdown(classes_html, unsafe_allow_html=True)


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <h1 class='hero-title'>Neural Vision</h1>
  <p class='hero-sub'>ResNet-style CIFAR-10 · Dual Binary + Multi-Class · Grad-CAM Explainability</p>
  <span class='hero-line'></span>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── Load models ──────────────────────────────────────────────────────────────
with st.spinner("⚡ Initialising neural networks..."):
    mc_model, bin_model = load_models()

models_ok = mc_model is not None and bin_model is not None

if not models_ok:
    st.markdown("""
    <div class='glass-card'>
      <div class='section-label'>⚠️ Model Status</div>
      <p style='color:#ff6b6b; font-family:Space Mono,monospace; font-size:0.85rem;'>
        No trained models found. Run the trainer first:
      </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 How to train & run", expanded=True):
        st.code("""
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the models (saves to models/)
python models/model_builder.py

# 3. Launch Streamlit
streamlit run app.py
        """, language="bash")

    st.stop()


# ─── Main UI ──────────────────────────────────────────────────────────────────
col_upload, col_results = st.columns([1, 1.6], gap="large")

# ── Upload column ──
with col_upload:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>📁 Image Upload</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop an image or click to browse",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded image", use_container_width=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>📊 Image Info</div>", unsafe_allow_html=True)
        w, h = img.size
        st.markdown(f"""
        <div class='metric-row'>
          <div class='metric-card'>
            <div class='metric-value'>{w}×{h}</div>
            <div class='metric-label'>Resolution</div>
          </div>
          <div class='metric-card'>
            <div class='metric-value'>{uploaded.name.split('.')[-1].upper()}</div>
            <div class='metric-label'>Format</div>
          </div>
          <div class='metric-card'>
            <div class='metric-value'>{round(len(uploaded.getvalue())/1024, 1)}<span style='font-size:0.9rem'>KB</span></div>
            <div class='metric-label'>File Size</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Sample images hint
    st.markdown("""
    <div style='color:#64748b; font-size:0.72rem; font-family:Space Mono,monospace;
                text-align:center; margin-top:0.5rem;'>
      💡 Works best with clear, single-subject photos.
    </div>
    """, unsafe_allow_html=True)


# ── Results column ──
with col_results:
    if not uploaded:
        st.markdown("""
        <div class='glass-card' style='min-height:320px; display:flex; flex-direction:column;
             align-items:center; justify-content:center; text-align:center;'>
          <div style='font-size:3rem; margin-bottom:1rem;'>🔬</div>
          <div style='font-family:Space Mono,monospace; color:#64748b; font-size:0.85rem;
                      line-height:1.8;'>
            Upload an image to run<br>dual-model inference.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        with st.spinner("🧠 Running inference..."):
            mc_result  = predict_multiclass(mc_model, img)
            bin_result = predict_binary(bin_model, img)
            heatmap    = get_gradcam_heatmap(mc_model, img) if show_gradcam else None

        # ── Low-confidence warning ──
        if mc_result["confidence"] < confidence_threshold:
            st.warning(
                f"⚠️ Low confidence ({mc_result['confidence']:.1%}). "
                "The model is uncertain — try a clearer, single-subject image."
            )

        # ── Top result banner ──
        mc_conf_pct  = f"{mc_result['confidence']:.1%}"
        bin_conf_pct = f"{bin_result['confidence']:.1%}"

        st.markdown(f"""
        <div class='glass-card'>
          <div class='section-label'>🎯 Prediction Results</div>
          <div style='display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1rem;'>
            <div>
              <div style='font-size:0.65rem; color:#64748b; font-family:Space Mono,monospace;
                          text-transform:uppercase; letter-spacing:0.1em; margin-bottom:5px;'>
                Multi-Class
              </div>
              <span class='badge-multiclass'>{mc_result["emoji"]} {mc_result["class"].upper()}</span>
            </div>
            <div>
              <div style='font-size:0.65rem; color:#64748b; font-family:Space Mono,monospace;
                          text-transform:uppercase; letter-spacing:0.1em; margin-bottom:5px;'>
                Binary (Airplane?)
              </div>
              <span class='{"badge-binary-pos" if bin_result["is_airplane"] else "badge-binary-neg"}'>
                {"✅ YES" if bin_result["is_airplane"] else "❌ NO"} · {bin_conf_pct}
              </span>
            </div>
          </div>

          <div class='metric-row'>
            <div class='metric-card'>
              <div class='metric-value'>{mc_conf_pct}</div>
              <div class='metric-label'>MC Confidence</div>
            </div>
            <div class='metric-card'>
              <div class='metric-value'>{mc_result["class_index"]}</div>
              <div class='metric-label'>Class ID</div>
            </div>
            <div class='metric-card'>
              <div class='metric-value'>{f"{bin_result['airplane_prob']:.2f}"}</div>
              <div class='metric-label'>Airplane Prob</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Tabs: Probabilities / Grad-CAM ──
        tab_probs, tab_gradcam, tab_raw = st.tabs(
            ["📊 Class Probabilities", "🔥 Grad-CAM", "🔢 Raw Scores"]
        )

        with tab_probs:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>All class confidences</div>", unsafe_allow_html=True)

            probs = mc_result["probabilities"]
            sorted_idx = np.argsort(probs)[::-1]

            for rank, idx in enumerate(sorted_idx if show_all_probs else sorted_idx[:5]):
                pct = probs[idx]
                tier = "top" if rank == 0 else ("mid" if rank < 3 else "low")
                st.markdown(f"""
                <div class='conf-bar-wrap'>
                  <div class='conf-label'>
                    <span class='cls'>{CLASS_EMOJIS[idx]} {CLASS_NAMES_CLEAN[idx]}</span>
                    <span>{pct:.2%}</span>
                  </div>
                  <div class='conf-bar-bg'>
                    <div class='conf-bar-fill {tier}' style='width:{pct*100:.1f}%'></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        with tab_gradcam:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Gradient-weighted Class Activation Map</div>",
                        unsafe_allow_html=True)
            if heatmap is not None:
                overlay = overlay_gradcam(img, heatmap, alpha=0.45)
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.image(img.resize((128, 128), Image.NEAREST),
                             caption="Original", use_container_width=True)
                with c2:
                    hmap_display = Image.fromarray(
                        (__import__('matplotlib').cm.get_cmap("jet")(heatmap)[:, :, :3] * 255).astype(np.uint8)
                    ).resize((128, 128), Image.NEAREST)
                    st.image(hmap_display, caption="Heatmap", use_container_width=True)
                with c3:
                    st.image(overlay.resize((128, 128), Image.LANCZOS),
                             caption="Overlay", use_container_width=True)
                st.markdown("""
                <div style='color:#64748b; font-size:0.72rem; font-family:Space Mono,monospace;
                            margin-top:0.5rem;'>
                  🔥 Hot (red) regions = areas the model focused on for its prediction.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Enable Grad-CAM in sidebar settings or toggle it on.")
            st.markdown("</div>", unsafe_allow_html=True)

        with tab_raw:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Raw softmax output</div>", unsafe_allow_html=True)
            raw_data = {
                "Class": [f"{CLASS_EMOJIS[i]} {CLASS_NAMES_CLEAN[i]}" for i in range(10)],
                "Probability": [f"{p:.6f}" for p in probs],
                "Logit %": [f"{p*100:.3f}%" for p in probs],
            }
            import pandas as pd
            df = pd.DataFrame(raw_data)
            df = df.sort_values("Probability", ascending=False).reset_index(drop=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)


# ─── Bottom: How it works ─────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("⚙️ How this works", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        **🏗️ Architecture**
        - ResNet-style residual blocks
        - BatchNorm + Dropout (L2 reg)
        - GlobalAveragePooling head
        - No fully-connected bottleneck
        """)
    with c2:
        st.markdown("""
        **🎓 Training**
        - AdamW optimizer
        - Cosine annealing LR
        - Label smoothing (MC)
        - Data augmentation pipeline
        """)
    with c3:
        st.markdown("""
        **🔍 Explainability**
        - Grad-CAM visualisation
        - Last Conv layer gradients
        - Confidence calibration
        - Per-class probability bars
        """)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding:2rem 0 1rem; color:#1f2937;
            font-family:Space Mono,monospace; font-size:0.65rem; letter-spacing:0.1em;'>
  CIFAR-10 NEURAL VISION · RESNET CNN · DUAL CLASSIFIER · GRAD-CAM
</div>
""", unsafe_allow_html=True)
