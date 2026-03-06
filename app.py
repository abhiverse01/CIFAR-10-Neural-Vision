"""
app.py  —  CIFAR-10 Neural Vision · Streamlit + PyTorch
Run: streamlit run app.py
"""

import os, sys
import numpy as np
import streamlit as st
from PIL import Image
import torch

st.set_page_config(page_title="CIFAR-10 Neural Vision", page_icon="🧠", layout="wide",
                   initial_sidebar_state="expanded")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from utils.predictor import (
    predict_multiclass, predict_binary,
    get_gradcam_heatmap, overlay_gradcam,
    CLASS_NAMES, CLASS_EMOJIS,
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
  :root {
    --bg:#05070f; --surface:#0d1117; --surface2:#161b27; --border:#1f2937;
    --accent:#00f5a0; --accent2:#00d4ff; --accent3:#ff6b6b;
    --text:#e2e8f0; --muted:#64748b;
  }
  html,body,[class*="css"]{font-family:'Syne',sans-serif!important;background-color:var(--bg)!important;color:var(--text)!important;}
  #MainMenu,footer,header{visibility:hidden;}
  .stDeployButton{display:none;}
  .block-container{padding:1.5rem 2rem 3rem!important;max-width:1400px!important;}
  [data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
  [data-testid="stSidebar"] *{color:var(--text)!important;}
  .hero{text-align:center;padding:2.5rem 1rem 1rem;}
  .hero-title{font-family:'Syne',sans-serif;font-size:clamp(2rem,5vw,3.5rem);font-weight:800;
    background:linear-gradient(135deg,var(--accent) 0%,var(--accent2) 60%,var(--accent3) 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    line-height:1.1;letter-spacing:-.03em;margin:0;}
  .hero-sub{color:var(--muted);font-size:.95rem;margin-top:.6rem;font-family:'Space Mono',monospace;}
  .hero-line{display:block;width:80px;height:2px;background:linear-gradient(90deg,var(--accent),var(--accent2));margin:1.2rem auto 0;border-radius:2px;}
  .glass-card{background:var(--surface)!important;border:1px solid var(--border);border-radius:16px;
    padding:1.4rem 1.6rem;margin-bottom:1rem;position:relative;overflow:hidden;}
  .glass-card::before{content:'';position:absolute;top:0;left:0;width:100%;height:2px;
    background:linear-gradient(90deg,var(--accent),var(--accent2),var(--accent3));}
  .badge-mc{display:inline-flex;align-items:center;gap:.5rem;background:linear-gradient(135deg,#003d26,#004d30);
    border:1px solid var(--accent);color:var(--accent);font-family:'Space Mono',monospace;
    font-size:1.05rem;font-weight:700;padding:.5rem 1.2rem;border-radius:50px;letter-spacing:.04em;}
  .badge-pos{display:inline-flex;align-items:center;gap:.5rem;background:linear-gradient(135deg,#1a0533,#2d0a5e);
    border:1px solid var(--accent2);color:var(--accent2);font-family:'Space Mono',monospace;
    font-size:1rem;font-weight:700;padding:.5rem 1.2rem;border-radius:50px;}
  .badge-neg{display:inline-flex;align-items:center;gap:.5rem;background:linear-gradient(135deg,#2d0a0a,#4a1010);
    border:1px solid var(--accent3);color:var(--accent3);font-family:'Space Mono',monospace;
    font-size:1rem;font-weight:700;padding:.5rem 1.2rem;border-radius:50px;}
  .conf-label{display:flex;justify-content:space-between;font-family:'Space Mono',monospace;
    font-size:.75rem;color:var(--muted);margin-bottom:3px;}
  .conf-label .cls{color:var(--text);font-weight:700;}
  .conf-bar-bg{background:var(--surface2);border-radius:4px;height:8px;overflow:hidden;border:1px solid var(--border);}
  .conf-bar-fill{height:100%;border-radius:4px;}
  .conf-bar-fill.top{background:linear-gradient(90deg,var(--accent),var(--accent2));}
  .conf-bar-fill.mid{background:linear-gradient(90deg,#7c3aed,#a78bfa);}
  .conf-bar-fill.low{background:var(--surface2);}
  .metric-row{display:flex;gap:1rem;margin:1rem 0;flex-wrap:wrap;}
  .metric-card{flex:1;min-width:130px;background:var(--surface2);border:1px solid var(--border);border-radius:12px;padding:1rem 1.2rem;text-align:center;}
  .metric-value{font-family:'Space Mono',monospace;font-size:1.6rem;font-weight:700;
    background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
  .metric-label{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em;margin-top:3px;}
  .section-label{font-family:'Space Mono',monospace;font-size:.7rem;color:var(--accent);
    text-transform:uppercase;letter-spacing:.15em;margin-bottom:.6rem;display:flex;align-items:center;gap:6px;}
  .section-label::after{content:'';flex:1;height:1px;background:linear-gradient(90deg,var(--border),transparent);}
  .info-pill{background:var(--surface2);border:1px solid var(--border);border-radius:8px;
    padding:.5rem .8rem;font-family:'Space Mono',monospace;font-size:.72rem;color:var(--muted);margin-bottom:.4rem;}
  .info-pill span{color:var(--accent);font-weight:700;}
  [data-testid="stFileUploader"]{background:var(--surface2)!important;border:2px dashed var(--border)!important;border-radius:14px!important;}
  .stTabs [data-baseweb="tab-list"]{background:var(--surface2)!important;border-radius:10px;gap:4px;padding:4px;}
  .stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--muted)!important;border-radius:8px!important;
    font-family:'Space Mono',monospace!important;font-size:.8rem!important;padding:.5rem 1rem!important;}
  .stTabs [aria-selected="true"]{background:var(--surface)!important;color:var(--accent)!important;border:1px solid var(--border)!important;}
  .stButton button{background:linear-gradient(135deg,var(--accent),var(--accent2))!important;color:#05070f!important;
    font-family:'Space Mono',monospace!important;font-weight:700!important;border:none!important;
    border-radius:50px!important;padding:.6rem 2rem!important;font-size:.85rem!important;}
  [data-testid="stImage"] img{border-radius:12px!important;border:1px solid var(--border)!important;}
  hr{border-color:var(--border)!important;margin:1.5rem 0!important;}
  ::-webkit-scrollbar{width:6px;}
  ::-webkit-scrollbar-track{background:var(--surface);}
  ::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}
</style>
""", unsafe_allow_html=True)


# ─── Model loader (cached) ────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    from models.model_builder import MultiClassCNN, BinaryCNN
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mc_path  = os.path.join(ROOT, "models", "model_multiclass.pt")
    bin_path = os.path.join(ROOT, "models", "model_binary.pt")

    mc_model  = None
    bin_model = None

    if os.path.exists(mc_path):
        mc_model = MultiClassCNN()
        mc_model.load_state_dict(torch.load(mc_path, map_location=device))
        mc_model.eval()

    if os.path.exists(bin_path):
        bin_model = BinaryCNN()
        bin_model.load_state_dict(torch.load(bin_path, map_location=device))
        bin_model.eval()

    return mc_model, bin_model, device


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0 .5rem;'>
      <div style='font-family:Space Mono,monospace;font-size:1.4rem;
        background:linear-gradient(135deg,#00f5a0,#00d4ff);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;font-weight:700;'>🧠 NeuralVision</div>
      <div style='color:#64748b;font-size:.7rem;font-family:Space Mono,monospace;letter-spacing:.1em;margin-top:4px;'>
        CIFAR-10 · PYTORCH CNN</div>
    </div><hr>
    """, unsafe_allow_html=True)

    show_gradcam   = st.toggle("Show Grad-CAM", value=True)
    show_all_probs = st.toggle("Show all 10 classes", value=True)
    conf_threshold = st.slider("Confidence warning threshold", 0.0, 1.0, 0.5, 0.05)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("**📐 Architecture**")
    for k, v in [
        ("Framework", "PyTorch"), ("Architecture", "ResNet CNN"),
        ("Input", "32×32×3"), ("Augmentation", "Flip/Crop/Jitter"),
        ("Optimizer", "AdamW + CosineAnneal"), ("Loss (MC)", "CrossEntropy + ε=0.1"),
        ("Loss (Bin)", "BCEWithLogits"), ("Classes", "10 CIFAR-10"),
    ]:
        st.markdown(f"<div class='info-pill'>{k}: <span>{v}</span></div>", unsafe_allow_html=True)

    st.markdown("<hr>**🎯 Classes**", unsafe_allow_html=True)
    for i in range(10):
        st.markdown(f"<div class='info-pill'>{CLASS_EMOJIS[i]} <span>{CLASS_NAMES[i]}</span></div>",
                    unsafe_allow_html=True)


# ─── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <h1 class='hero-title'>Neural Vision</h1>
  <p class='hero-sub'>PyTorch ResNet · Dual Binary + Multi-Class · Grad-CAM · CIFAR-10</p>
  <span class='hero-line'></span>
</div><br>
""", unsafe_allow_html=True)


# ─── Load models ──────────────────────────────────────────────────────────────
with st.spinner("⚡ Loading models..."):
    mc_model, bin_model, device = load_models()

if mc_model is None or bin_model is None:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.error("⚠️ No trained models found in `models/`. Train them first:")
    st.code("""python models/model_builder.py""", language="bash")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# ─── Main layout ──────────────────────────────────────────────────────────────
col_up, col_res = st.columns([1, 1.6], gap="large")

with col_up:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-label'>📁 Image Upload</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload", type=["jpg","jpeg","png","webp","bmp"],
                                label_visibility="collapsed")
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded image", use_container_width=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        w, h = img.size
        st.markdown(f"""
        <div class='metric-row'>
          <div class='metric-card'><div class='metric-value'>{w}×{h}</div><div class='metric-label'>Resolution</div></div>
          <div class='metric-card'><div class='metric-value'>{uploaded.name.split('.')[-1].upper()}</div><div class='metric-label'>Format</div></div>
          <div class='metric-card'><div class='metric-value'>{round(len(uploaded.getvalue())/1024,1)}<span style='font-size:.9rem'>KB</span></div><div class='metric-label'>File Size</div></div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_res:
    if not uploaded:
        st.markdown("""
        <div class='glass-card' style='min-height:320px;display:flex;flex-direction:column;
             align-items:center;justify-content:center;text-align:center;'>
          <div style='font-size:3rem;margin-bottom:1rem;'>🔬</div>
          <div style='font-family:Space Mono,monospace;color:#64748b;font-size:.85rem;line-height:1.8;'>
            Upload an image to run<br>dual-model inference.</div>
        </div>""", unsafe_allow_html=True)
    else:
        with st.spinner("🧠 Running inference..."):
            mc_r  = predict_multiclass(mc_model, img, device)
            bin_r = predict_binary(bin_model, img, device)
            hmap  = get_gradcam_heatmap(mc_model, img, device) if show_gradcam else None

        if mc_r["confidence"] < conf_threshold:
            st.warning(f"⚠️ Low confidence ({mc_r['confidence']:.1%}) — model is uncertain.")

        st.markdown(f"""
        <div class='glass-card'>
          <div class='section-label'>🎯 Prediction Results</div>
          <div style='display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1rem;'>
            <div>
              <div style='font-size:.65rem;color:#64748b;font-family:Space Mono,monospace;text-transform:uppercase;letter-spacing:.1em;margin-bottom:5px;'>Multi-Class</div>
              <span class='badge-mc'>{mc_r["emoji"]} {mc_r["class"].upper()}</span>
            </div>
            <div>
              <div style='font-size:.65rem;color:#64748b;font-family:Space Mono,monospace;text-transform:uppercase;letter-spacing:.1em;margin-bottom:5px;'>Binary (Airplane?)</div>
              <span class='{"badge-pos" if bin_r["is_airplane"] else "badge-neg"}'>
                {"✅ YES" if bin_r["is_airplane"] else "❌ NO"} · {bin_r["confidence"]:.1%}
              </span>
            </div>
          </div>
          <div class='metric-row'>
            <div class='metric-card'><div class='metric-value'>{mc_r["confidence"]:.1%}</div><div class='metric-label'>MC Confidence</div></div>
            <div class='metric-card'><div class='metric-value'>{mc_r["class_index"]}</div><div class='metric-label'>Class ID</div></div>
            <div class='metric-card'><div class='metric-value'>{bin_r["airplane_prob"]:.2f}</div><div class='metric-label'>Airplane Prob</div></div>
          </div>
        </div>""", unsafe_allow_html=True)

        tab_p, tab_g, tab_r = st.tabs(["📊 Class Probabilities", "🔥 Grad-CAM", "🔢 Raw Scores"])

        with tab_p:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            probs = mc_r["probabilities"]
            sorted_idx = np.argsort(probs)[::-1]
            for rank, idx in enumerate(sorted_idx if show_all_probs else sorted_idx[:5]):
                pct = probs[idx]
                tier = "top" if rank == 0 else ("mid" if rank < 3 else "low")
                st.markdown(f"""
                <div style='margin:.3rem 0;'>
                  <div class='conf-label'><span class='cls'>{CLASS_EMOJIS[idx]} {CLASS_NAMES[idx]}</span><span>{pct:.2%}</span></div>
                  <div class='conf-bar-bg'><div class='conf-bar-fill {tier}' style='width:{pct*100:.1f}%'></div></div>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with tab_g:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            if hmap is not None:
                overlay = overlay_gradcam(img, hmap, alpha=0.45)
                c1, c2, c3 = st.columns(3)
                with c1: st.image(img.resize((128,128), Image.NEAREST), caption="Original", use_container_width=True)
                with c2:
                    import matplotlib.cm as cm
                    hmap_img = Image.fromarray((cm.get_cmap("jet")(hmap)[:,:,:3]*255).astype(np.uint8)).resize((128,128), Image.NEAREST)
                    st.image(hmap_img, caption="Heatmap", use_container_width=True)
                with c3: st.image(overlay.resize((128,128), Image.LANCZOS), caption="Overlay", use_container_width=True)
                st.markdown("<div style='color:#64748b;font-size:.72rem;font-family:Space Mono,monospace;margin-top:.5rem;'>🔥 Red = regions the model focused on.</div>", unsafe_allow_html=True)
            else:
                st.info("Enable Grad-CAM in sidebar settings.")
            st.markdown("</div>", unsafe_allow_html=True)

        with tab_r:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            import pandas as pd
            df = pd.DataFrame({
                "Class": [f"{CLASS_EMOJIS[i]} {CLASS_NAMES[i]}" for i in range(10)],
                "Probability": [f"{p:.6f}" for p in probs],
                "Pct": [f"{p*100:.3f}%" for p in probs],
            }).sort_values("Probability", ascending=False).reset_index(drop=True)
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)


st.markdown("""
<div style='text-align:center;padding:2rem 0 1rem;color:#1f2937;
  font-family:Space Mono,monospace;font-size:.65rem;letter-spacing:.1em;'>
  CIFAR-10 NEURAL VISION · PYTORCH RESNET · DUAL CLASSIFIER · GRAD-CAM
</div>""", unsafe_allow_html=True)
