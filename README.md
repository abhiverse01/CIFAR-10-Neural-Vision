# 🧠 CIFAR-10 Neural Vision

> **Rewrite** of the classic CIFAR-10 classifier.  
> ResNet-style CNN · Dual Binary + Multi-Class · Grad-CAM Explainability · Streamlit UI

---

## ✨ What's New vs. the Original

| Feature | Original Flask App | This Rewrite |
|---|---|---|
| Architecture | Simple Conv stack | **ResNet residual blocks** |
| Data Augmentation | None | **Flip / Rotate / Zoom / Contrast** |
| Optimizer | Adam | **AdamW + Cosine LR Annealing** |
| Regularization | Dropout only | **L2 + Dropout + Label Smoothing** |
| Explainability | None | **Grad-CAM heatmaps** |
| UI | Bootstrap HTML | **Streamlit dark-mode dashboard** |
| Results display | Text only | **Confidence bars + probability table** |
| Image preview | None | **Upload preview + metadata** |
| Deployment | Flask (manual) | **Streamlit Cloud (1-click)** |
| Model format | `.h5` (legacy) | **`.keras` (modern) + `.h5` fallback** |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <your-repo>
cd cifar10_streamlit
pip install -r requirements.txt
```

### 2. Train Models

```bash
# Trains both models and saves to models/ directory
python models/model_builder.py
```

This will:
- Download CIFAR-10 automatically via Keras
- Train the **ResNet multi-class** model (~85% val acc with 30 epochs)
- Train the **ResNet binary** model (airplane vs rest, ~97% val acc)
- Save `models/model_multiclass.keras` and `models/model_binary.keras`

> **Tip:** Training takes ~30 min on CPU, ~5 min on GPU.  
> Use `epochs=10` in `train_models()` for a quick test run.

### 3. Launch the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🌐 Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → select `app.py`
4. Add `requirements.txt` — Streamlit Cloud handles the rest

> **Note:** Pre-train your models locally and commit the `.keras` files to the repo  
> (or use Git LFS for large model files).

---

## 📁 Project Structure

```
cifar10_streamlit/
├── app.py                    # Streamlit main app
├── requirements.txt
├── README.md
├── models/
│   ├── model_builder.py      # ResNet architecture + training
│   ├── model_multiclass.keras   # (generated after training)
│   └── model_binary.keras       # (generated after training)
└── utils/
    └── predictor.py          # Preprocessing, inference, Grad-CAM
```

---

## 🏗️ Architecture Highlights

### ResNet-style Backbone
- 4 residual stages: 64 → 128 → 256 → 512 filters
- Each stage doubles filters, halves spatial dims (stride-2 Conv)
- Skip connections with 1×1 projection when dims change
- No FC bottleneck — GlobalAveragePooling → head

### Training Pipeline
- **AdamW** with weight decay for better generalization
- **Cosine annealing** LR schedule prevents plateau stagnation
- **Label smoothing** (ε=0.1) reduces overconfidence
- **Early stopping** restores best weights

### Grad-CAM
Computes gradient of the predicted class score with respect to  
the last Conv layer's feature maps, producing a spatial heatmap  
showing *where* the model looked to make its decision.

---

## 🎯 Expected Performance

| Model | Val Accuracy | Notes |
|---|---|---|
| Multi-class (10 classes) | ~85-88% | With 30 epochs + augmentation |
| Binary (airplane vs rest) | ~96-98% | Easier task, fewer epochs needed |

---

## 🧪 Test Images

The model works best with:
- Clear, single-subject images
- Objects similar to CIFAR-10 categories
- Any resolution (auto-resized to 32×32 internally)

CIFAR-10 classes: **airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck**
