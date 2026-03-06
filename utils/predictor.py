"""
predictor.py  —  PyTorch inference utilities + Grad-CAM
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
CLASS_EMOJIS = ["✈️", "🚗", "🐦", "🐱", "🦌", "🐶", "🐸", "🐴", "🚢", "🚚"]

# CIFAR-10 normalisation stats
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)


def preprocess(image: Image.Image) -> torch.Tensor:
    """PIL image → normalised (1,3,32,32) tensor."""
    img = image.convert("RGB").resize((32, 32), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - np.array(MEAN)) / np.array(STD)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


def predict_multiclass(model, image: Image.Image, device="cpu") -> dict:
    model.eval()
    tensor = preprocess(image).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    idx = int(np.argmax(probs))
    return {
        "class":       CLASS_NAMES[idx],
        "emoji":       CLASS_EMOJIS[idx],
        "confidence":  float(probs[idx]),
        "probabilities": probs.tolist(),
        "class_index": idx,
    }


def predict_binary(model, image: Image.Image, device="cpu") -> dict:
    model.eval()
    tensor = preprocess(image).to(device)
    with torch.no_grad():
        logit = model(tensor).squeeze()
        airplane_prob = float(torch.sigmoid(logit).item())
    is_airplane = airplane_prob > 0.5
    return {
        "label":        "Airplane" if is_airplane else "Non-Airplane",
        "is_airplane":  is_airplane,
        "confidence":   airplane_prob if is_airplane else 1.0 - airplane_prob,
        "airplane_prob": airplane_prob,
    }


# ─── Grad-CAM ──────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients   = None
        target_layer.register_forward_hook(self._save_act)
        target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, _, __, output):
        self.activations = output.detach()

    def _save_grad(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, tensor, class_idx=None):
        self.model.eval()
        logits = self.model(tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        self.model.zero_grad()
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def get_gradcam_heatmap(model, image: Image.Image, device="cpu"):
    """Returns normalised (32,32) heatmap or None."""
    # Find last Conv2d layer
    target_layer = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            target_layer = m

    if target_layer is None:
        return None

    try:
        gc = GradCAM(model, target_layer)
        tensor = preprocess(image).to(device).requires_grad_(True)
        heatmap = gc(tensor)

        # Resize to 32×32
        hmap = Image.fromarray((heatmap * 255).astype(np.uint8))
        hmap = hmap.resize((32, 32), Image.LANCZOS)
        return np.array(hmap) / 255.0
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None


def overlay_gradcam(original: Image.Image, heatmap: np.ndarray, alpha=0.5) -> Image.Image:
    import matplotlib.cm as cm
    colored = (cm.get_cmap("jet")(heatmap)[:, :, :3] * 255).astype(np.uint8)
    overlay = Image.fromarray(colored).resize(original.size, Image.LANCZOS)
    return Image.blend(original.convert("RGB"), overlay, alpha=alpha)
