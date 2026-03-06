"""
model_builder.py  —  PyTorch rewrite
ResNet-style CNN for CIFAR-10, dual binary + multi-class.
Run: python models/model_builder.py
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
CLASS_EMOJIS = ["✈️", "🚗", "🐦", "🐱", "🦌", "🐶", "🐸", "🐴", "🚢", "🚚"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class MultiClassCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(ResBlock(64, 64), ResBlock(64, 64))
        self.layer2 = nn.Sequential(ResBlock(64, 128, 2), ResBlock(128, 128))
        self.layer3 = nn.Sequential(ResBlock(128, 256, 2), ResBlock(256, 256))
        self.layer4 = nn.Sequential(ResBlock(256, 512, 2), ResBlock(512, 512))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.4), nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x))))))


class BinaryCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResBlock(64, 64),
            ResBlock(64, 128, 2),
            ResBlock(128, 256, 2),
            ResBlock(256, 512, 2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.4), nn.Linear(512, 128),
            nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_models(epochs=30, batch_size=128, save_dir="models/"):
    os.makedirs(save_dir, exist_ok=True)
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_ds = torchvision.datasets.CIFAR10("data/", train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10("data/", train=False, download=True, transform=test_tf)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    def run(model, criterion, name, label_fn=None):
        model = model.to(DEVICE)
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        best_acc, best_state = 0.0, None
        for ep in range(epochs):
            model.train()
            for imgs, labels in train_dl:
                imgs = imgs.to(DEVICE)
                tgt = label_fn(labels).to(DEVICE) if label_fn else labels.to(DEVICE)
                opt.zero_grad()
                criterion(model(imgs).squeeze(), tgt).backward()
                opt.step()
            sched.step()
            model.eval(); correct = total = 0
            with torch.no_grad():
                for imgs, labels in test_dl:
                    imgs = imgs.to(DEVICE)
                    tgt = label_fn(labels).to(DEVICE) if label_fn else labels.to(DEVICE)
                    out = model(imgs)
                    if label_fn:
                        correct += ((torch.sigmoid(out.squeeze()) > 0.5).long() == tgt.long()).sum().item()
                    else:
                        correct += (out.argmax(1) == tgt).sum().item()
                    total += tgt.size(0)
            acc = correct / total
            print(f"  [{name}] Ep {ep+1}/{epochs} val_acc={acc:.4f}")
            if acc > best_acc:
                best_acc, best_state = acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_{name}.pt"))
        print(f"  Saved model_{name}.pt  (best={best_acc:.4f})")

    print("🚀 Multi-Class..."); run(MultiClassCNN(), nn.CrossEntropyLoss(label_smoothing=0.1), "multiclass")
    print("🚀 Binary...");      run(BinaryCNN(), nn.BCEWithLogitsLoss(), "binary", lambda y: (y==0).float())


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    train_models()
