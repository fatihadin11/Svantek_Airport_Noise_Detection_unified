"""
╔══════════════════════════════════════════════════════════════╗
║              Train CNN  —  5 Sınıf                          ║
║   Mel Spectrogram + SpecAugment + AirportCNN                ║
╚══════════════════════════════════════════════════════════════╝

Sıralama:
  1. python dataset_builder_v3.py     → manifest_v3.csv üretir
  2. python train_cnn.py              → best_cnn.pt üretir

Gereksinimler:
  pip install torch torchaudio librosa scikit-learn imbalanced-learn
              joblib numpy pandas matplotlib seaborn tqdm

Not: GPU yoksa CPU'da ~5-10 dk/epoch → 50 epoch ≈ 4-8 saat.
     Colab veya GPU önerilir (cuda otomatik algılanır).
"""

import os
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio.transforms as T

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ================================================================
# ⚙️  AYARLAR
# ================================================================

PROJECT_ROOT = r"C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise"
MANIFEST_CSV = os.path.join(PROJECT_ROOT, "cache",  "manifest_v3.csv")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "outputs", "training_cnn")

# Mel Spectrogram parametreleri
SR        = 22050
N_MELS    = 128
N_FFT     = 2048
HOP_FFT   = 512
DURATION  = 5.0          # saniye — çıkan zaman frame: ⌊SR*DURATION/HOP_FFT⌋+1 ≈ 216

# Eğitim hiper-parametreleri
EPOCHS        = 50
PATIENCE      = 10        # early stopping
BATCH_SIZE    = 32
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
TEST_SIZE     = 0.15
VAL_SIZE      = 0.15      # eğitim içinden validation ayrılır
RANDOM_SEED   = 42

# noise_detector.py ile senkronize
MANUAL_CLASS_WEIGHTS = {
    "AIRCRAFT": 0.5,
    "SPEECH":   2.0,
    "TRAFFIC":  2.5,
    "WIND":     2.5,
    "AMBIENT":  2.0,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# 🔊  DATASET
# ================================================================

class SpecAugment(nn.Module):
    """
    Eğitim sırasında Mel spektrogramı maskele (Park et al. 2019).
    time_mask:  zaman ekseninde rastgele T frame sıfırla
    freq_mask:  frekans ekseninde rastgele F bin sıfırla
    """
    def __init__(self, time_mask: int = 30, freq_mask: int = 15,
                 n_time: int = 2, n_freq: int = 2):
        super().__init__()
        self.time_masks = nn.ModuleList([
            T.TimeMasking(time_mask_param=time_mask) for _ in range(n_time)
        ])
        self.freq_masks = nn.ModuleList([
            T.FrequencyMasking(freq_mask_param=freq_mask) for _ in range(n_freq)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.time_masks:
            x = m(x)
        for m in self.freq_masks:
            x = m(x)
        return x


class MelSpectrogramDataset(Dataset):
    """
    CSV manifest'ten ses dosyası yükler, Mel spektrograma dönüştürür.
    manifest sütunları: path, label  (label_enc train sırasında eklenir)
    """

    def __init__(self, records: list[dict], augment: bool = False):
        self.records    = records
        self.augment    = augment
        self.target_len = int(SR * DURATION)

        # torchaudio transform'ları — CPU'da çalışır, GPU'ya taşımaya gerek yok
        self.mel_tf    = T.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT,
            hop_length=HOP_FFT, n_mels=N_MELS
        )
        self.db_tf     = T.AmplitudeToDB(top_db=80)
        self.spec_aug  = SpecAugment() if augment else None

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]

        # Ses yükle — librosa center-trim veya zero-pad yapar
        try:
            y, _ = librosa.load(rec["path"], sr=SR, mono=True,
                                 duration=DURATION + 0.5)
        except Exception as e:
            # Bozuk dosya: sıfır sinyal
            y = np.zeros(self.target_len, dtype=np.float32)

        # Sabit uzunluk
        if len(y) >= self.target_len:
            start = (len(y) - self.target_len) // 2
            y = y[start: start + self.target_len]
        else:
            y = np.pad(y, (0, self.target_len - len(y)))

        y_tensor = torch.FloatTensor(y).unsqueeze(0)          # (1, samples)
        spec     = self.mel_tf(y_tensor)                       # (1, N_MELS, frames)
        spec     = self.db_tf(spec)                            # dB normalize

        if self.augment and self.spec_aug is not None:
            spec = self.spec_aug(spec)

        return spec, rec["label_enc"]


# ================================================================
# 🏗️  MODEL
# ================================================================

class AirportCNN(nn.Module):
    """
    3 Conv bloğu + AdaptiveAvgPool → Classifier
    Giriş: (B, 1, N_MELS, T) = (B, 1, 128, ~216)
    Çıkış: (B, n_classes)
    """

    def __init__(self, n_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            # Blok 1 — (B, 1, 128, 216) → (B, 32, 64, 108)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # Blok 2 — (B, 32, 64, 108) → (B, 64, 32, 54)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.1),

            # Blok 3 — (B, 64, 32, 54) → (B, 128, 4, 4)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),   # boyutu sabitler
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ================================================================
# 🏋  EĞİTİM DÖNGÜSÜ
# ================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(specs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device)
        out  = model(specs)
        loss = criterion(out, labels)
        total_loss += loss.item() * len(labels)
        preds       = out.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += len(labels)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return total_loss / total, correct / total, all_preds, all_labels


def plot_training_curves(history: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(True)

    plt.suptitle("CNN Eğitim Eğrileri", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(save_dir, "training_curves_cnn.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  → {out}")


def plot_confusion_matrix(y_true, y_pred, label_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    cm     = confusion_matrix(y_true, y_pred, normalize="true")
    cm_raw = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, title in zip(axes,
                                [cm_raw, cm],
                                ["Ham Sayılar", "Normalize (satır %)"]):
        fmt = "d" if data is cm_raw else ".2f"
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=label_names, yticklabels=label_names, ax=ax)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("Confusion Matrix — AirportCNN", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(save_dir, "confusion_matrix_cnn.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  → {out}")


# ================================================================
# 🏃  ANA PIPELINE
# ================================================================

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,  exist_ok=True)

    print("=" * 56)
    print("  Train CNN  —  Mel Spectrogram + AirportCNN")
    print(f"  Device: {device}")
    print("=" * 56)

    # ── 1. Manifest yükle ────────────────────────────────────────
    if not os.path.exists(MANIFEST_CSV):
        print(f"[HATA] Manifest bulunamadı: {MANIFEST_CSV}")
        print(f"       Önce çalıştır: python dataset_builder_v3.py")
        raise SystemExit(1)

    df = pd.read_csv(MANIFEST_CSV)
    # Var olan dosyaları filtrele
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
    print(f"\n  Toplam örnek: {len(df)}")

    dist = Counter(df["label"])
    print("\n── Sınıf Dağılımı ─────────────────────────────────")
    for lbl, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        bar = "█" * (cnt // 20)
        print(f"  {lbl:10s} {cnt:5d}  {bar}")

    # ── 2. Label encode ──────────────────────────────────────────
    le     = LabelEncoder()
    df["label_enc"] = le.fit_transform(df["label"])
    labels = list(le.classes_)
    print(f"\n  Sınıflar: {labels}")

    # ── 3. Eğitim / Test bölünmesi ───────────────────────────────
    train_val_df, test_df = train_test_split(
        df, test_size=TEST_SIZE,
        stratify=df["label_enc"], random_state=RANDOM_SEED
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=VAL_SIZE / (1 - TEST_SIZE),
        stratify=train_val_df["label_enc"], random_state=RANDOM_SEED
    )
    print(f"  Eğitim: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")

    # ── 4. Dataset & DataLoader ──────────────────────────────────
    train_records = train_df.to_dict("records")
    val_records   = val_df.to_dict("records")
    test_records  = test_df.to_dict("records")

    train_ds = MelSpectrogramDataset(train_records, augment=True)
    val_ds   = MelSpectrogramDataset(val_records,   augment=False)
    test_ds  = MelSpectrogramDataset(test_records,  augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=False)

    # ── 5. Model, loss, optimizer ────────────────────────────────
    model = AirportCNN(n_classes=len(labels)).to(device)
    print(f"\n  Parametre sayısı: {sum(p.numel() for p in model.parameters()):,}")

    weights = torch.FloatTensor([
        MANUAL_CLASS_WEIGHTS[labels[i]] for i in range(len(labels))
    ]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )

    # ── 6. Eğitim döngüsü ────────────────────────────────────────
    print("\n" + "=" * 56)
    print("  EĞİTİM BAŞLADI")
    print("=" * 56)
    print(f"  {'Epoch':>5}  {'TrLoss':>8}  {'TrAcc':>7}  "
          f"{'VaLoss':>8}  {'VaAcc':>7}  {'F1':>7}")
    print("  " + "─" * 52)

    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   [], "val_f1": []}

    best_val_f1  = -1.0
    patience_cnt = 0
    best_model_path = os.path.join(MODELS_DIR, "best_cnn.pt")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, va_preds, va_true = evaluate(
            model, val_loader, criterion, device)

        va_f1 = f1_score(va_true, va_preds, average="macro", zero_division=0)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_f1"].append(va_f1)

        elapsed = time.time() - t0
        print(f"  {epoch:5d}  {tr_loss:8.4f}  {tr_acc:7.4f}  "
              f"{va_loss:8.4f}  {va_acc:7.4f}  {va_f1:7.4f}  ({elapsed:.0f}s)")

        # Early stopping + en iyi model kaydı
        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            patience_cnt = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "label_names": labels,
                "n_classes":   len(labels),
                "val_f1":      best_val_f1,
            }, best_model_path)
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"\n  ⏹  Early stopping (patience={PATIENCE})")
                break

    # ── 7. Test değerlendirme ─────────────────────────────────────
    print("\n" + "=" * 56)
    print("  EN İYİ MODEL YÜKLENİYOR...")
    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Epoch {ckpt['epoch']}  |  Val F1: {ckpt['val_f1']:.4f}")

    print("\n  TEST SONUÇLARI")
    print("=" * 56)
    _, te_acc, te_preds, te_true = evaluate(model, test_loader, criterion, device)
    te_f1_mac = f1_score(te_true, te_preds, average="macro")
    te_f1_wt  = f1_score(te_true, te_preds, average="weighted")

    print(f"\n  Accuracy   : {te_acc:.4f}  ({te_acc:.1%})")
    print(f"  F1 Macro   : {te_f1_mac:.4f}   ← ASIL METRİK")
    print(f"  F1 Weighted: {te_f1_wt:.4f}")
    print(f"\n{classification_report(te_true, te_preds, target_names=labels, digits=3)}")

    plot_training_curves(history, PLOTS_DIR)
    plot_confusion_matrix(te_true, te_preds, labels, PLOTS_DIR)

    # ── 8. Metadata kaydet ───────────────────────────────────────
    # LabelEncoder'ı SVM ile aynı, ama CNN için ayrı yedek
    joblib.dump(le, os.path.join(MODELS_DIR, "cnn_label_encoder.pkl"))
    joblib.dump({
        "model_name":    "AirportCNN",
        "test_accuracy": float(te_acc),
        "f1_macro":      float(te_f1_mac),
        "label_names":   labels,
        "sr":            SR,
        "n_mels":        N_MELS,
        "n_fft":         N_FFT,
        "hop_fft":       HOP_FFT,
        "duration":      DURATION,
        "best_epoch":    int(ckpt["epoch"]),
        "version":       "v1",
    }, os.path.join(MODELS_DIR, "cnn_meta.pkl"))

    print("\n" + "=" * 56)
    print("  🎉  CNN EĞİTİMİ TAMAMLANDI")
    print(f"     Model    : AirportCNN")
    print(f"     F1 Macro : {te_f1_mac:.1%}")
    print(f"     Epoch    : {ckpt['epoch']}")
    print(f"     Kayıt    : {best_model_path}")
    print("=" * 56)


if __name__ == "__main__":
    main()
