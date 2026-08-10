"""
╔═══════════════════════════════════════════════════════════════╗
║  train_beats.py  v2  —  BEATs MLP  |  Group-Split + Augment ║
╚═══════════════════════════════════════════════════════════════╝

v2 Değişiklikleri:
  ✦ Group-aware split  →  ESC-50 A/B/C varyantları aynı split'e düşer,
                          data leakage kapatıldı, metrikler artık güvenilir
  ✦ Waveform augmentation  →  Gaussian noise + gain + pitch shift
                               Eğitim setine N_AUGMENTS kat sentetik veri
                               Domain shift direnci artar

Veri kaynağı (v6 — taksonomi + kaynak yenilendi):
  manifest_v6.csv  →  dataset_builder.py çıktısı:
                       airport-audio-collector SQLite pipeline'ı (kabul
                       edilmiş örnekler) + onaylı live klipler.
                       Eski ESC-50/AeroSonicDB/GENERIC_AUDIO veri seti
                       tamamen iptal edildi — artık kullanılmıyor.
  + Svantek_Recordings  →  CSV'siz, doğrudan klasör taraması (aşağıda)

Aşamalar:
  1. BEATs encoder yükle  (frozen)
  2. BASE cache: tüm dosyalar → 768-dim embedding  (bir kez)
  3. Group-aware split   (aynı kaynaktan gelen dosyalar leakage-safe gruplanır)
  4. AUG cache: eğitim seti × N_AUGMENTS augmented embedding  (bir kez)
  5. MLP  (768 → 256 → 10)  eğit
  6. D:\\models\\beats_mlp.pt  kaydet

Cache:
  D:\\models\\beats_embed_cache.pkl   — base embeddingler (tüm veri, temiz)
  D:\\models\\beats_aug_cache.pkl     — augmented embeddingler (sadece train)
  Manifest / train seti değişirse cache otomatik yenilenir.
  Zorla yenilemek:  python train_beats.py --rebuild-cache

Checkpoint uyumluluğu — noise_detector.py:
  beats.mlp.load_state_dict(ckpt["model_state"])
  ckpt["model_state"] = model.net.state_dict()
"""

import os
import re
import sys
import time
import pickle
import hashlib
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import joblib
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, recall_score)

warnings.filterwarnings("ignore")

# ── Sınıf tanımları — TEK kaynak class_config.py ────────────────
from class_config import CLASSES, TRAINING_CLASS_WEIGHTS
from audio_chunking import decode_chunk_path, chunk_path_exists, chunk_source_file, extract_source_id

# ── BEATs modülü ───────────────────────────────────────────────
try:
    from BEATs import BEATs, BEATsConfig
    BEATS_OK = True
except ImportError:
    print("[HATA] BEATs modülü bulunamadı.")
    print("       github.com/microsoft/unilm  →  beats/ klasörünü proje köküne kopyalayın.")
    sys.exit(1)

# ── tqdm (opsiyonel) ───────────────────────────────────────────
try:
    from tqdm import tqdm
    TQDM_OK = True
except ImportError:
    TQDM_OK = False


# ================================================================
# ⚙️  AYARLAR
# ================================================================

PROJECT_ROOT  = r"C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise"
MODELS_DIR    = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR     = os.path.join(PROJECT_ROOT, "outputs", "training_beats")

BEATS_ENCODER = r"D:\models\BEATs_iter3_plus_AS2M.pt"
BEATS_MLP_OUT = r"D:\models\beats_mlp.pt"
EMBED_CACHE   = r"D:\models\beats_embed_cache.pkl"
AUG_CACHE     = r"D:\models\beats_aug_cache.pkl"
SVANTEK_DIR   = r"D:\Svantek_Recordings"          # CSV gerektirmeden taranır

# manifest — dataset_builder.py'nin tek, birleşik çıktısı
# (SQLite collector verisi + onaylı live klipler, artık v4/v5 gibi
# ayrı "temel/genişletilmiş" varyant yok — eski veri seti tamamen iptal)
MANIFEST_CSV = os.path.join(PROJECT_ROOT, "cache", "manifest_v6.csv")

# ── Ses parametreleri — noise_detector.py ile AYNI ─────────────
SR         = 22050
DURATION   = 5.0
TARGET_LEN = int(SR * DURATION)   # 110 250 örnek

# ── BEATs sabitleri — class_config.py ile AYNI (TEK kaynak orada) ──
BEATS_CLASSES = CLASSES
EMBED_DIM     = 768
N_CLASSES     = len(BEATS_CLASSES)

# ── Embedding çıkarım batch boyutu ────────────────────────────
# GPU ≥ 8GB → 64  |  GPU 4GB → 32  |  CPU → 16
EMBED_BATCH = 32

# ── Augmentation parametreleri ────────────────────────────────
N_AUGMENTS     = 2      # Her eğitim klibinin kaç augmented versiyonu
AUG_NOISE_PROB = 0.65   # Gaussian noise uygulama olasılığı
AUG_GAIN_PROB  = 0.50   # Random gain (±4 dB) olasılığı
AUG_PITCH_PROB = 0.35   # Pitch shift (±1.5 st) olasılığı — yavaş, 0.0 → devre dışı

# ── MLP eğitim hiper-parametreleri ───────────────────────────
EPOCHS       = 50
BATCH_SIZE   = 256
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 12

# ── Veri bölünmesi ────────────────────────────────────────────
TEST_SIZE   = 0.15
VAL_SIZE    = 0.15
RANDOM_SEED = 42

# ── Sınıf ağırlıkları — class_config.py ile AYNI (TEK kaynak orada),
#    train_efficientnet.py de aynı sözlüğü kullanır.
# Yeni taksonomi için henüz tune EDİLMEDİ — hepsi nötr (1.0). İlk eğitim
# sonrası per-class recall'a bakıp class_config.TRAINING_CLASS_WEIGHTS'i
# tune et (özellikle az örnekli sınıflarda, ör. APU_GSE, PRECIPITATION).
MANUAL_CLASS_WEIGHTS = TRAINING_CLASS_WEIGHTS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# 🏷️  GROUP-AWARE SPLIT — SOURCE ID ÇIKARIM
# ================================================================

# ================================================================
# 🏷️  GROUP-AWARE SPLIT — SOURCE ID
# ================================================================
# extract_source_id artık audio_chunking.py'de (train_efficientnet.py da
# aynısını kullanıyor) — burada elle kopyalanmıyor, yukarıda import edildi.


# ================================================================
# 📂  SVANTEK KAYIT TARAYICI
# ================================================================

def scan_svantek_recordings(svantek_dir: str = SVANTEK_DIR) -> pd.DataFrame:
    """
    D:\\Svantek_Recordings\\ altındaki alt klasörleri tarar.
    Alt klasör adı (büyük harf) sınıf etiketi olur.
    Yalnızca BEATS_CLASSES içindeki klasörler dahil edilir.
    CSV gerekmez — doğrudan manifest ile birleştirilir.

    ⚠ ÖNEMLİ — taksonomi değişti: BEATS_CLASSES artık eski 6 sınıf değil,
    yeni 10 sınıf (JET_AIRCRAFT, HELICOPTER, APU_GSE, ...). D: sürücünüzdeki
    Svantek_Recordings altındaki klasörler hâlâ eski isimlerle
    (Aircraft/Speech/Wind/...) ise BEATS_CLASSES ile eşleşmediği için
    HEPSİ atlanacak (aşağıdaki "atlanıyor" uyarısıyla, hata vermez).
    Bu veriyi kullanmaya devam etmek istiyorsan klasörleri yeni sınıf
    isimleriyle yeniden adlandırman/organize etmen gerekir — bu dosya
    sürücünüze erişemediğim için bunu sizin yapmanız gerekiyor.

    Beklenen yapı (yeni isimlerle):
        D:\\Svantek_Recordings\\OTHER\\Event651.wav
        D:\\Svantek_Recordings\\SPEECH\\Event481.wav

    Döndürür: ["path", "label"] sütunlu DataFrame
    """
    if not os.path.isdir(svantek_dir):
        print(f"  [Svantek] Dizin bulunamadı: {svantek_dir} — atlanıyor")
        return pd.DataFrame(columns=["path", "label"])

    rows = []
    for subfolder in sorted(os.listdir(svantek_dir)):
        label = subfolder.strip().upper()
        if label not in BEATS_CLASSES:
            print(f"  [Svantek] '{subfolder}' → BEATS_CLASSES dışında, atlanıyor")
            continue
        folder_path = os.path.join(svantek_dir, subfolder)
        if not os.path.isdir(folder_path):
            continue
        wav_files = sorted(
            f for f in os.listdir(folder_path) if f.lower().endswith(".wav")
        )
        for fname in wav_files:
            full_path = os.path.join(folder_path, fname)
            for cp in chunk_source_file(full_path):
                rows.append({"path": cp, "label": label})

    df_sv = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["path", "label"])

    if df_sv.empty:
        print("  [Svantek] Hiç .wav dosyası bulunamadı.")
    else:
        print(f"\n── Svantek Kayıtları ({svantek_dir}) ───────────────────────")
        for lbl, grp in df_sv.groupby("label"):
            print(f"  {lbl:10s} {len(grp):5d}  {'█' * min(len(grp), 40)}")
        print(f"  Toplam  : {len(df_sv)}")
    return df_sv


# ================================================================
# 🎵  WAVEFORM AUGMENTATION
# ================================================================

def augment_waveform(y: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Ham waveform'a tekrarlanabilir rastgele augmentasyon uygula.

    Uygulanan dönüşümler (olasılıksal):
      • Gaussian noise  (SNR 10–25 dB)  — mikrofon gürültüsü simülasyonu
      • Random gain     (±4 dB)         — kayıt seviyesi varyasyonu
      • Pitch shift     (±1.5 yarı ton) — ses tonu kayması

    seed: aynı seed → aynı augmentasyon (cache uyumluluğu için)
    """
    rng = np.random.RandomState(seed)
    y   = y.copy().astype(np.float32)

    # Gaussian noise
    if rng.random() < AUG_NOISE_PROB:
        rms     = float(np.sqrt(np.mean(y ** 2))) + 1e-9
        snr_db  = rng.uniform(10, 25)
        n_rms   = rms / (10 ** (snr_db / 20))
        y      += rng.normal(0, n_rms, len(y)).astype(np.float32)

    # Random gain
    if rng.random() < AUG_GAIN_PROB:
        gain_db = rng.uniform(-4, 4)
        y      *= float(10 ** (gain_db / 20))

    # Pitch shift (yavaş — devre dışı için AUG_PITCH_PROB = 0.0)
    if AUG_PITCH_PROB > 0 and rng.random() < AUG_PITCH_PROB:
        n_steps = float(rng.uniform(-1.5, 1.5))
        try:
            y = librosa.effects.pitch_shift(y, sr=SR, n_steps=n_steps)
        except Exception:
            pass

    return np.clip(y, -1.0, 1.0).astype(np.float32)


# ================================================================
# 🎧  BEATs ENCODER & EMBEDDİNG ÇIKARIM
# ================================================================

def load_beats_encoder() -> "BEATs":
    """BEATs encoder yükle ve dondur. Config checkpoint'ten okunur."""
    if not os.path.exists(BEATS_ENCODER):
        print(f"[HATA] Encoder bulunamadı: {BEATS_ENCODER}")
        sys.exit(1)

    print(f"  Encoder yükleniyor: {BEATS_ENCODER}")
    ckpt  = torch.load(BEATS_ENCODER, map_location=device, weights_only=False)
    cfg   = BEATsConfig(ckpt["cfg"])
    enc   = BEATs(cfg)
    enc.load_state_dict(ckpt["model"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    enc = enc.to(device)

    total = sum(p.numel() for p in enc.parameters())
    print(f"  Encoder hazır  |  {total:,} parametre  (tamamı donduruldu)")
    return enc


def load_audio_clip(path: str) -> np.ndarray:
    """22050 Hz mono yükle, 5s merkez-kırp / sıfır-doldur.
    path 'gerçek_yol::start_sec' biçiminde kodlanmış olabilir (uzun
    dosyalardan çoklu chunk için, bkz. audio_chunking.py) — çözülüp
    doğru zaman penceresinden okunur."""
    real_path, start_sec = decode_chunk_path(path)
    try:
        y, _ = librosa.load(real_path, sr=SR, mono=True,
                            offset=start_sec, duration=DURATION + 0.5)
    except Exception:
        return np.zeros(TARGET_LEN, dtype=np.float32)

    if len(y) >= TARGET_LEN:
        start = (len(y) - TARGET_LEN) // 2
        y = y[start: start + TARGET_LEN]
    else:
        y = np.pad(y, (0, TARGET_LEN - len(y)))
    return y.astype(np.float32)


@torch.no_grad()
def _embed_batch(encoder: "BEATs", waveforms: torch.Tensor) -> np.ndarray:
    """(B, T) waveform → (B, 768) embedding  —  temporal mean-pool."""
    waveforms = waveforms.to(device)
    pad_mask  = torch.zeros(waveforms.shape[0], waveforms.shape[1],
                             dtype=torch.bool, device=device)
    features, _ = encoder.extract_features(waveforms, padding_mask=pad_mask)
    return features.mean(dim=1).cpu().numpy()


def extract_all_embeddings(paths: list, encoder: "BEATs") -> np.ndarray:
    """Tüm dosyalar için temiz embedding çıkar. (N, 768) döndürür."""
    n, results, t0 = len(paths), [], time.time()
    print(f"\n  {n} dosyadan BASE embedding  (batch={EMBED_BATCH}, device={device})")

    batches  = range(0, n, EMBED_BATCH)
    iterator = (tqdm(batches, desc="  BASE", unit="batch",
                     total=(n + EMBED_BATCH - 1) // EMBED_BATCH)
                if TQDM_OK else batches)

    for bi, i in enumerate(iterator):
        waves = np.stack([load_audio_clip(p) for p in paths[i: i + EMBED_BATCH]])
        results.append(_embed_batch(encoder, torch.FloatTensor(waves)))
        if not TQDM_OK and bi % 50 == 0 and bi > 0:
            done = min(i + EMBED_BATCH, n)
            eta  = (time.time() - t0) / done * (n - done)
            print(f"  [{done}/{n}]  ETA ≈ {eta/60:.0f} dk")

    arr = np.vstack(results).astype(np.float32)
    print(f"  BASE tamamlandı  |  {(time.time()-t0)/60:.1f} dk  |  {arr.shape}")
    return arr


def extract_aug_embeddings(train_paths: list, train_labels_enc: np.ndarray,
                            encoder: "BEATs") -> tuple:
    """
    Eğitim seti × N_AUGMENTS augmented embedding çıkar.
    Her (path, aug_idx) çifti için deterministik seed kullanılır.
    Döndürür: (aug_embeddings (N×N_AUGMENTS, 768), aug_labels (N×N_AUGMENTS,))
    """
    n = len(train_paths)
    all_embs, all_lbls = [], []
    print(f"\n  AUG embedding: {n} klip × {N_AUGMENTS} = {n*N_AUGMENTS} örnek")

    for aug_idx in range(N_AUGMENTS):
        batches  = range(0, n, EMBED_BATCH)
        iterator = (tqdm(batches, desc=f"  AUG {aug_idx+1}/{N_AUGMENTS}",
                         unit="batch",
                         total=(n + EMBED_BATCH - 1) // EMBED_BATCH)
                    if TQDM_OK else batches)

        epoch_embs = []
        for i in iterator:
            batch_paths = train_paths[i: i + EMBED_BATCH]
            waves = []
            for j, p in enumerate(batch_paths):
                y    = load_audio_clip(p)
                seed = (hash(p) + aug_idx * 99991) % (2 ** 31)
                waves.append(augment_waveform(y, seed=seed))
            epoch_embs.append(
                _embed_batch(encoder, torch.FloatTensor(np.stack(waves)))
            )
        all_embs.append(np.vstack(epoch_embs))
        all_lbls.append(train_labels_enc)

    return (np.vstack(all_embs).astype(np.float32),
            np.concatenate(all_lbls))


# ================================================================
# 💾  CACHE YÖNETİMİ
# ================================================================

def _paths_hash(paths: list) -> str:
    return hashlib.md5("||".join(sorted(paths)).encode()).hexdigest()[:12]


def save_cache(embeddings, paths, labels):
    os.makedirs(os.path.dirname(EMBED_CACHE), exist_ok=True)
    with open(EMBED_CACHE, "wb") as f:
        pickle.dump({"embeddings": embeddings, "paths": paths, "labels": labels,
                     "sr": SR, "duration": DURATION,
                     "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "n_samples": len(paths)}, f, protocol=4)
    print(f"  BASE cache → {EMBED_CACHE}  "
          f"({os.path.getsize(EMBED_CACHE)/1024/1024:.1f} MB)")


def load_cache(expected_paths: list):
    if not os.path.exists(EMBED_CACHE):
        return None
    try:
        with open(EMBED_CACHE, "rb") as f:
            c = pickle.load(f)
        if set(c["paths"]) == set(expected_paths):
            print(f"  BASE cache  |  {c['n_samples']} örnek  |  {c['created']}")
            return c["embeddings"], c["paths"], c["labels"]
        new = len(set(expected_paths) - set(c["paths"]))
        print(f"  BASE cache geçersiz (+{new} yeni dosya) → yeniden…")
        return None
    except Exception as e:
        print(f"  BASE cache yüklenemedi ({e}) → yeniden…")
        return None


def save_aug_cache(embeddings, labels, train_paths):
    h = _paths_hash(train_paths)
    with open(AUG_CACHE, "wb") as f:
        pickle.dump({"embeddings": embeddings, "labels": labels,
                     "paths_hash": h, "n_augments": N_AUGMENTS,
                     "created": time.strftime("%Y-%m-%d %H:%M:%S")}, f, protocol=4)
    print(f"  AUG cache  → {AUG_CACHE}  "
          f"({os.path.getsize(AUG_CACHE)/1024/1024:.1f} MB)")


def load_aug_cache(train_paths: list):
    if not os.path.exists(AUG_CACHE):
        return None
    try:
        with open(AUG_CACHE, "rb") as f:
            c = pickle.load(f)
        if c["paths_hash"] == _paths_hash(train_paths) and \
           c["n_augments"] == N_AUGMENTS:
            print(f"  AUG cache  |  {len(c['embeddings'])} örnek  |  {c['created']}")
            return c["embeddings"], c["labels"]
        print("  AUG cache geçersiz (train seti / N_AUGMENTS değişti) → yeniden…")
        return None
    except Exception as e:
        print(f"  AUG cache yüklenemedi ({e}) → yeniden…")
        return None


# ================================================================
# 📦  DATASET
# ================================================================

class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, labels: np.ndarray):
        self.X = torch.FloatTensor(embeddings)
        self.y = torch.LongTensor(labels)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ================================================================
# 🏗️  MLP — noise_detector.BEATsClassifier.mlp ile AYNI MİMARİ
# ================================================================

class BEATsMLP(nn.Module):
    """
    768 → 256 → ReLU → Dropout(0.3) → n_classes
    !! noise_detector.BEATsClassifier.mlp ile BİREBİR AYNI olmalı !!
    Checkpoint: ckpt["model_state"] = model.net.state_dict()
    """
    def __init__(self, embed_dim=EMBED_DIM, n_classes=N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ================================================================
# 🏋  EĞİTİM & DEĞERLENDİRME
# ================================================================

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = correct = total = 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (out.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate_model(model, loader, criterion):
    model.eval()
    total_loss = correct = total = 0
    preds_all, labels_all = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out  = model(X)
        loss = criterion(out, y)
        total_loss += loss.item() * len(y)
        p           = out.argmax(1)
        correct    += (p == y).sum().item()
        total      += len(y)
        preds_all.extend(p.cpu().tolist())
        labels_all.extend(y.cpu().tolist())
    return total_loss / total, correct / total, preds_all, labels_all


def per_class_metrics(y_true, y_pred, label_names):
    # ⚠ labels=range(len(label_names)) ŞART. Olmadan sklearn sadece
    # y_true/y_pred'de FİİLEN görülen sınıflar için skor döndürüyor —
    # az örnekli/hiç örneği olmayan sınıflarla (bizim durumumuzda 7-8
    # sınıf 0 örnekli) dict(zip(label_names, skorlar)) YANLIŞ isimlerle
    # eşleşiyordu (zip kısa listede kesiliyor) ve sonunda KeyError
    # veriyordu. labels= ile her sınıf için (yoksa 0.0) garanti skor gelir.
    idx  = range(len(label_names))
    f1s  = f1_score(y_true, y_pred, average=None, zero_division=0, labels=idx)
    recs = recall_score(y_true, y_pred, average=None, zero_division=0, labels=idx)
    return dict(zip(label_names, f1s)), dict(zip(label_names, recs))


def print_per_class(f1d, rcd, tag=""):
    print(f"  ── Per-Class {tag} ─────────────────────────────────")
    print(f"  {'Sınıf':12s}  {'F1':>6}  {'Recall':>7}")
    for cls in sorted(f1d):
        warn = " ⚠" if rcd[cls] < 0.5 else ""
        bar  = "▓" * int(f1d[cls] * 10)
        print(f"  {cls:12s}  {f1d[cls]:6.4f}  {rcd[cls]:7.4f}  {bar}{warn}")
    print()


# ================================================================
# 📊  PLOT FONKSİYONLARI
# ================================================================

def plot_training_curves(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train", color="#2196F3")
    axes[0].plot(epochs, history["val_loss"],   label="Val",   color="#F44336")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train", color="#2196F3")
    axes[1].plot(epochs, history["val_acc"],   label="Val",   color="#F44336")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(epochs, history["val_f1"], color="#4CAF50", linewidth=2,
            label="Val F1 Macro")
    if history["val_f1"]:
        best_f1 = max(history["val_f1"])
        best_ep = history["val_f1"].index(best_f1) + 1
        ax.scatter([best_ep], [best_f1], color="red", zorder=5)
        ax.annotate(f"  {best_f1:.4f} (ep{best_ep})",
                    xy=(best_ep, best_f1), fontsize=8, color="red")
    ax.set_title("F1 Macro (Val)"); ax.set_ylim(0, 1.05)
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle("BEATs MLP v2  (Group-Split + Aug)", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    out = os.path.join(save_dir, "beats_training_curves.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  → {out}")


def plot_per_class_f1_curve(history, label_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    # 10 sınıf için 10 ayırt edici renk (6'dan büyütüldü)
    colors = ["#E91E63","#2196F3","#4CAF50","#FF9800","#9C27B0","#00BCD4",
              "#FFEB3B","#795548","#607D8B","#8BC34A"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, cls in enumerate(label_names):
        if history.get(f"val_f1_{cls}"):
            ax.plot(epochs, history[f"val_f1_{cls}"], label=cls,
                    color=colors[i % len(colors)], linewidth=1.8)
    ax.set_title("Per-Class F1 (Val) — BEATs MLP v2", fontsize=12,
                 fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05); ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(save_dir, "beats_per_class_f1.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  → {out}")


def plot_confusion_matrix(y_true, y_pred, label_names, save_dir, suffix=""):
    os.makedirs(save_dir, exist_ok=True)
    # ⚠ labels=range(...) ŞART — yoksa cm'nin boyutu sadece fiilen görülen
    # sınıf sayısı kadar olur, ama xticklabels/yticklabels her zaman TAM
    # 10 sınıf — boyut uyuşmazlığı (aynı kök sebep: per_class_metrics'teki
    # gibi, burada plot çizerken patlardı).
    cm = confusion_matrix(y_true, y_pred, labels=range(len(label_names)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title in [
        (axes[0], cm, "d", "Confusion Matrix (sayı)"),
        (axes[1], cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9),
         ".2f", "Confusion Matrix (normalize)"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=label_names, yticklabels=label_names, ax=ax)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
        ax.tick_params(axis="x", rotation=45)
    plt.suptitle(f"BEATs MLP v2{suffix}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(save_dir, f"beats_confusion_matrix{suffix}.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  → {out}")


# ================================================================
# 🏃  ANA PIPELINE
# ================================================================

def main(rebuild_cache=False):
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,  exist_ok=True)

    print("=" * 65)
    print("  train_beats.py v2  —  Group-Split + Waveform Augmentation")
    print(f"  Device  : {device}")
    print(f"  Manifest: {os.path.basename(MANIFEST_CSV)}")
    print(f"  N_AUGMENTS={N_AUGMENTS}  NOISE={AUG_NOISE_PROB}  "
          f"GAIN={AUG_GAIN_PROB}  PITCH={AUG_PITCH_PROB}")
    print("=" * 65)

    # ── 1. Manifest ──────────────────────────────────────────────
    if not os.path.exists(MANIFEST_CSV):
        print(f"[HATA] Manifest bulunamadı: {MANIFEST_CSV}")
        print("       Önce: python dataset_builder.py")
        raise SystemExit(1)

    df = pd.read_csv(MANIFEST_CSV)
    n_before = len(df)
    df = df[df["path"].apply(chunk_path_exists)].reset_index(drop=True)
    if len(df) < n_before:
        print(f"  ⚠ {n_before - len(df)} manifest satırı dosya bulunamadığı için elendi")

    # Svantek kayıtlarını CSV olmadan birleştir
    df_sv = scan_svantek_recordings()
    df_sv = df_sv[df_sv["path"].apply(chunk_path_exists)].reset_index(drop=True)
    if not df_sv.empty:
        df = pd.concat([df, df_sv], ignore_index=True)
        print(f"  +{len(df_sv)} Svantek kaydı manifest'e eklendi")

    print(f"\n  Toplam örnek: {len(df)}")

    dist = Counter(df["label"])
    print("\n── Sınıf Dağılımı ──────────────────────────────────────────")
    for lbl, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {lbl:10s} {cnt:5d}  {'█' * (cnt // 50)}")

    # ── 2. Label encode ──────────────────────────────────────────
    le = LabelEncoder()
    le.fit(BEATS_CLASSES)

    unknown = set(df["label"].unique()) - set(BEATS_CLASSES)
    if unknown:
        print(f"\n  ⚠  Bilinmeyen sınıflar filtreleniyor: {unknown}")
        df = df[df["label"].isin(BEATS_CLASSES)].reset_index(drop=True)

    df["label_enc"] = le.transform(df["label"])
    labels = list(le.classes_)
    assert labels == BEATS_CLASSES, f"Sınıf sırası hatası: {labels}"
    print(f"\n  Sınıflar: {labels}")

    # ── 3. Source ID — group-aware split için ────────────────────
    paths      = df["path"].tolist()
    lbls       = df["label"].tolist()
    enc_labels = le.transform(lbls)
    source_ids = np.array([extract_source_id(p) for p in paths])

    n_groups = len(set(source_ids))
    esc_groups = sum(1 for s in source_ids if s.startswith("esc50_"))
    print(f"\n── Source ID İstatistikleri ────────────────────────────────")
    print(f"  Toplam pencere    : {len(paths)}")
    print(f"  Benzersiz kaynak  : {n_groups}")
    print(f"  ESC-50 grubu      : {esc_groups}  "
          f"(A/B varyantları artık ayrılmıyor)")

    # ── 4. BASE embedding cache ──────────────────────────────────
    if rebuild_cache:
        for c in [EMBED_CACHE, AUG_CACHE]:
            if os.path.exists(c):
                os.remove(c)
                print(f"  Silindi: {c}")

    encoder      = None
    cache_result = load_cache(paths)

    if cache_result is None:
        encoder    = load_beats_encoder()
        embeddings = extract_all_embeddings(paths, encoder)
        save_cache(embeddings, paths, lbls)
    else:
        cached_embs, cached_paths, _ = cache_result
        idx_map    = {p: i for i, p in enumerate(cached_paths)}
        order      = [idx_map[p] for p in paths]
        embeddings = cached_embs[order]
        print(f"  Embeddings: {embeddings.shape}")

    # ── 5. GROUP-AWARE SPLIT ─────────────────────────────────────
    indices = np.arange(len(embeddings))

    gss_test = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                  random_state=RANDOM_SEED)
    train_val_idx, test_idx = next(
        gss_test.split(indices, enc_labels, groups=source_ids)
    )

    gss_val = GroupShuffleSplit(n_splits=1,
                                 test_size=VAL_SIZE / (1 - TEST_SIZE),
                                 random_state=RANDOM_SEED)
    train_rel_idx, val_rel_idx = next(
        gss_val.split(train_val_idx,
                      enc_labels[train_val_idx],
                      groups=source_ids[train_val_idx])
    )
    train_idx = train_val_idx[train_rel_idx]   # Relatif → Mutlak indeks dönüşümü
    val_idx   = train_val_idx[val_rel_idx]     # Relatif → Mutlak indeks dönüşümü

    # Leakage kontrolü — kaynak grupları arasında kesişim olmamalı
    train_src = set(source_ids[train_idx])
    val_src   = set(source_ids[val_idx])
    test_src  = set(source_ids[test_idx])
    leak_tv = len(train_src & val_src)
    leak_tt = len(train_src & test_src)

    print(f"\n── Group-Aware Split ───────────────────────────────────────")
    print(f"  Train  : {len(train_idx):>5} örnek  |  "
          f"{len(train_src):>4} kaynak grup")
    print(f"  Val    : {len(val_idx):>5} örnek  |  "
          f"{len(val_src):>4} kaynak grup")
    print(f"  Test   : {len(test_idx):>5} örnek  |  "
          f"{len(test_src):>4} kaynak grup")
    print(f"  Leakage (train∩val): {leak_tv}  |  "
          f"(train∩test): {leak_tt}")
    if leak_tv == 0 and leak_tt == 0:
        print("  ✓ Kaynak grupları arasında sızıntı yok")
    else:
        print("  ⚠  Sızıntı tespit edildi — source_id mantığı gözden geçir")

    X_val,  y_val  = embeddings[val_idx],  enc_labels[val_idx]
    X_test, y_test = embeddings[test_idx], enc_labels[test_idx]

    # ── 6. AUG embedding (sadece train) ─────────────────────────
    train_paths_list = [paths[i] for i in train_idx]
    train_enc        = enc_labels[train_idx]

    aug_result = load_aug_cache(train_paths_list)

    if aug_result is None:
        if encoder is None:
            encoder = load_beats_encoder()
        X_aug, y_aug = extract_aug_embeddings(
            train_paths_list, train_enc, encoder)
        save_aug_cache(X_aug, y_aug, train_paths_list)
        if encoder is not None:
            del encoder
            if device.type == "cuda":
                torch.cuda.empty_cache()
    else:
        X_aug, y_aug = aug_result

    # Eğitim seti = orijinal + augmented
    X_train = np.vstack([embeddings[train_idx], X_aug])
    y_train = np.concatenate([train_enc, y_aug])

    print(f"\n  Eğitim seti:  {len(train_idx)} orijinal  +  "
          f"{len(X_aug)} augmented  =  {len(X_train)} toplam")
    print(f"  Val: {len(X_val)}  |  Test: {len(X_test)}")

    # ── 7. DataLoader ─────────────────────────────────────────────
    pin = device.type == "cuda"
    train_loader = DataLoader(EmbeddingDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(EmbeddingDataset(X_val, y_val),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=pin)
    test_loader  = DataLoader(EmbeddingDataset(X_test, y_test),
                              batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=pin)

    # ── 8. Model & optimizer ─────────────────────────────────────
    model     = BEATsMLP(n_classes=len(labels)).to(device)
    trainable = sum(p.numel() for p in model.parameters())
    print(f"\n  BEATsMLP  |  {trainable:,} parametre")

    weights   = torch.FloatTensor(
        [MANUAL_CLASS_WEIGHTS[labels[i]] for i in range(len(labels))]
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5)

    # ── 9. Eğitim döngüsü ────────────────────────────────────────
    best_f1, patience_cnt = -1.0, 0
    history = {"train_loss":[], "train_acc":[],
               "val_loss":[], "val_acc":[], "val_f1":[]}
    for cls in labels:
        history[f"val_f1_{cls}"] = []

    print(f"\n{'='*65}")
    print(f"  MLP Eğitimi  ({EPOCHS} epoch max, patience={PATIENCE})")
    print(f"{'='*65}")
    print(f"  {'Ep':>3}  {'TrLoss':>8}  {'TrAcc':>6}  "
          f"{'VaLoss':>8}  {'VaAcc':>6}  {'F1Mac':>6}  {'Süre':>5}  Best")
    print("  " + "─" * 63)

    current_epoch = 0
    try:
        for epoch in range(1, EPOCHS + 1):
            current_epoch = epoch
            t0 = time.time()

            tr_loss, tr_acc = train_one_epoch(
                model, train_loader, criterion, optimizer)
            va_loss, va_acc, va_preds, va_true = evaluate_model(
                model, val_loader, criterion)
            va_f1 = f1_score(va_true, va_preds, average="macro",
                             zero_division=0, labels=range(len(labels)))
            scheduler.step()

            elapsed = time.time() - t0
            is_best = va_f1 > best_f1
            mark    = "★" if is_best else ""

            print(f"  {epoch:3d}  {tr_loss:8.4f}  {tr_acc:6.4f}  "
                  f"{va_loss:8.4f}  {va_acc:6.4f}  {va_f1:6.4f}  "
                  f"{elapsed:4.0f}s  {mark}")

            f1d, rcd = per_class_metrics(va_true, va_preds, labels)
            print("  Recall → " + "  ".join(
                f"{c[:4]}:{rcd[c]:.2f}" for c in labels))
            print()

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(va_loss)
            history["val_acc"].append(va_acc)
            history["val_f1"].append(va_f1)
            for cls in labels:
                history[f"val_f1_{cls}"].append(f1d.get(cls, 0.0))

            if is_best:
                best_f1, patience_cnt = va_f1, 0
                torch.save({
                    "epoch":       epoch,
                    "model_state": model.net.state_dict(),
                    "label_names": labels,
                    "n_classes":   len(labels),
                    "val_f1":      best_f1,
                    "embed_dim":   EMBED_DIM,
                }, BEATS_MLP_OUT)
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    print(f"  ⏹  Early stopping (patience={PATIENCE})")
                    break

    except KeyboardInterrupt:
        print(f"\n  ⚡  Ctrl+C — ep {current_epoch}")
        _, _, preds, true = evaluate_model(model, val_loader, criterion)
        f1m      = f1_score(true, preds, average="macro", zero_division=0,
                            labels=range(len(labels)))
        f1d, rcd = per_class_metrics(true, preds, labels)
        print(f"  Val F1 Macro: {f1m:.4f}")
        print_per_class(f1d, rcd, tag=f"Ctrl+C ep{current_epoch}")
        if history["train_loss"]:
            plot_training_curves(history, PLOTS_DIR)
            plot_per_class_f1_curve(history, labels, PLOTS_DIR)
        plot_confusion_matrix(true, preds, labels, PLOTS_DIR,
                              suffix=f"_interrupt_ep{current_epoch}")
        intr = BEATS_MLP_OUT.replace(".pt", f"_interrupt_ep{current_epoch}.pt")
        torch.save({"epoch": current_epoch,
                    "model_state": model.net.state_dict(),
                    "label_names": labels, "n_classes": len(labels),
                    "val_f1": history["val_f1"][-1] if history["val_f1"] else 0.0,
                    "embed_dim": EMBED_DIM}, intr)
        print(f"  → {intr}")
        sys.exit(0)

    # ── 10. Test ─────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  EN İYİ MODEL — TEST")
    ckpt = torch.load(BEATS_MLP_OUT, map_location=device, weights_only=False)
    model.net.load_state_dict(ckpt["model_state"])
    print(f"  Epoch {ckpt['epoch']}  |  Val F1: {ckpt['val_f1']:.4f}")

    _, te_acc, te_preds, te_true = evaluate_model(model, test_loader, criterion)
    te_f1_mac = f1_score(te_true, te_preds, average="macro",  zero_division=0,
                         labels=range(len(labels)))
    te_f1_wt  = f1_score(te_true, te_preds, average="weighted", zero_division=0,
                         labels=range(len(labels)))
    te_f1d, te_rcd = per_class_metrics(te_true, te_preds, labels)

    print(f"\n  Accuracy     : {te_acc:.4f}  ({te_acc:.1%})")
    print(f"  F1 Macro     : {te_f1_mac:.4f}   ← ASIL METRİK (leakage-free)")
    print(f"  F1 Weighted  : {te_f1_wt:.4f}")
    print()
    print_per_class(te_f1d, te_rcd, tag="(Test — Group Split)")
    print(classification_report(te_true, te_preds, labels=range(len(labels)),
                                 target_names=labels, digits=3, zero_division=0))

    # ── 11. Grafikler ─────────────────────────────────────────────
    plot_training_curves(history, PLOTS_DIR)
    plot_per_class_f1_curve(history, labels, PLOTS_DIR)
    plot_confusion_matrix(te_true, te_preds, labels, PLOTS_DIR, suffix="_test")
    _, _, va_fin, va_true_fin = evaluate_model(model, val_loader, criterion)
    plot_confusion_matrix(va_true_fin, va_fin, labels, PLOTS_DIR, suffix="_val")

    joblib.dump(le, os.path.join(MODELS_DIR, "beats_label_encoder.pkl"))

    print("=" * 65)
    print("  🎉  BEATs MLP v2 EĞİTİMİ TAMAMLANDI")
    print(f"     F1 Macro (leakage-free) : {te_f1_mac:.1%}")
    print(f"     Best Epoch              : {ckpt['epoch']}")
    print(f"     MLP model               : {BEATS_MLP_OUT}")
    print(f"     BASE cache              : {EMBED_CACHE}")
    print(f"     AUG  cache              : {AUG_CACHE}")
    print("=" * 65)


# ================================================================
if __name__ == "__main__":
    rebuild = "--rebuild-cache" in sys.argv
    if rebuild:
        print("  --rebuild-cache: tüm cache'ler silinecek")
    main(rebuild_cache=rebuild)
