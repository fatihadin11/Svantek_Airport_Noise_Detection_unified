"""
╔══════════════════════════════════════════════════════════════╗
║              Train Model v3  —  5 Sınıf  (TEMİZLENDİ)      ║
║   AIRCRAFT / SPEECH / TRAFFIC / WIND / AMBIENT              ║
╚══════════════════════════════════════════════════════════════╝

Değişiklikler (temizlik):
  - RandomForest ve GradientBoosting karşılaştırması kaldırıldı
  - GridSearchCV kaldırıldı — C=50, gamma=scale zaten optimize
  - Sadece SVM eğitiliyor (daha hızlı, daha sade)

Sıralama:
  1. python env_audio_processor.py    → env_clips/ üretir
  2. python dataset_builder_v3.py     → features_v3.pkl üretir
  3. python train_model_v3.py         → best_model.pkl günceller

Gereksinimler:
  pip install scikit-learn imbalanced-learn joblib numpy pandas
              matplotlib seaborn
"""

import os
import time
import warnings
from collections import Counter

import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ================================================================
# ⚙️  AYARLAR
# ================================================================

PROJECT_ROOT = r"C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise"
CACHE_PATH   = os.path.join(PROJECT_ROOT, "cache",  "features_v3.pkl")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "outputs", "training_v3")

CV_FOLDS    = 5
TEST_SIZE   = 0.15
RANDOM_SEED = 42

# noise_detector.py'deki PRIOR_WEIGHTS ile uyumlu
MANUAL_CLASS_WEIGHTS = {
    "AIRCRAFT": 0.5,
    "SPEECH":   2.0,
    "TRAFFIC":  2.5,
    "AMBIENT":  1.5,   
}


# ================================================================
# 📥  VERİ YÜKLEME
# ================================================================

def load_features():
    if not os.path.exists(CACHE_PATH):
        print(f"[HATA] Önbellek bulunamadı: {CACHE_PATH}")
        print(f"       Önce çalıştır: python dataset_builder_v3.py")
        raise SystemExit(1)

    data = joblib.load(CACHE_PATH)
    X, y = data["X"], data["y"]

    dist = Counter(y)
    print(f"\n  X.shape : {X.shape}")
    print(f"  Sınıflar: {sorted(set(y))}")
    print("\n── Sınıf Dağılımı ─────────────────────────────────")
    for lbl, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        bar = "█" * (cnt // 20)
        print(f"  {lbl:10s} {cnt:5d}  {bar}")

    naive_label = max(dist, key=dist.get)
    naive_acc   = dist[naive_label] / len(y)
    print(f"\n  ⚠  Naive baseline (hep '{naive_label}'): {naive_acc:.1%}")
    return X, y


# ================================================================
# 🤖  SVM EĞİTİMİ
# ================================================================

def train_svm(X_tr: np.ndarray, y_enc: np.ndarray, le: LabelEncoder) -> Pipeline:
    """
    SVM_RBF (C=50, gamma=scale) eğit.
    5-Fold CV ile F1 Macro raporla, ardından tüm veriyle fit et.
    """
    labels = list(le.classes_)
    cw_enc = {i: MANUAL_CLASS_WEIGHTS.get(lbl, 1.0) for i, lbl in enumerate(labels)}

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(kernel="rbf", C=50, gamma="scale",
                       probability=True, class_weight=cw_enc,
                       random_state=RANDOM_SEED)),
    ])

    cv  = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    print("=" * 56)
    print("  SVM_RBF  (C=50, gamma=scale)  — 5-Fold CV")
    print("=" * 56)

    t0  = time.time()
    f1s = cross_val_score(pipe, X_tr, y_enc, cv=cv,
                          scoring="f1_macro", n_jobs=-1)
    accs = cross_val_score(pipe, X_tr, y_enc, cv=cv,
                           scoring="accuracy", n_jobs=-1)
    elapsed = time.time() - t0

    print(f"  F1 Macro : {f1s.mean():.4f} ± {f1s.std():.4f}")
    print(f"  Accuracy : {accs.mean():.4f} ± {accs.std():.4f}")
    print(f"  Süre     : {elapsed:.0f}s\n")

    # Tüm eğitim verisiyle nihai fit
    pipe.fit(X_tr, y_enc)
    return pipe


# ================================================================
# 📊  GÖRSELLEŞTIRME
# ================================================================

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
        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gerçek")
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("Confusion Matrix v3 — SVM_RBF (C=50)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(save_dir, "confusion_matrix_v3.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  → {out}")


# ================================================================
# 🏃  ANA PIPELINE
# ================================================================

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,  exist_ok=True)

    print("=" * 56)
    print("  Train Model v3  —  SVM Only")
    print("=" * 56)

    # 1. Veri yükle
    X, y_raw = load_features()

    # Label encode
    le     = LabelEncoder()
    y_enc  = le.fit_transform(y_raw)
    labels = list(le.classes_)
    print(f"\n  Sınıflar: {labels}")

    # Test ayır
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=TEST_SIZE,
        stratify=y_enc, random_state=RANDOM_SEED
    )
    print(f"  Eğitim: {len(X_tr)}  |  Test: {len(X_te)}")

    # 2. SMOTE
    print("\n  SMOTE uygulanıyor...")
    smote      = SMOTE(random_state=RANDOM_SEED, k_neighbors=3)
    X_sm, y_sm = smote.fit_resample(X_tr, y_tr)
    print(f"  SMOTE: {len(X_tr)} → {len(X_sm)} örnek")

    # 3. SVM eğit
    model = train_svm(X_sm, y_sm, le)

    # 4. Test değerlendirme
    print("=" * 56)
    print("  TEST SONUÇLARI")
    print("=" * 56)
    y_pred = model.predict(X_te)
    acc    = (y_pred == y_te).mean()
    f1_mac = f1_score(y_te, y_pred, average="macro")
    f1_wt  = f1_score(y_te, y_pred, average="weighted")

    print(f"\n  Accuracy   : {acc:.4f}  ({acc:.1%})")
    print(f"  F1 Macro   : {f1_mac:.4f}   ← ASIL METRİK")
    print(f"  F1 Weighted: {f1_wt:.4f}")
    print(f"\n{classification_report(y_te, y_pred, target_names=labels, digits=3)}")

    plot_confusion_matrix(y_te, y_pred, labels, PLOTS_DIR)

    # 5. Kaydet
    joblib.dump(model, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(le,    os.path.join(MODELS_DIR, "label_encoder.pkl"))
    joblib.dump({
        "model_name":    "SVM_RBF",
        "params":        {"C": 50, "gamma": "scale", "kernel": "rbf"},
        "test_accuracy": float(acc),
        "f1_macro":      float(f1_mac),
        "label_names":   labels,
        "feature_dim":   X.shape[1],
        "sr":            22050,
        "duration":      5.0,
        "n_mfcc":        40,
        "version":       "v3",
    }, os.path.join(MODELS_DIR, "training_meta.pkl"))

    print("\n" + "=" * 56)
    print("  🎉  EĞİTİM TAMAMLANDI — v3")
    print(f"     Model    : SVM_RBF (C=50, gamma=scale)")
    print(f"     F1 Macro : {f1_mac:.1%}")
    print(f"     Sınıflar : {labels}")
    print(f"     Kayıt    : {MODELS_DIR}/best_model.pkl")
    print("=" * 56)


if __name__ == "__main__":
    main()
