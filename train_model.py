"""
╔══════════════════════════════════════════════════════════════╗
║              Train Model v3  —  5 Sınıf                     ║
║   AIRCRAFT / SPEECH / TRAFFIC / WIND / AMBIENT              ║
╚══════════════════════════════════════════════════════════════╝

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
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     GridSearchCV, train_test_split)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
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

# noise_detector.py'deki PRIOR_WEIGHTS ile uyumlu olmalı
# Eğitimde bu ağırlıkları class_weight olarak kullan
MANUAL_CLASS_WEIGHTS = {
    "AIRCRAFT": 0.5,   # bilinçli bastır
    "SPEECH":   2.0,
    "TRAFFIC":  2.5,
    "WIND":     2.5,
    "AMBIENT":  2.0,
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
# 🤖  MODEL EĞİTİMİ
# ================================================================

def get_pipeline(model_name: str, class_weight: dict | None) -> Pipeline:
    """StandardScaler + classifier pipeline."""
    if model_name == "SVM_RBF":
        clf = SVC(kernel="rbf", C=10, gamma="scale",
                  probability=True, class_weight=class_weight)
    elif model_name == "RandomForest":
        clf = RandomForestClassifier(
            n_estimators=300, class_weight=class_weight,
            random_state=RANDOM_SEED, n_jobs=-1)
    else:  # GradientBoosting
        clf = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5,
            random_state=RANDOM_SEED)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])


def compare_models(X_tr, y_enc, le):
    """5-Fold CV ile 3 modeli karşılaştır."""
    labels   = list(le.classes_)
    cv       = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                               random_state=RANDOM_SEED)

    # Sınıf ağırlıkları — hem encode edilmiş index hem de string isim
    cw_enc = {i: MANUAL_CLASS_WEIGHTS.get(lbl, 1.0)
              for i, lbl in enumerate(labels)}

    print("\n" + "=" * 56)
    print("  Model Karşılaştırma (5-Fold CV) — Macro F1")
    print("=" * 56)
    print(f"  {'Model':20s} {'F1 Macro':>9} {'Std':>7}  {'Acc':>7}")
    print("  " + "─" * 52)

    results = {}
    for name in ["SVM_RBF", "RandomForest", "GradientBoosting"]:
        pipe = get_pipeline(name, cw_enc if name != "GradientBoosting" else None)
        t0   = time.time()
        f1s  = cross_val_score(pipe, X_tr, y_enc,
                               cv=cv, scoring="f1_macro", n_jobs=-1)
        accs = cross_val_score(pipe, X_tr, y_enc,
                               cv=cv, scoring="accuracy", n_jobs=-1)
        elapsed = time.time() - t0
        results[name] = {"f1": f1s, "acc": accs, "time": elapsed}
        print(f"  {name:20s} {f1s.mean():9.4f} {f1s.std():7.4f}  "
              f"{accs.mean():7.4f}  ({elapsed:.0f}s)")

    best = max(results, key=lambda k: results[k]["f1"].mean())
    print(f"\n  ✅ En iyi model: {best}  "
          f"(F1={results[best]['f1'].mean():.4f})\n")
    return results, best


PARAM_GRIDS = {
    "SVM_RBF": [
        {"clf__C": [1, 5, 10, 50], "clf__gamma": ["scale", "auto"]}
    ],
    "RandomForest": [
        {"clf__n_estimators": [200, 300, 500],
         "clf__max_depth": [None, 20],
         "clf__min_samples_split": [2, 5]}
    ],
    "GradientBoosting": [
        {"clf__n_estimators": [100, 200],
         "clf__learning_rate": [0.05, 0.1],
         "clf__max_depth": [3, 5]}
    ],
}


def tune_model(best_name: str, X_tr, y_enc, le):
    """GridSearchCV ile en iyi parametreleri bul."""
    labels = list(le.classes_)
    cw_enc = {i: MANUAL_CLASS_WEIGHTS.get(lbl, 1.0)
              for i, lbl in enumerate(labels)}
    pipe = get_pipeline(best_name,
                        cw_enc if best_name != "GradientBoosting" else None)
    cv   = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                            random_state=RANDOM_SEED)

    print("=" * 56)
    print(f"  GridSearchCV: {best_name} (scoring=f1_macro)")
    grid = GridSearchCV(pipe, PARAM_GRIDS[best_name],
                        cv=cv, scoring="f1_macro",
                        n_jobs=-1, verbose=1)
    grid.fit(X_tr, y_enc)
    print(f"  En iyi params: {grid.best_params_}")
    print(f"  En iyi F1    : {grid.best_score_:.4f}\n")
    return grid.best_estimator_, grid.best_params_


# ================================================================
# 📊  GÖRSELLEŞTIRME
# ================================================================

def plot_confusion_matrix(y_true, y_pred, label_names, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    cm_raw    = confusion_matrix(y_true, y_pred)

    for ax, data, title in zip(axes,
                                [cm_raw, cm],
                                ["Ham Sayılar", "Normalize (satır %)"]):
        fmt = "d" if data is cm_raw else ".2f"
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=label_names, yticklabels=label_names, ax=ax)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek")
        ax.tick_params(axis="x", rotation=45)

    plt.suptitle("Confusion Matrix v3 — AMBIENT dahil",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(save_dir, "confusion_matrix_v3.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"  → {out}")


# ================================================================
# 🏃  ANA PIPELINE
# ================================================================

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR,  exist_ok=True)

    # 1. Veri yükle
    print("=" * 56)
    print("  Train Model v3")
    print("=" * 56)
    X, y_raw = load_features()

    # Label encode
    le    = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
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
    smote    = SMOTE(random_state=RANDOM_SEED, k_neighbors=3)
    X_sm, y_sm = smote.fit_resample(X_tr, y_tr)
    print(f"  SMOTE: {len(X_tr)} → {len(X_sm)} örnek")

    # 3. Model karşılaştırma
    results, best_name = compare_models(X_sm, y_sm, le)

    # 4. Hiperparametre optimizasyonu
    tuned_model, best_params = tune_model(best_name, X_sm, y_sm, le)

    # 5. Final eğitim + test değerlendirme
    print("=" * 56)
    print("  TEST SONUÇLARI")
    print("=" * 56)
    tuned_model.fit(X_sm, y_sm)
    y_pred = tuned_model.predict(X_te)
    acc    = (y_pred == y_te).mean()

    from sklearn.metrics import f1_score
    f1_mac = f1_score(y_te, y_pred, average="macro")
    f1_wt  = f1_score(y_te, y_pred, average="weighted")

    print(f"\n  Accuracy   : {acc:.4f}  ({acc:.1%})")
    print(f"  F1 Macro   : {f1_mac:.4f}   ← ASIL METRİK")
    print(f"  F1 Weighted: {f1_wt:.4f}")
    print(f"\n{classification_report(y_te, y_pred, target_names=labels, digits=3)}")

    plot_confusion_matrix(y_te, y_pred, labels, PLOTS_DIR)

    # 6. Kaydet — mevcut modelin üzerine yaz
    joblib.dump(tuned_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(le,          os.path.join(MODELS_DIR, "label_encoder.pkl"))
    joblib.dump({
        "model_name":    best_name,
        "best_params":   best_params,
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
    print(f"     Model    : {best_name}")
    print(f"     F1 Macro : {f1_mac:.1%}")
    print(f"     Sınıflar : {labels}")
    print(f"     Kayıt    : {MODELS_DIR}/best_model.pkl")
    print("=" * 56)


if __name__ == "__main__":
    main()