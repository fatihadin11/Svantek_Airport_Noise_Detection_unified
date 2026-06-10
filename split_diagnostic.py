"""
split_diagnostic.py  —  Leakage check without training
=======================================================
Manifest'i okur, source_id'leri hesaplar,
farkli split stratejilerini test eder ve leakage raporu uretir.
GPU gerektirmez, embedding cache gerektirmez.
Calisma suresi: ~10-30 saniye
"""

import os, re, sys
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit, train_test_split

# ─── Konfigurasyon ────────────────────────────────────────────────────────────
PROJECT_ROOT = r"C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise"
MANIFEST     = os.path.join(PROJECT_ROOT, "cache", "manifest_v5.csv")

if not os.path.exists(MANIFEST):
    MANIFEST = os.path.join(PROJECT_ROOT, "cache", "manifest_v4.csv")

TEST_SIZE = 0.15
VAL_SIZE  = 0.15
SEED      = 42

# ─── Yardimci fonksiyonlar ────────────────────────────────────────────────────

def extract_source_id_v1(path):
    """Mevcut (potansiyel olarak hatayi) implementasyon."""
    fname = os.path.basename(path)
    m = re.match(r'^\d+-(\d+)-[A-Z]-\d+\.wav$', fname)
    if m:
        return f"esc50_{m.group(1)}"
    return os.path.splitext(fname)[0]   # Sadece dosya adi — collision riski var


def extract_source_id_v2(path):
    """
    Duzeltilmis implementasyon: ESC-50 icin clip_id gruplamasi korunur,
    diger dosyalar icin tam normalize yol kullanilir (dosya adi collision'i yok).
    """
    fname = os.path.basename(path)
    m = re.match(r'^\d+-(\d+)-[A-Z]-\d+\.wav$', fname)
    if m:
        return f"esc50_{m.group(1)}"
    return os.path.normpath(os.path.abspath(path)).lower()


def check_leakage(train_idx, val_idx, test_idx, groups):
    """
    Her bir test/val sampli icin source_id'sinin train'de de
    gorunup gozukmedigini sayar.
    """
    train_grps = set(groups[i] for i in train_idx)
    tv = sum(1 for i in val_idx  if groups[i] in train_grps)
    tt = sum(1 for i in test_idx if groups[i] in train_grps)
    return tv, tt


def split_random_then_group(idx, labels, groups):
    """
    Hipotez: Stage-1 random (train_test_split),
             Stage-2 group-aware (GroupShuffleSplit)
    → Buyuk ihtimalle mevcut hatayi uygulayan kod budur.
    """
    labels_arr = np.array(labels)
    groups_arr = np.array(groups)

    # Stage 1: HATALI — random split, gruplar dikkate alinmiyor
    tv_idx, te_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=SEED, stratify=labels_arr
    )

    # Stage 2: Dogru — GroupShuffleSplit
    val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=SEED)
    tr_rel, va_rel = next(gss.split(
        tv_idx,
        labels_arr[tv_idx],
        groups_arr[tv_idx]
    ))
    tr_idx = tv_idx[tr_rel]
    va_idx = tv_idx[va_rel]
    return tr_idx, va_idx, te_idx


def split_full_group(idx, labels, groups):
    """
    Duzeltilmis: Her iki asamada da GroupShuffleSplit.
    train∩val ve train∩test her ikisi de 0 olmali.
    """
    labels_arr = np.array(labels)
    groups_arr = np.array(groups)

    # Stage 1: GroupShuffleSplit — test setini gruplar duzeyinde ayir
    gss1 = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    tv_rel, te_rel = next(gss1.split(idx, labels_arr, groups_arr))
    tv_idx = idx[tv_rel]
    te_idx = idx[te_rel]

    # Stage 2: GroupShuffleSplit — val setini gruplar duzeyinde ayir
    val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=SEED)
    tr_rel, va_rel = next(gss2.split(
        tv_idx,
        labels_arr[tv_idx],
        groups_arr[tv_idx]
    ))
    tr_idx = tv_idx[tr_rel]
    va_idx = tv_idx[va_rel]
    return tr_idx, va_idx, te_idx


def print_report(tag, tr_idx, va_idx, te_idx, groups, labels):
    groups_arr = np.array(groups)
    labels_arr = np.array(labels)
    tv, tt = check_leakage(tr_idx, va_idx, te_idx, groups_arr)

    tr_u = len(set(groups_arr[i] for i in tr_idx))
    va_u = len(set(groups_arr[i] for i in va_idx))
    te_u = len(set(groups_arr[i] for i in te_idx))

    # Sinif dagilimi (train)
    cls_counts = Counter(labels_arr[i] for i in tr_idx)

    print(f"\n{'='*60}")
    print(f"  {tag}")
    print(f"{'='*60}")
    print(f"  Train  : {len(tr_idx):>6} ornek  |  {tr_u:>6} benzersiz grup")
    print(f"  Val    : {len(va_idx):>6} ornek  |  {va_u:>6} benzersiz grup")
    print(f"  Test   : {len(te_idx):>6} ornek  |  {te_u:>6} benzersiz grup")
    print(f"\n  Leakage (train∩val)  : {tv}")
    print(f"  Leakage (train∩test) : {tt}")
    if tt == 0 and tv == 0:
        print("  ✓ TEMIZ — leakage yok")
    else:
        print("  ✗ SIZMA MEVCUT")

    print(f"\n  Train sinif dagilimi:")
    for cls in sorted(cls_counts):
        bar = "█" * (cls_counts[cls] // 50)
        print(f"    {cls:<12} {cls_counts[cls]:>6}  {bar}")


# ─── Ana akis ─────────────────────────────────────────────────────────────────

def main():
    if not os.path.exists(MANIFEST):
        print(f"HATA: Manifest bulunamadi: {MANIFEST}")
        sys.exit(1)

    df = pd.read_csv(MANIFEST)
    print(f"\nManifest yuklendi: {len(df)} ornek  ({MANIFEST})")

    if "path" not in df.columns:
        # Bazi manifest versiyonlarinda kolon adi 'file_path' olabilir
        path_col = [c for c in df.columns if "path" in c.lower()][0]
        df = df.rename(columns={path_col: "path"})

    if "label" not in df.columns:
        label_col = [c for c in df.columns if "label" in c.lower()][0]
        df = df.rename(columns={label_col: "label"})

    paths  = df["path"].tolist()
    labels = df["label"].tolist()
    idx    = np.arange(len(paths))

    # ── v1 source_id (mevcut) ─────────────────────────────────────────────
    groups_v1 = [extract_source_id_v1(p) for p in paths]
    n_unique_v1 = len(set(groups_v1))
    n_collision = len(paths) - n_unique_v1
    print(f"\nSource-ID v1 (mevcut): {n_unique_v1} benzersiz  |  {n_collision} collision")

    # Collision ornegi goster
    cnt = Counter(groups_v1)
    collisions = [(sid, c) for sid, c in cnt.items() if c > 1]
    if collisions:
        print(f"  Ornek collision'lar (ilk 5):")
        for sid, c in sorted(collisions, key=lambda x: -x[1])[:5]:
            idxs = [i for i, g in enumerate(groups_v1) if g == sid]
            sample_paths = [os.path.basename(paths[i]) for i in idxs[:3]]
            print(f"    '{sid}' → {c} dosya: {sample_paths}")

    # ── v2 source_id (duzeltilmis) ────────────────────────────────────────
    groups_v2 = [extract_source_id_v2(p) for p in paths]
    n_unique_v2 = len(set(groups_v2))
    print(f"\nSource-ID v2 (duzeltilmis): {n_unique_v2} benzersiz")

    # ── TEST 1: Muhtemel mevcut hata (Random + Group) ─────────────────────
    tr, va, te = split_random_then_group(idx, labels, groups_v1)
    print_report(
        "TEST 1 — Hipotez: Stage-1 random, Stage-2 group  [MEVCUT KOD]",
        tr, va, te, groups_v1, labels
    )

    # ── TEST 2: Her iki asamada Group, eski source_id ─────────────────────
    tr, va, te = split_full_group(idx, labels, groups_v1)
    print_report(
        "TEST 2 — Her iki asamada GroupShuffleSplit, source_id v1",
        tr, va, te, groups_v1, labels
    )

    # ── TEST 3: Her iki asamada Group, duzeltilmis source_id ─────────────
    tr, va, te = split_full_group(idx, labels, groups_v2)
    print_report(
        "TEST 3 — Her iki asamada GroupShuffleSplit, source_id v2  [HEDEF]",
        tr, va, te, groups_v2, labels
    )

    print("\n" + "="*60)
    print("  YORUM")
    print("="*60)
    print("  Eger TEST 1 terminaldeki sayilara eslesiyorsa hipotez dogru.")
    print("  TEST 3'te her iki leakage da 0 ise bu konfigurasyon kullanilmali.")
    print("  train_beats.py'deki split bolumunu buna gore guncelleyin.")


if __name__ == "__main__":
    main()
