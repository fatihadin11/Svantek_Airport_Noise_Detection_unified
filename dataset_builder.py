"""
╔══════════════════════════════════════════════════════════════════╗
║         Dataset Builder v6  —  Collector SQLite + LIVE           ║
╚══════════════════════════════════════════════════════════════════╝

v6'da v4/v5'ten fark — ESKİ VERİ SETİ TAMAMEN İPTAL EDİLDİ:
  - KALDIRILDI: ESC-50, AeroSonicDB, env_audio (AMBIENT), GENERIC_AUDIO_CLASSIFIER
    yükleyicileri ve klasör→sınıf eşleme tabloları. Bu veri kaynakları
    eski taksonomiye (AIRCRAFT/AMBIENT/SPEECH/TRAFFIC/WIND/OTHER) özgüydü
    ve yeni taksonomiyle (10 sınıf) anlamlı biçimde eşleşmiyor.
  - EKLENDİ: airport-audio-collector projesinin pipeline.sqlite3
    veritabanından DOĞRUDAN okuma (load_from_collector_db). Bu, iki
    projeyi tam olarak birbirine bağlar — ayrı bir "veri seti indirme/
    organize etme" adımına gerek kalmaz.
  - manifest_v4/v5 ayrımı (temel + canlı klip varyantı) kaldırıldı;
    artık tek, birleşik manifest_v6.csv üretiliyor (collector DB + onaylı
    live klipler her zaman birlikte).
  - live klip akışı (GUI → PendingClipManager → approved_manifest.csv)
    DEĞİŞMEDİ — o modül zaten sınıf adı almıyor, çalışan davranışı
    korunuyor (bkz. load_live_records, aynı fonksiyon).

Sınıflar: bkz. class_config.py (9 aktif + OTHER = 10 sınıf)

Çalıştırma sırası:
  python dataset_builder.py         → manifest_v6.csv üretir
  python train_beats.py             → BEATs MLP eğitir
  python train_efficientnet.py      → EfficientNet eğitir
"""

import os
import sqlite3
import warnings
from pathlib import Path
from collections import Counter

import csv
import numpy as np
import pandas as pd
import librosa
import joblib
from tqdm import tqdm

from class_config import CLASSES
from audio_chunking import chunk_source_file, chunk_path_exists, decode_chunk_path, CLIP_DUR as CHUNK_CLIP_DUR

warnings.filterwarnings("ignore")

# ================================================================
# ⚙️  AYARLAR
# ================================================================

PROJECT_ROOT    = r"C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise"
CACHE_DIR       = os.path.join(PROJECT_ROOT, "cache")
MANIFEST_OUT    = os.path.join(PROJECT_ROOT, "cache", "manifest_v6.csv")

# Canlı mikrofon klipler (D diskinde) — DEĞİŞMEDİ
LIVE_CLIPS_DIR     = r"D:\Airport_Live_Clips"
APPROVED_MANIFEST  = os.path.join(LIVE_CLIPS_DIR, "approved_manifest.csv")

# ----------------------------------------------------------------
# airport-audio-collector SQLite entegrasyonu — YENİ (v6)
# ----------------------------------------------------------------
# ⚠ VARSAYIM: collector projesinin schema.py'sindeki durum akışına göre
# ("discovered -> ... -> accepted / rejected") nihai kabul durumu
# 'accepted'. Bu, collector'ın downloaders/validators/quality modüllerini
# görmediğim için schema.py'deki YORUMDAN çıkardığım bir varsayım.
# Aşağıdaki fonksiyon artık DB'deki gerçek status dağılımını her zaman
# yazdırıyor — eğer 0 örnek geliyorsa, muhtemelen örnekler henüz
# 'accepted'e ulaşmamış (validate/dedup/quality/diversity aşamaları
# tamamlanmamış) demektir. O durumda COLLECTOR_ACCEPTED_STATUS'a
# ihtiyacın olan diğer durumları da (liste olarak) ekleyebilirsin.
COLLECTOR_DB_PATH = os.environ.get(
    "COLLECTOR_DB_PATH",
    r"C:\Users\Fatih\Desktop\TUBITAK\Airport_Audio_Collector\airport-audio-collector\db\pipeline.sqlite3",
)

# Tek string ("quality_scored") ya da liste (["quality_scored", ...]) olabilir.
# DÜZELTME: eski değer ("accepted") collector'ın downloaders/validators/quality
# modüllerini görmeden schema.py'deki bir yorumdan çıkarılmış bir varsayımdı
# (bkz. yukarıdaki not) ve gerçek DB'de hiç var olmayan bir status'tü --
# validate_backlog/run_discovery çıktılarındaki gerçek status dağılımı hep
# rejected/downloaded/quality_scored/downloading idi. orchestrator.py'nin
# GERÇEK akışına göre (downloaded -> validated -> dedup_checked ->
# quality_scored) eğitime hazır tek terminal durum 'quality_scored'.
# 'dedup_checked'te takılı kalanlar (crash/kesinti sonucu) kasıtlı olarak
# DAHİL EDİLMEDİ -- final_score'ları yok, bu yüzden eğitime hazır değiller.
COLLECTOR_ACCEPTED_STATUS = "quality_scored"
COLLECTOR_MIN_QUALITY     = 0.60   # quality_scores.final_score eşiği — ayarlanabilir


def _print_collector_status_breakdown(conn: sqlite3.Connection) -> None:
    """DB'deki samples.status dağılımını yazdırır — 0 sonuç aldığında
    'neden' sorusuna hemen cevap versin diye her çalıştırmada basılır."""
    rows = conn.execute(
        "SELECT status, COUNT(*) as n FROM samples GROUP BY status ORDER BY n DESC"
    ).fetchall()
    if not rows:
        print("  [collector_db] samples tablosu tamamen boş — henüz hiç indirme/keşif olmamış.")
        return
    print("  [collector_db] DB'deki gerçek status dağılımı:")
    for r in rows:
        print(f"    {r['status']:15s} {r['n']}")


# ================================================================
# 🆕  COLLECTOR SQLite — YENİ (v6)
# ================================================================

def load_from_collector_db(
    db_path: str = COLLECTOR_DB_PATH,
    min_quality: float = COLLECTOR_MIN_QUALITY,
    status=COLLECTOR_ACCEPTED_STATUS,
) -> list[dict]:
    """
    airport-audio-collector'ın pipeline.sqlite3'ünden doğrudan okur.
    Ayrı bir "veri setini indir/organize et" adımına gerek kalmaz —
    collector'ın kabul ettiği (status + kalite eşiği geçen) her örnek
    doğrudan buradan akar.

    status: tek string ("accepted") ya da liste (["accepted","quality_scored"]).

    predicted_class değerleri collector'ın config/settings.py'sindeki
    TARGET_CLASSES ile ZATEN yeni taksonomide (JET_AIRCRAFT, HELICOPTER,
    ...) — bu iki proje aynı oturumda birlikte güncellendiği için ekstra
    bir sınıf-ismi eşlemesine gerek yok. Yine de beklenmeyen bir değer
    gelirse (eski veri, elle düzeltme vb.) sessizce atlanır ve raporlanır.
    """
    if not os.path.exists(db_path):
        print(f"[⚠ ] Collector DB bulunamadı: {db_path}")
        print(f"      COLLECTOR_DB_PATH sabitini (veya COLLECTOR_DB_PATH "
              f"ortam değişkenini) kendi pipeline.sqlite3 yolunla güncelle.")
        return []

    statuses = [status] if isinstance(status, str) else list(status)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _print_collector_status_breakdown(conn)

    placeholders = ",".join("?" * len(statuses))
    try:
        rows = conn.execute(
            f"""
            SELECT s.id, s.file_path, s.predicted_class, q.final_score
            FROM samples s
            LEFT JOIN quality_scores q ON q.sample_id = s.id
            WHERE s.status IN ({placeholders})
              AND s.file_path IS NOT NULL
              AND (q.final_score IS NULL OR q.final_score >= ?)
            """,
            (*statuses, min_quality),
        ).fetchall()
    except sqlite3.OperationalError as e:
        print(f"[⚠ ] Collector DB sorgusu başarısız: {e}")
        print(f"      Şema beklenenden farklı olabilir (bkz. collector/db/schema.py)")
        conn.close()
        return []
    conn.close()

    if not rows:
        print(f"  [collector_db] status={statuses} + final_score>={min_quality} "
              f"koşuluna uyan 0 satır. Yukarıdaki gerçek dağılıma bakıp "
              f"COLLECTOR_ACCEPTED_STATUS'u genişletmen gerekebilir "
              f"(ör. henüz 'accepted'e ulaşmamış ama indirilmiş örnekleri de "
              f"dahil etmek için status=['accepted','quality_scored',...]).")

    records, missing, unknown_class, n_files, n_chunks = [], 0, Counter(), 0, 0
    for row in rows:
        path = row["file_path"]
        label = row["predicted_class"]
        if label not in CLASSES:
            unknown_class[label] += 1
            continue
        if not path:
            missing += 1
            continue
        chunks = chunk_source_file(path)
        if not chunks:
            missing += 1
            continue
        n_files += 1
        n_chunks += len(chunks)
        for cp in chunks:
            records.append({"path": cp, "label": label, "source": "COLLECTOR_DB"})

    print(f"[collector_db]     {n_files:5d} dosya → {len(records):5d} klip  "
          f"(dosyası bulunamayan: {missing}, ortalama {n_chunks/max(n_files,1):.1f} klip/dosya)")
    if unknown_class:
        print(f"  [ATLANAN — CLASSES dışında sınıf isimleri]")
        for lbl, cnt in sorted(unknown_class.items()):
            print(f"    '{lbl}' → {cnt} örnek atlandı")

    dist = Counter(r["label"] for r in records)
    for lbl, cnt in sorted(dist.items()):
        print(f"  ↳ {lbl:15s}: {cnt:5d}")

    return records


# Eski taksonomiden (AIRCRAFT/AMBIENT/SPEECH/TRAFFIC/WIND/OTHER) yeni
# taksonomiye otomatik eşleme — approved_manifest.csv, GUI güncellenmeden
# ÖNCE toplanmış klipleri de içeriyor (satırlar birikimli, geçmişe dönük
# değişmez). İsim birebir aynıysa direkt geçer (OTHER/SPEECH/TRAFFIC/WIND
# yeni taksonomide de var). AIRCRAFT tek bir yeni sınıfa zorlanamaz ama en
# yakın karşılığı JET_AIRCRAFT'tır (bkz. class_config.py yorumu) — az
# sayıdaysa GUI'den elle gözden geçirmeni öneririm. AMBIENT KASITLI OLARAK
# YOK: yeni taksonomide WIND/PRECIPITATION/NATURE'a bölündü, hangi klip
# hangisine gittiği dinlemeden bilinemez; bu klipler elenir + raporlanır
# (approved_manifest.csv'de durmaya devam ederler, kaybolmazlar).
LEGACY_LABEL_MAP = {
    "OTHER":    "OTHER",
    "SPEECH":   "SPEECH",
    "TRAFFIC":  "TRAFFIC",
    "WIND":     "WIND",
    "AIRCRAFT": "JET_AIRCRAFT",
}


def load_live_records() -> list[dict]:
    """
    D:\\Airport_Live_Clips\\approved_manifest.csv içindeki
    onaylanmış canlı mikrofon kliplerini yükler.

    Sınıf adı hardcode etmez, GUI'nin (PendingClipManager) ürettiği
    approved_manifest.csv'deki corrected_label değerini kullanır — zaten
    yeni taksonomideyse (gui_main.py güncellemesinden SONRA toplandıysa)
    olduğu gibi geçer. Eski taksonomideyse LEGACY_LABEL_MAP üzerinden
    çevrilir (bkz. yukarıdaki yorum); eşlenemeyenler (AMBIENT gibi)
    raporlanıp elenir.
    """
    if not os.path.exists(APPROVED_MANIFEST):
        print(f"[live_clips]       Approved manifest bulunamadı: {APPROVED_MANIFEST}")
        print(f"                   GUI'den klip toplayıp onayladıktan sonra çalıştır.")
        return []

    records = []
    missing = 0
    legacy_dropped = Counter()
    with open(APPROVED_MANIFEST, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "approved":
                continue
            path = row.get("clip_path", "")
            if not os.path.exists(path):
                missing += 1
                continue

            raw_label = row["corrected_label"]
            if raw_label in CLASSES:
                label = raw_label                              # zaten yeni taksonomi
            elif raw_label in LEGACY_LABEL_MAP:
                label = LEGACY_LABEL_MAP[raw_label]              # eski -> yeni çevrildi
            else:
                legacy_dropped[raw_label] += 1                   # eşlenemedi (ör. AMBIENT)
                continue

            for cp in chunk_source_file(path):   # canlı klipler zaten ~5s, pratikte tek chunk döner
                records.append({
                    "path":   cp,
                    "label":  label,
                    "source": "LIVE_MIC",
                })

    print(f"[live_clips]       {len(records):5d} onaylı klip  "
          f"(bulunamayan: {missing})")
    if legacy_dropped:
        print(f"  [ATLANAN — eski taksonomiden yeni taksonomiye net eşlemesi olmayan]")
        for lbl, cnt in sorted(legacy_dropped.items()):
            print(f"    '{lbl}' → {cnt} klip atlandı (approved_manifest.csv'de duruyor, kaybolmadı)")

    dist = Counter(r["label"] for r in records)
    for lbl, cnt in sorted(dist.items()):
        print(f"  ↳ {lbl:15s}: {cnt:4d}")

    return records


# ================================================================
# 📊  DAĞILIM RAPORU
# ================================================================

def show_distribution(records: list[dict]):
    dist = Counter(r["label"] for r in records)
    print(f"\n── Sınıf Dağılımı ({len(CLASSES)} Sınıf) ────────────────────────────")
    for lbl in CLASSES:
        cnt = dist.get(lbl, 0)
        bar = "█" * (cnt // 20)
        print(f"  {lbl:15s} {cnt:6d}  {bar}")
    # Beklenmeyen sınıflar varsa göster
    for lbl, cnt in sorted(dist.items()):
        if lbl not in CLASSES:
            print(f"  {lbl:15s} {cnt:6d}  [!BEKLENMEDİK]")
    print(f"\n  TOPLAM: {sum(dist.values())}\n")


# ================================================================
# 🎛️  ÖZELLİK ÇIKARIM  —  DEĞİŞMEDİ (264 boyut, SVM cache için)
# ================================================================

SR        = 22050
CLIP_DUR  = CHUNK_CLIP_DUR  # audio_chunking.py ile AYNI (import edildi, elle kopyalanmadı)
N_MFCC    = 40
N_FFT     = 2048
HOP_FFT   = 512


def load_audio_fixed(path: str, sr: int = SR,
                     duration: float = CLIP_DUR) -> np.ndarray | None:
    real_path, start_sec = decode_chunk_path(path)
    try:
        y, _ = librosa.load(real_path, sr=sr, mono=True,
                            offset=start_sec, duration=duration + 0.5)
        target = int(sr * duration)
        if len(y) >= target:
            start = (len(y) - target) // 2
            y = y[start:start + target]
        else:
            y = np.pad(y, (0, target - len(y)))
        # RMS Normalizasyonu — mikrofon seviye farkını dengele
        rms = np.sqrt(np.mean(y ** 2))
        if rms > 1e-8:
            y = y * (0.1 / rms)
        y = np.clip(y, -1.0, 1.0)

        return y.astype(np.float32)

    except Exception as e:
        print(f"  [!] {os.path.basename(real_path)}@{start_sec:.1f}s: {e}")
        return None


def extract_features(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """264 boyutlu özellik vektörü — DEĞİŞMEDİ."""
    features = []

    mfcc    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                    n_fft=N_FFT, hop_length=HOP_FFT)
    d_mfcc  = librosa.feature.delta(mfcc)
    d2_mfcc = librosa.feature.delta(mfcc, order=2)
    for m in [mfcc, d_mfcc, d2_mfcc]:
        features.extend([m.mean(axis=1), m.std(axis=1)])

    chroma = librosa.feature.chroma_stft(y=y, sr=sr,
                                          n_fft=N_FFT, hop_length=HOP_FFT)
    features.append(chroma.mean(axis=1))

    sc  = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_FFT)
    sb  = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_FFT)
    sr_ = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_FFT)
    sf  = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_FFT)
    for feat in [sc, sb, sr_, sf]:
        features.extend([feat.mean(axis=1), feat.std(axis=1)])

    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_FFT)
    rms = librosa.feature.rms(y=y, hop_length=HOP_FFT)
    for feat in [zcr, rms]:
        features.extend([feat.mean(axis=1), feat.std(axis=1)])

    return np.concatenate(features, axis=0).astype(np.float32)


def build_feature_matrix(records: list[dict],
                          cache_path: str | None = None) -> tuple:
    """Özellik matrisi oluştur. Önbellekten yükler veya hesaplar. DEĞİŞMEDİ."""
    if cache_path and os.path.exists(cache_path):
        print(f"[Önbellekten] {cache_path}")
        d = joblib.load(cache_path)
        return d["X"], d["y"], d["paths"]

    X, y, paths = [], [], []
    for rec in tqdm(records, ncols=80, desc="  Özellik"):
        audio = load_audio_fixed(rec["path"])
        if audio is None:
            continue
        feat = extract_features(audio)
        X.append(feat)
        y.append(rec["label"])
        paths.append(rec["path"])

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        joblib.dump({"X": X, "y": y, "paths": paths}, cache_path)
        print(f"  → Önbellek kaydedildi: {cache_path}")

    print(f"  X.shape = {X.shape}")
    return X, y, paths


# ================================================================
# 🏃  ANA
# ================================================================

if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("=" * 60)
    print("  Dataset Builder v6  —  COLLECTOR SQLite + LIVE")
    print("=" * 60)

    collector_records = load_from_collector_db()
    live_records      = load_live_records()

    records = collector_records + live_records

    if not records:
        print("[HATA] Hiç veri yüklenemedi.")
        print("       COLLECTOR_DB_PATH doğru mu? Live klip onayladın mı?")
        raise SystemExit(1)

    show_distribution(records)

    cache_path = os.path.join(CACHE_DIR, "features_v6.pkl")
    X, y, paths = build_feature_matrix(records, cache_path=cache_path)

    manifest = pd.DataFrame({"path": paths, "label": y})
    manifest.to_csv(MANIFEST_OUT, index=False)
    print(f"\n  manifest_v6.csv kaydedildi → {MANIFEST_OUT}")
    print(f"  ✅ train_beats.py / train_efficientnet.py bu dosyayı okuyacak.")
