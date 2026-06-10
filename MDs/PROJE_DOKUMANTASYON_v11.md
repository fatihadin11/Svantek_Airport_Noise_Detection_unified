# ✈ Havalimanı Çevresel Gürültü Tespit Sistemi
**Airport Environmental Noise Detection System**

> **Durum:** Aktif Geliştirme — v4.1
> **Son Güncelleme:** Mayıs 2026
> **Mimari:** PyQt6 GUI + Çok Modelli ML (EfficientNet-B0 / CNN / SVM) + Canlı Mikrofon + Etiketleme Sistemi

---

## İçindekiler

1. [Proje Özeti](#1-proje-özeti)
2. [Klasör Yapısı](#2-klasör-yapısı)
3. [Mimari — Temel Modüller](#3-mimari--temel-modüller)
4. [ML Modelleri](#4-ml-modelleri)
5. [Eğitim Pipeline](#5-eğitim-pipeline)
6. [Veri Akışı](#6-veri-akışı)
7. [Bu Sohbette Yapılan Değişiklikler — v4.1](#7-bu-sohbette-yapılan-değişiklikler--v41)
8. [Mevcut Durum ve Bilinen Sorunlar](#8-mevcut-durum-ve-bilinen-sorunlar)
9. [Yapılacaklar — Sıradaki Adım](#9-yapılacaklar--sıradaki-adım)
10. [Sonraki Sohbet İçin Bağlam](#10-sonraki-sohbet-için-bağlam)

---

## 1. Proje Özeti

Havalimanı yakınındaki ortam seslerini gerçek zamanlı olarak sınıflandıran bir sistemdir. Temel hedef, uçak gürültüsünü diğer çevresel seslerden (trafik, rüzgar, konuşma, ortam sesi) ayırt etmek ve bu verileri kullanıcı etiketleriyle zenginleştirerek modeli sürekli iyileştirmektir.

**Sınıflar:** `AIRCRAFT` | `AMBIENT` | `OTHER` | `SPEECH` | `TRAFFIC` | `WIND`

> ⚠️ `OTHER` sınıfı v4.0 ile eklendi. Modelin tanımadığı sesleri `AIRCRAFT` olarak yanlış etiketleme sorununu çözmek amacıyla eklenmiştir.

**İki temel çalışma modu:**
- **Faz 1 — Dosya Analizi:** WAV/MP3/FLAC dosyası yükle, analiz et, görselleştir
- **Faz 2 — Canlı Mikrofon:** Gerçek zamanlı ses akışı, anlık sınıflandırma, kayıt ve etiketleme

---

## 2. Klasör Yapısı

```
C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise\
│
├── noise_detector.py          # Tüm ML modelleri + ses işleme motoru
├── gui_main.py                # PyQt6 arayüzü (tek dosya)
├── mic_map.py                 # Harita sekmesi (opsiyonel, MapTab)
│
├── dataset_builder_v4.py      # Aktif dataset hazırlama scripti
├── train_efficientnet.py      # EfficientNet eğitim scripti (aktif)
├── train_cnn.py               # CNN eğitim scripti
├── env_audio_processor.py     # AMBIENT klip üretici
│
├── models/
│   ├── best_efficientnet.pt           # EfficientNet-B0 checkpoint (6 sınıf)
│   ├── efficientnet_label_encoder.pkl
│   ├── efficientnet_meta.pkl
│   ├── best_cnn.pt
│   ├── cnn_label_encoder.pkl
│   ├── best_model.pkl                 # SVM
│   └── label_encoder.pkl
│
├── cache/
│   ├── manifest_v3.csv        # ESKİ — 5 sınıf — DOKUNMA
│   ├── manifest_v4.csv        # AKTİF — 6 sınıf, 26079 örnek
│   ├── features_v3.pkl        # ESKİ önbellek — DOKUNMA
│   └── features_v4.pkl        # AKTİF önbellek (CMVN uygulandı, RMS KALDIRILDI)
│
├── Dataset_Airplane/          # AeroSonicDB ses dosyaları + env_audio_manifest.csv
├── Dataset_ESC50/             # ESC-50 ses dosyaları + esc50.csv
│
├── live_clips/
│   ├── pending/               # Henüz onaylanmamış klipler
│   │   ├── AIRCRAFT/
│   │   ├── AMBIENT/
│   │   ├── OTHER/
│   │   ├── SPEECH/
│   │   ├── TRAFFIC/
│   │   └── WIND/
│   ├── approved/              # Onaylanmış, eğitime hazır klipler
│   │   └── (aynı yapı)
│   ├── rejected/
│   ├── pending_manifest.csv
│   └── approved_manifest.csv
│
├── outputs_gui/               # Faz 1 dosya analizi çıktıları (PNG, CSV)
├── outputs/
│   └── training_efficientnet/ # Eğitim grafikleri, confusion matrix PNG'leri
└── outputs_mic/               # Faz 2 kayıt çıktıları
    ├── session_YYYYMMDD_HHMMSS.wav
    └── session_YYYYMMDD_HHMMSS.csv
```

**Google Drive (Colab için):**
```
/content/drive/MyDrive/TUBITAK/Airport_Noise/
├── DATASET/           # D:\Downloads_2\DATASET buraya taşındı
├── Dataset_Airplane/
├── Dataset_ESC50/
├── cache/
├── models/
└── outputs/
```

**Harici Veri Seti:**
```
DATASET\
├── Animals\   (CATS, DOGS, ELEPHANT, HORSE, LIONS → OTHER)
├── Birds\     (CROWS, PARROT, PEACOCK, SPARROW   → OTHER)
├── Environment\ (CROWD→SPEECH, MILITARY→OTHER, OFFICE→AMBIENT,
│                 RAINFALL→AMBIENT, TRAFFIC→TRAFFIC, WIND→WIND)
└── Vehicles\  (airplane,helicopter→AIRCRAFT | car,bus,truck→TRAFFIC)
```

---

## 3. Mimari — Temel Modüller

### `noise_detector.py` içindeki sınıflar

| Sınıf | Görev |
|---|---|
| `AudioLoader` | WAV/MP3/FLAC yükle, `librosa` ile 22050 Hz mono normalize |
| `AudioAnalyzer` | Waveform, STFT spektrogram, Mel spektrogram, dBFS/zaman |
| `FeatureExtractor` | 264-boyutlu ML özellik vektörü (MFCC×3, chroma, SC, ZCR, RMS…) |
| `NoiseFilter` | Band-pass, spectral gating, Wiener filtre |
| `NoiseClassifier` | Kural tabanlı sınıflandırıcı (yedek, ML yoksa devreye girer) |
| `Visualizer` | Dashboard PNG, pasta grafik, waveform, Mel, MFCC çıktıları |
| `AirportNoiseSystem` | Tüm sınıfları orkestre eden ana sınıf |
| `AirportCNN` | 3-katmanlı CNN mimarisi (PyTorch) |
| `EfficientNetAirport` | EfficientNet-B0 transfer learning (torchvision) |

### `gui_main.py` içindeki widget sınıfları

| Sınıf | Görev |
|---|---|
| `MainWindow` | Ana pencere, sekmeler, dosya açma, analiz |
| `SidePanel` | Faz 1 istatistik paneli, sınıf dağılım barları |
| `ClassificationTab` | Sınıflandırma şeridi, dB grafiği, softmax eğrileri, annotation modu |
| `SpectrogramTab` | 128-Mel spektrogram görüntüleyici |
| `FeaturesTab` | ZCR, RMS, Spectral Centroid grafikleri |
| `MicrophoneTab` | Canlı mikrofon, VU metre, rolling strip, etiketleme butonu |
| `DataReviewTab` | Pending + Approved iki tablolu veri yönetimi |
| `AnnotationDialog` | Faz 1 annotation — 6 sınıf |
| `LiveLabelDialog` | Faz 2 etiketleme — 6 sınıf |
| `PendingClipManager` | WAV klip yönetimi, manifest CSV okuma/yazma |
| `LiveSpectrumWidget` | Gerçek zamanlı FFT spektrum görüntüleyici |
| `VUMeter` | Animasyonlu ses seviyesi metre |
| `RollingClassStrip` | Son 60 saniyelik sınıflandırma geçmişi |
| `DbHistoryWidget` | Son 60 saniyelik dBFS geçmişi |
| `MicrophoneWorker` | QThread — mikrofon akışı + inference + majority voting |

---

## 4. ML Modelleri

### Model Hiyerarşisi (öncelik sırası)
```
EfficientNet-B0  →  CNN  →  SVM  →  Kural Tabanlı (yedek)
```

### Ortak Parametreler
```python
SR          = 22050 Hz
WINDOW      = 5.0 saniye
HOP         = 2.5 saniye (%50 overlap)
N_FFT       = 2048
HOP_FFT     = 512
N_MELS      = 128
```

### EfficientNet-B0 (Aktif Model — v4.1)
- 6 sınıf: `AIRCRAFT | AMBIENT | OTHER | SPEECH | TRAFFIC | WIND`
- Giriş: 224×224 RGB Mel Spectrogram
- **CMVN normalizasyonu** uygulanıyor (eğitim ve inference'da)
- **RMS normalizasyonu KALDIRILDI** (AIRCRAFT F1'ini 0.94→0.74 düşürüyordu)
- 2 aşamalı eğitim: Freeze (backbone donuk) → FineTune (son 3 blok açık)

**Son eğitim sonuçları (v4.1 — RMS norm kaldırıldıktan sonra):**
```
AIRCRAFT : Recall=0.69  ← hâlâ iyileştirme gerekiyor
AMBIENT  : Recall=0.93
OTHER    : Recall=0.97
SPEECH   : Recall=0.94
TRAFFIC  : Recall=0.95
WIND     : Recall=0.90
```

> AIRCRAFT'ın 59 örneği OTHER'a, 34'ü AMBIENT'e kaçıyor.
> Nedeni: AeroSonicDB uzak mesafe / düşük enerjili kayıtlar → model "belirsiz = OTHER" öğrendi.
> Geçici çözüm: PRIOR_WEIGHTS'te AIRCRAFT'ı 0.10 → 0.25-0.30'a çek.

### Prior Ağırlık Düzeltmesi (güncel önerilen)
```python
PRIOR_WEIGHTS = {
    "AIRCRAFT": 0.25,   # 0.10'dan yükseltildi — fine-tune öncesi geçici düzeltme
    "SPEECH":   3.5,
    "TRAFFIC":  2.0,
    "AMBIENT":  2.0,
    "WIND":     2.0,
    "OTHER":    1.5,
}
```

### Sınıf Renkleri
```python
CLASS_COLORS = {
    "AIRCRAFT": "#FF6B35",
    "AMBIENT":  "#7EE8A2",
    "SPEECH":   "#A8DADC",
    "TRAFFIC":  "#FFE66D",
    "WIND":     "#4ECDC4",
    "OTHER":    "#9E9E9E",
    "UNKNOWN":  "#6C757D",
}
```

---

## 5. Eğitim Pipeline

```
env_audio_processor.py
  → Dataset_Airplane/env_audio_manifest.csv  (AMBIENT klipler)

dataset_builder_v4.py
  ├── load_esc50_records()       → ESC-50 (560 örnek)
  ├── load_aerosonic_records()   → AeroSonicDB (1895 AIRCRAFT)
  ├── load_ambient_records()     → env_audio (322 AMBIENT)
  └── load_generic_records()     → DATASET (23302 örnek)
      → cache/manifest_v4.csv   (26079 toplam, 6 sınıf)
      → cache/features_v4.pkl   (SVM için, CMVN uygulanmış, RMS YOK)

train_efficientnet.py
  → MANIFEST_CSV = cache/manifest_v4.csv
  → models/best_efficientnet.pt
```

**Önemli:** `features_v4.pkl` silindikten sonra `dataset_builder_v4.py`
yeniden çalıştırılmalıdır (~33 dakika yerel, ~15 dk Colab yerel disk).

---

## 6. Veri Akışı

### Faz 2 — Canlı Mikrofon Inference (güncel)
```
sounddevice InputStream
  → BLOCK_SIZE=1024 chunk'lar kuyruğa girer
  → 5s kayan pencere (WINDOW_SAMPLES=110250)
  → Her SLIDE_SAMPLES=22050 (1s) inference tetiklenir
  → AirportNoiseSystem.classify_chunk_live()
      → (RMS Norm YOK — kaldırıldı)
      → EfficientNet / CNN / SVM
      → CMVN → PRIOR_WEIGHTS düzeltmesi
      → Confidence Threshold (< 0.45 → UNKNOWN)
      → {label, probs, db_rms}
  → MicrophoneWorker: Majority Voting (son 5 tahmin)
  → GUI sinyal (result_signal) ile güncellenir
```

---

## 7. Bu Sohbette Yapılan Değişiklikler — v4.1

### 7.1 Sorun Analizi

**Train-Serve Skew (kök neden):**
- Eğitimde: izole, tek sınıflı temiz klip → RMS norm → Mel → CMVN
- Canlıda: karışık ortam sesi → chunk üzerinde RMS norm → Mel → CMVN
- İki RMS norm işlemi birbirinin eşdeğeri değil → AIRCRAFT'ın enerji özelliği siliniyor

**OTHER sınıfının yan etkisi:**
- OTHER eklendikten sonra SPEECH tamamen bozuldu
- Model canlıda gürültülü/karışık sesleri "belirsiz = OTHER" olarak öğrendi
- Raw prob SPEECH ≈ 0.0, AIRCRAFT aşırı tahmin

---

### 7.2 noise_detector.py — Değişiklik A: RMS Normalizasyonu Kaldırıldı

`_infer_efficientnet_chunk` içinden şu 5 satır **silindi:**

```python
# SİLİNDİ — bu blok artık yok
rms = np.sqrt(np.mean(chunk ** 2))
if rms > 1e-8:
    chunk = chunk * (0.1 / rms)
chunk = np.clip(chunk, -1.0, 1.0)
```

`with torch.no_grad():` satırı hemen üstte, değişmedi.

---

### 7.3 noise_detector.py — Değişiklik B: Confidence Threshold

`classify_chunk_live` içinde `except` bloğunun altına, `return`'den önce eklendi:

```python
        except Exception as e:
            print(f"[classify_chunk_live] {e}")
            label, probs = "UNKNOWN", {}

        # ── Confidence threshold ──────────────────────────────────────
        if probs:
            max_prob = max(probs.values())
            if max_prob < 0.45:
                label = "UNKNOWN"
        # ─────────────────────────────────────────────────────────────

        return {"label": label, "probs": probs, "db_rms": db_rms}
```

> Eşik ayarı: Çok sık UNKNOWN → 0.40'a indir. Hâlâ yanlış tahmin → 0.50'ye çık.

---

### 7.4 gui_main.py — Değişiklik C: Majority Voting

**Import düzeltmesi** (dosya başında `import deque, counter` satırı silinerek):
```python
from collections import deque, Counter   # Bu satır kalır, diğeri silindi
```

**`__init__` metoduna ekleme** (`self._audio_q = queue.Queue()` altına):
```python
        self._pred_buffer = deque(maxlen=5)   # son 5 tahmin (≈5 saniye)
```

**`run` metodunda inference bloğu** (majority voting ile):
```python
                    s_inf += n
                    if s_inf >= self.SLIDE_SAMPLES:
                        s_inf = 0
                        elapsed = time.time() - start_time
                        try:
                            res = self._system.classify_chunk_live(
                                buffer.copy(), self._model_pref
                            )
                            res["elapsed"] = elapsed
                            res["samples"]  = buffer.copy()

                            # ── Majority voting ───────────────────────
                            self._pred_buffer.append(res["label"])
                            votes    = Counter(self._pred_buffer)
                            smoothed = votes.most_common(1)[0][0]
                            res["raw_label"] = res["label"]   # ham tahmin
                            res["label"]     = smoothed       # yumuşatılmış
                            # ─────────────────────────────────────────

                            self.result_signal.emit(res)
                        except Exception as e:
                            print(f"[Inference] {e}")
```

> `raw_label` alanı debug/loglama için eklendi, GUI'yi bozmaz.
> UNKNOWN tahminler buffer'a dahil edilir — ortam gerçekten karışıksa UNKNOWN kazanması doğru davranış.

---

### 7.5 Google Colab Entegrasyonu

#### dataset_builder_v4.py — Path Bloğu
```python
# ── Ortam algılama: Colab / Yerel ────────────────────────────
import sys
_COLAB = "google.colab" in sys.modules or "COLAB_GPU" in os.environ

if _COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    _LOCAL = os.environ.get("AIRPORT_LOCAL", "")
    if _LOCAL and os.path.exists(_LOCAL):
        PROJECT_ROOT = _LOCAL
        GENERIC_ROOT = os.path.join(_LOCAL, "DATASET")
    else:
        PROJECT_ROOT = "/content/drive/MyDrive/TUBITAK/Airport_Noise"
        GENERIC_ROOT = os.path.join(PROJECT_ROOT, "DATASET")
else:
    PROJECT_ROOT = r"C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise"
    GENERIC_ROOT = r"D:\Downloads_2\DATASET"

AIRPLANE_PATH = os.path.join(PROJECT_ROOT, "Dataset_Airplane")
ESC50_PATH    = os.path.join(PROJECT_ROOT, "Dataset_ESC50")
CACHE_DIR     = os.path.join(PROJECT_ROOT, "cache")
MANIFEST_OUT  = os.path.join(PROJECT_ROOT, "cache", "manifest_v4.csv")
# ─────────────────────────────────────────────────────────────
```

#### train_efficientnet.py — Path Bloğu
```python
# ── Ortam algılama: Colab / Yerel ────────────────────────────
import sys
_COLAB = "google.colab" in sys.modules or "COLAB_GPU" in os.environ

if _COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    PROJECT_ROOT = "/content/drive/MyDrive/TUBITAK/Airport_Noise"
else:
    PROJECT_ROOT = r"C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise"

MANIFEST_CSV = os.path.join(PROJECT_ROOT, "cache", "manifest_v4.csv")
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR    = os.path.join(PROJECT_ROOT, "outputs", "training_efficientnet")

num_workers = 4 if _COLAB else 0
# ─────────────────────────────────────────────────────────────
```

#### train_efficientnet.py — DataLoader
```python
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=num_workers, pin_memory=_COLAB)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=num_workers, pin_memory=_COLAB)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=num_workers, pin_memory=_COLAB)
```

#### train_efficientnet.py — Manifest Path Düzeltmesi
`pd.read_csv(MANIFEST_CSV)` satırının hemen altına:

```python
df = pd.read_csv(MANIFEST_CSV)

# ── Colab path düzeltmesi ─────────────────────────────────────
if _COLAB:
    _LOCAL = os.environ.get("AIRPORT_LOCAL",
             "/content/drive/MyDrive/TUBITAK/Airport_Noise")

    def _fix_path(p: str) -> str:
        p = p.replace("\\", "/")
        for old in [
            "C:/Users/Fatih/Desktop/TUBITAK/Airport_Noise",
            "D:/Downloads_2",
        ]:
            if old in p:
                return p.replace(old, _LOCAL)
        return p

    df["path"] = df["path"].apply(_fix_path)
    before = len(df)
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)
    print(f"[Path fix] {before} → {len(df)} örnek ({before-len(df)} eksik dosya çıkarıldı)")
# ─────────────────────────────────────────────────────────────
```

#### Colab Notebook Hücre Sırası
```python
# Hücre 1 — Kütüphaneler
!pip install -q librosa audioread soundfile torchvision

# Hücre 2 — Dataset'i Colab yerel diskine kopyala (tek seferlik, ~10 dk)
import shutil, os
BASE  = "/content/drive/MyDrive/TUBITAK/Airport_Noise"
LOCAL = "/content/Airport_Noise"
os.makedirs(LOCAL, exist_ok=True)
for folder in ["Dataset_Airplane", "Dataset_ESC50", "DATASET", "cache", "models"]:
    src, dst = f"{BASE}/{folder}", f"{LOCAL}/{folder}"
    if not os.path.exists(dst):
        print(f"Kopyalanıyor: {folder}...")
        shutil.copytree(src, dst)
        print(f"✓ {folder} tamam")

# Hücre 3 — LOCAL path'i aktif et
import os
os.environ["AIRPORT_LOCAL"] = "/content/Airport_Noise"

# Hücre 4 — Dataset builder
%run /content/drive/MyDrive/TUBITAK/Airport_Noise/dataset_builder_v4.py

# Hücre 5 — Eğitim
%run /content/drive/MyDrive/TUBITAK/Airport_Noise/train_efficientnet.py

# Hücre 6 — Modeli Drive'a kaydet
import shutil
shutil.copy("/content/Airport_Noise/models/best_efficientnet.pt",
            "/content/drive/MyDrive/TUBITAK/Airport_Noise/models/best_efficientnet.pt")
print("✓ Model Drive'a kaydedildi")
```

---

## 8. Mevcut Durum ve Bilinen Sorunlar

### Çalışan Özellikler
- Faz 1 dosya analizi tam çalışıyor
- Faz 2 mikrofon akışı çalışıyor (CUDA ile)
- 6 sınıflı EfficientNet v4.1 eğitildi (RMS norm kaldırıldı)
- Confidence threshold aktif (eşik: 0.45)
- Majority voting aktif (buffer: 5 tahmin)
- Colab eğitimi çalışıyor
- Etiketleme sistemi çalışıyor

### Bilinen Sorunlar

**1. AIRCRAFT Canlı Performansı — Orta Öncelik**
Confusion matrix'te recall=0.69. 59 örnek OTHER'a, 34'ü AMBIENT'e kaçıyor.
Nedeni: AeroSonicDB düşük enerjili uzak mesafe kayıtlar + OTHER sınıfının "belirsiz ses" kovası haline gelmesi.
Geçici çözüm: `PRIOR_WEIGHTS["AIRCRAFT"]` = 0.10 → 0.25 yap.
Kalıcı çözüm: Gerçek ortam fine-tune (bkz. Bölüm 9).

**2. Domain Shift — Kritik (çözüm planlandı)**
Model studio/lab kayıtlarıyla eğitildi. Gerçek mikrofon farklı akustik özellik taşıyor.
Confidence threshold ve majority voting kısmen telafi ediyor.
Kalıcı çözüm: fine-tune.

**3. Annotation Marker Layout Sorunu (Faz 1) — Düşük Öncelik**
Şerit grafiğine annotation eklendiğinde yükseklik zaman zaman küçülüyor.

---

## 9. Yapılacaklar — Sıradaki Adım

### 🔴 Gerçek Ortam Fine-Tune (en önemli açık madde)

Tüm canlı performans sorunlarının kalıcı çözümü budur. Altyapı hazır.

**Adımlar:**

**1. Veri Toplama**
GUI'deki Faz 2 kayıt sistemiyle hedef ortamda (havalimanı yakını, gerçek mikrofon) ses topla.
Her sınıf için en az 30-50 klip hedefle. Az veri yeterli çünkü backbone dondurulacak.

**2. Onaylama**
DataReviewTab → Dinle → Onayla → `live_clips/approved/`
`approved_manifest.csv` otomatik güncellenir.

**3. dataset_builder_v5.py Yaz**
```python
# approved_manifest.csv + manifest_v4.csv birleştir
approved = pd.read_csv("live_clips/approved_manifest.csv")
base     = pd.read_csv("cache/manifest_v4.csv")
combined = pd.concat([base, approved]).reset_index(drop=True)
combined.to_csv("cache/manifest_v5.csv", index=False)
```

**4. Fine-Tune Eğitimi**
`train_efficientnet.py`'de backbone'u tamamen dondur, yalnızca son classifier katmanını eğit:

```python
# Freeze bloğunda tüm backbone dondur
for param in model.parameters():
    param.requires_grad = False

# Sadece classifier açık
for param in model.classifier.parameters():
    param.requires_grad = True

# Küçük learning rate
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

# 5-10 epoch yeterli
EPOCHS = 8
```

**5. Checkpoint**
Fine-tune sonrası modeli `models/best_efficientnet_finetune.pt` olarak kaydet.
Önce test et, başarılı olursa `best_efficientnet.pt` ile değiştir.

---

### 🟡 AIRCRAFT Prior Geçici Düzeltmesi

Fine-tune beklerken `noise_detector.py` içinde:
```python
PRIOR_WEIGHTS = {
    "AIRCRAFT": 0.25,   # 0.10'dan değiştirildi
    ...
}
```

---

### 🟡 Orta Öncelik

- [ ] Mikrofon cihaz listesi yenileme butonu (`gui_main.py` — `sd._terminate()` + `sd._initialize()`)
- [ ] Annotation marker layout sorunu kalıcı çözüm (`gui_main.py`)

---

## 10. Sonraki Sohbet İçin Bağlam

### Devam Ederken

1. Bu dokümanı (`PROJE_DOKUMANTASYON_v11.md`) paylaş
2. İlgili kod dosyalarını paylaş — Claude ilk mesajda hangileri gerektiğini belirtir
3. Fine-tune için `train_efficientnet.py` ve `dataset_builder_v4.py` gerekecek

### Temel Kurallar
- `manifest_v3.csv` ve `features_v3.pkl`'ye **dokunma** — yedek
- `num_workers=0` Windows'ta kalmalı, Colab'da 4
- `dataset_builder_v4.py` çalıştırılmadan önce `features_v4.pkl` silinmeli (normalizasyon değişirse)
- GPU değişiklikleri yapıldı, `map_location="cpu"` satırları **yok**
- `import deque, counter` satırı gui_main.py'de **yok** — `from collections import deque, Counter` kullanılıyor

### Hızlı Başvuru — Temel Parametreler
```python
SR              = 22050    # Örnekleme frekansı
WINDOW          = 5.0      # Pencere süresi (saniye)
HOP_SEC         = 2.5      # Pencere atlama (%50 overlap)
SLIDE_SAMPLES   = 22050    # Her 1s inference tetiklenir
N_FFT           = 2048
HOP_FFT         = 512
N_MELS          = 128
EFF_IMG         = 224      # EfficientNet giriş boyutu
FEAT_DIM        = 264      # SVM özellik vektörü boyutu
BATCH_SIZE      = 32       # EfficientNet eğitim batch boyutu
CONF_THRESHOLD  = 0.45     # Confidence threshold (UNKNOWN sınırı)
VOTING_BUFFER   = 5        # Majority voting penceresi (tahmin sayısı)
```

### v4.1'de Değişen / Eklenen
| Bileşen | Değişiklik |
|---------|-----------|
| `noise_detector.py` | RMS norm kaldırıldı (`_infer_efficientnet_chunk`) |
| `noise_detector.py` | Confidence threshold eklendi (`classify_chunk_live`) |
| `gui_main.py` | Majority voting eklendi (`MicrophoneWorker`) |
| `gui_main.py` | `import deque, counter` satırı düzeltildi |
| `dataset_builder_v4.py` | Colab/yerel çift ortam path bloğu |
| `train_efficientnet.py` | Colab/yerel çift ortam path bloğu |
| `train_efficientnet.py` | `num_workers` dinamik, `pin_memory=_COLAB` |
| `train_efficientnet.py` | Manifest Windows→Colab path düzeltme bloğu |
