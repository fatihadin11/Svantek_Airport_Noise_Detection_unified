# ✈ Havalimanı Çevresel Gürültü Tespit Sistemi
**Airport Environmental Noise Detection System**

> **Durum:** Aktif Geliştirme — v5.0
> **Son Güncelleme:** Haziran 2026
> **Mimari:** PyQt6 GUI + Çok Modelli ML (EfficientNet-B0 / CNN / SVM) + Canlı Mikrofon + Etiketleme + Modern Mimari Kanalı (geliştiriliyor)

---

## İçindekiler

1. [Proje Özeti](#1-proje-özeti)
2. [Klasör Yapısı](#2-klasör-yapısı)
3. [Mimari — Temel Modüller](#3-mimari--temel-modüller)
4. [ML Modelleri — Mevcut Sistem](#4-ml-modelleri--mevcut-sistem)
5. [Eğitim Pipeline](#5-eğitim-pipeline)
6. [Veri Akışı](#6-veri-akışı)
7. [Tüm Değişiklik Geçmişi](#7-tüm-değişiklik-geçmişi)
8. [Mevcut Durum ve Bilinen Sorunlar](#8-mevcut-durum-ve-bilinen-sorunlar)
9. [Yapılacaklar](#9-yapılacaklar)
10. [Yeni Mimari — Modern Kanal (BEATs/CLAP)](#10-yeni-mimari--modern-kanal-beatsclap)
11. [Sonraki Sohbet İçin Kurallar ve Bağlam](#11-sonraki-sohbet-için-kurallar-ve-bağlam)

---

## 1. Proje Özeti

Havalimanı yakınındaki ortam seslerini gerçek zamanlı olarak sınıflandıran bir sistemdir. Uçak gürültüsünü diğer çevresel seslerden ayırt etmek ve etiketlenmiş canlı verilerle modeli sürekli iyileştirmek temel hedeftir.

**Sınıflar:** `AIRCRAFT` | `AMBIENT` | `OTHER` | `SPEECH` | `TRAFFIC` | `WIND`

> `OTHER` v4.0 ile eklendi — modelin tanımadığı sesleri AIRCRAFT'a yanlış atama sorununu çözmek için.

**Çalışma modları:**
- **Faz 1 — Dosya Analizi:** WAV/MP3/FLAC yükle, analiz et, görselleştir, annotation ekle
- **Faz 2 — Canlı Mikrofon:** Gerçek zamanlı ses akışı, sınıflandırma, kayıt ve etiketleme

**Sınıf Renkleri:**
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

## 2. Klasör Yapısı

```
C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise\          ← PROJECT_ROOT
│
├── noise_detector.py          # Tüm ML modelleri + ses işleme motoru
├── gui_main.py                # PyQt6 arayüzü (tek dosya)
├── mic_map.py                 # Harita sekmesi (opsiyonel, MapTab)
│
├── dataset_builder.py         # Aktif dataset hazırlama scripti (v4+v5 birleşik)
├── train_efficientnet.py      # EfficientNet eğitim scripti
├── train_cnn.py               # CNN eğitim scripti
├── env_audio_processor.py     # AMBIENT klip üretici
│
├── models/
│   ├── best_efficientnet.pt            # EfficientNet-B0 checkpoint (6 sınıf)
│   ├── efficientnet_label_encoder.pkl
│   ├── efficientnet_meta.pkl
│   ├── best_efficientnet_finetune.pt   # Fine-tune sonrası (henüz yok)
│   ├── best_cnn.pt
│   ├── cnn_label_encoder.pkl
│   ├── best_model.pkl                  # SVM
│   └── label_encoder.pkl
│
├── cache/
│   ├── manifest_v3.csv        # ESKİ — 5 sınıf — DOKUNMA
│   ├── manifest_v4.csv        # Temel veri seti — 26079 örnek
│   ├── manifest_v5.csv        # v4 + live klipler — AKTİF
│   ├── features_v3.pkl        # ESKİ önbellek — DOKUNMA
│   ├── features_v4.pkl        # v4 SVM önbelleği
│   └── features_v5.pkl        # v4 + live — AKTİF (live varsa kullan)
│
├── Dataset_Airplane/          # AeroSonicDB + env_audio_manifest.csv
├── Dataset_ESC50/             # ESC-50 + esc50.csv
│
├── outputs_gui/               # Faz 1 dosya analizi çıktıları (PNG, CSV)
├── outputs/
│   └── training_efficientnet/ # Eğitim grafikleri, confusion matrix PNG'leri
└── outputs_mic/               # Faz 2 oturum kayıtları
    ├── session_YYYYMMDD_HHMMSS.wav
    └── session_YYYYMMDD_HHMMSS.csv
```

**Harici Diskler:**
```
D:\Downloads_2\DATASET\                ← GENERIC_AUDIO_CLASSIFIER
├── Animals\   (CATS, DOGS, ELEPHANT, HORSE, LIONS → OTHER)
├── Birds\     (CROWS, PARROT, PEACOCK, SPARROW → OTHER)
├── Environment\ (CROWD→SPEECH, MILITARY→OTHER, OFFICE→AMBIENT,
│                 RAINFALL→AMBIENT, TRAFFIC→TRAFFIC, WIND→WIND)
└── Vehicles\  (airplane,helicopter→AIRCRAFT | car,bus,truck→TRAFFIC)

D:\Airport_Live_Clips\                 ← CANLI KLİP DEPOSU (v5.0'da taşındı)
├── pending\
│   ├── AIRCRAFT\
│   ├── AMBIENT\
│   ├── OTHER\
│   ├── SPEECH\
│   ├── TRAFFIC\
│   └── WIND\
├── approved\    (aynı yapı)
├── rejected\    (aynı yapı)
├── pending_manifest.csv
└── approved_manifest.csv
```

**Google Drive (Colab için):**
```
/content/drive/MyDrive/TUBITAK/Airport_Noise/
├── DATASET\
├── Dataset_Airplane\
├── Dataset_ESC50\
├── cache\
├── models\
└── outputs\
```

---

## 3. Mimari — Temel Modüller

### `noise_detector.py` Sınıfları

| Sınıf | Görev |
|---|---|
| `AudioLoader` | WAV/MP3/FLAC yükle, librosa ile 22050 Hz mono normalize |
| `AudioAnalyzer` | Waveform, STFT, Mel spektrogram, dBFS/zaman |
| `FeatureExtractor` | 264-boyutlu ML özellik vektörü (MFCC×3, chroma, SC, ZCR, RMS…) |
| `NoiseFilter` | Band-pass, spectral gating, Wiener filtre |
| `NoiseClassifier` | Kural tabanlı sınıflandırıcı (yedek, ML yoksa devreye girer) |
| `Visualizer` | Dashboard PNG, pasta grafik, waveform, Mel, MFCC |
| `AirportNoiseSystem` | Tüm sınıfları orkestre eden ana sınıf |
| `AirportCNN` | 3-katmanlı CNN mimarisi (PyTorch) |
| `EfficientNetAirport` | EfficientNet-B0 transfer learning (torchvision) |

### `gui_main.py` Widget Sınıfları

| Sınıf | Görev |
|---|---|
| `MainWindow` | Ana pencere, sekmeler, dosya açma, analiz |
| `SidePanel` | Faz 1 istatistik paneli, sınıf dağılım barları |
| `ClassificationTab` | Sınıflandırma şeridi, dB grafiği, softmax eğrileri |
| `SpectrogramTab` | 128-Mel spektrogram görüntüleyici |
| `FeaturesTab` | ZCR, RMS, Spectral Centroid grafikleri |
| `MicrophoneTab` | Canlı mikrofon, VU metre, rolling strip, etiketleme |
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

## 4. ML Modelleri — Mevcut Sistem

### Model Hiyerarşisi (öncelik sırası)
```
EfficientNet-B0  →  CNN  →  SVM  →  Kural Tabanlı (yedek)
```

### Temel Parametreler
```python
SR              = 22050    # Örnekleme frekansı (Hz)
WINDOW          = 5.0      # Pencere süresi (saniye)
HOP_SEC         = 2.5      # Pencere atlama (%50 overlap)
SLIDE_SAMPLES   = 22050    # Her 1s inference tetiklenir
N_FFT           = 2048
HOP_FFT         = 512
N_MELS          = 128
EFF_IMG         = 224      # EfficientNet giriş boyutu
FEAT_DIM        = 264      # SVM özellik vektörü boyutu
BATCH_SIZE      = 32
CONF_THRESHOLD  = 0.45     # Confidence threshold (UNKNOWN sınırı)
VOTING_BUFFER   = 5        # Majority voting penceresi (tahmin sayısı)
```

### EfficientNet-B0 Pipeline (eğitim ve tüm inference yollarında aynı)

```
librosa.load (SR=22050, mono)
  → center-crop veya zero-pad → 5s (110250 sample)
  → torch.FloatTensor
  → torchaudio.MelSpectrogram (n_fft=2048, hop=512, n_mels=128)
  → AmplitudeToDB (top_db=80)
  → CMVN: mean = spec.mean(dim=-1, keepdim=True)
           std  = spec.std(dim=-1, keepdim=True) + 1e-8
           spec = (spec - mean) / std
  → min-max [0,1]: spec = (spec - s_min) / (s_max - s_min)
  → spec.repeat(3,1,1)  →  RGB 3 kanal
  → Resize 224×224 + ImageNet normalize
  → EfficientNet-B0 → 6 sınıf softmax
```

> **KRİTİK:** CMVN her üç yolda (eğitim, dosya analizi `_classify_efficientnet`,
> canlı `_infer_efficientnet_chunk`) birebir aynı uygulanmalıdır.
> v5.0'da `_classify_efficientnet`'e CMVN eklendi — artık tutarlı.

### Prior Ağırlıkları (güncel)
```python
PRIOR_WEIGHTS = {
    "AIRCRAFT": 0.25,   # 0.10'dan yükseltildi (fine-tune öncesi geçici düzeltme)
    "SPEECH":   3.5,
    "TRAFFIC":  2.0,
    "AMBIENT":  2.0,
    "WIND":     2.0,
    "OTHER":    1.5,
}
```

### Son Eğitim Sonuçları (v4.1 — RMS norm kaldırıldıktan sonra)
```
AIRCRAFT : Recall=0.69  ← iyileştirme gerekiyor (59 örnek OTHER'a, 34'ü AMBIENT'e kaçıyor)
AMBIENT  : Recall=0.93
OTHER    : Recall=0.97
SPEECH   : Recall=0.94
TRAFFIC  : Recall=0.95
WIND     : Recall=0.90
```

---

## 5. Eğitim Pipeline

### `dataset_builder.py` Veri Kaynakları

```
load_esc50_records()      → ESC-50 (~560 örnek)
load_aerosonic_records()  → AeroSonicDB (~1895 AIRCRAFT WAV)
load_ambient_records()    → env_audio (~322 AMBIENT)
load_generic_records()    → D:\Downloads_2\DATASET (~23302 örnek)
load_live_records()       → D:\Airport_Live_Clips\approved_manifest.csv (CANLI klipler)

→ base_records + live_records  →  manifest_v5.csv (live varsa)
                                →  manifest_v4.csv (live yoksa)
→ features_v5.pkl              (SVM için, live varsa)
→ features_v4.pkl              (SVM için, live yoksa)
```

**Cache path mantığı (`__main__` bloğunda):**
```python
if live_records:
    cache_path = os.path.join(CACHE_DIR, "features_v5.pkl")
else:
    cache_path = os.path.join(CACHE_DIR, "features_v4.pkl")
```

### `train_efficientnet.py` — Manifest Path
```python
# v5 varsa onu kullan, yoksa v4
MANIFEST_CSV = os.path.join(PROJECT_ROOT, "cache", "manifest_v5.csv")
```

### `dataset_builder.py` Sabitler (ayarlar bloğu)
```python
LIVE_CLIPS_DIR    = r"D:\Airport_Live_Clips"
APPROVED_MANIFEST = os.path.join(LIVE_CLIPS_DIR, "approved_manifest.csv")
MANIFEST_OUT_V5   = os.path.join(PROJECT_ROOT, "cache", "manifest_v5.csv")
```

### SpecAugment (train_efficientnet.py içinde, sadece eğitimde)
```python
if self.augment and random.random() > 0.5:
    # Frequency masking
    f_mask = int(random.uniform(0, 0.15) * spec.shape[1])
    f0     = random.randint(0, spec.shape[1] - f_mask)
    spec[:, f0:f0+f_mask, :] = 0
    # Time masking
    t_mask = int(random.uniform(0, 0.15) * spec.shape[2])
    t0     = random.randint(0, spec.shape[2] - t_mask)
    spec[:, :, t0:t0+t_mask] = 0
```

### 2 Aşamalı EfficientNet Eğitimi
```
Aşama 1 — Freeze:      Backbone donduruldu, sadece classifier eğitildi
                       LR=1e-3, 10 epoch
Aşama 2 — FineTune:    Son 3 blok açıldı
                       LR=2e-5, 20 epoch
```

### Fine-Tune (Gerçek Ortam Uyarlaması — Sıradaki Adım)
```python
# Tüm backbone dondur, sadece classifier aç
for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)
EPOCHS = 8   # 5-10 epoch yeterli
```

### Colab Notebook Hücre Sırası
```python
# Hücre 1
!pip install -q librosa audioread soundfile torchvision

# Hücre 2 — Dataset'i Colab diskine kopyala (tek seferlik, ~10 dk)
import shutil, os
BASE  = "/content/drive/MyDrive/TUBITAK/Airport_Noise"
LOCAL = "/content/Airport_Noise"
os.makedirs(LOCAL, exist_ok=True)
for folder in ["Dataset_Airplane", "Dataset_ESC50", "DATASET", "cache", "models"]:
    src, dst = f"{BASE}/{folder}", f"{LOCAL}/{folder}"
    if not os.path.exists(dst):
        shutil.copytree(src, dst)

# Hücre 3
os.environ["AIRPORT_LOCAL"] = "/content/Airport_Noise"

# Hücre 4
%run /content/drive/MyDrive/TUBITAK/Airport_Noise/dataset_builder.py

# Hücre 5
%run /content/drive/MyDrive/TUBITAK/Airport_Noise/train_efficientnet.py

# Hücre 6 — Modeli kaydet
shutil.copy("/content/Airport_Noise/models/best_efficientnet.pt",
            "/content/drive/MyDrive/TUBITAK/Airport_Noise/models/best_efficientnet.pt")
```

---

## 6. Veri Akışı

### Faz 1 — Dosya Analizi
```
Dosya seçimi (WAV/MP3/FLAC)
  → AudioLoader → 22050 Hz mono
  → Pencereye böl (5s, %50 overlap)
  → _classify_efficientnet / _classify_cnn / _classify_svm
      → CMVN → min-max → RGB → EfficientNet   ← v5.0'da CMVN EKLENDİ
      → PRIOR_WEIGHTS düzeltmesi
      → Confidence Threshold (< 0.45 → UNKNOWN)
  → Görselleştirme (Mel, waveform, şerit, pasta)
  → Opsiyonel annotation (AnnotationDialog)
```

### Faz 2 — Canlı Mikrofon
```
sounddevice InputStream (SR=22050, blocksize=1024)
  → BLOCK_SIZE=1024 chunk'lar kuyruğa girer
  → İki ayrı rolling buffer:
      infer_buffer  = 5s  (WINDOW_SAMPLES = 110250)   → inference için
      clip_buffer   = 10s (CLIP_BUFFER_SAMPLES = 220500) → klip kaydetmek için
  → Her SLIDE_SAMPLES=22050 (1s) inference tetiklenir:
      → AirportNoiseSystem.classify_chunk_live(infer_buffer)
          → EfficientNet → CMVN → PRIOR_WEIGHTS
          → Confidence Threshold (< 0.45 → UNKNOWN)
      → MicrophoneWorker: Majority Voting (son 5 tahmin)
      → result_signal yayınla (label, probs, db_rms, samples=clip_buffer)
  → "Etiketle & Gönder":
      → son 5s clip_buffer'dan alınır
      → D:\Airport_Live_Clips\pending\<SINIF>\ kaydedilir
      → pending_manifest.csv güncellenir
```

### Klip Onaylama Akışı
```
GUI → DataReviewTab → Dinle → Onayla
  → D:\Airport_Live_Clips\approved\<SINIF>\
  → approved_manifest.csv güncellenir
  → dataset_builder.py çalıştır → manifest_v5.csv
  → train_efficientnet.py → fine-tune
```

---

## 7. Tüm Değişiklik Geçmişi

### v4.0 Değişiklikleri
| Bileşen | Değişiklik |
|---|---|
| `noise_detector.py` | OTHER sınıfı eklendi (6. sınıf) |
| `dataset_builder.py` | GENERIC_AUDIO_CLASSIFIER entegrasyonu |
| `cache/` | manifest_v4.csv, features_v4.pkl üretildi |

### v4.1 Değişiklikleri
| Bileşen | Değişiklik |
|---|---|
| `noise_detector.py` | `_infer_efficientnet_chunk` — RMS norm kaldırıldı |
| `noise_detector.py` | `classify_chunk_live` — Confidence threshold (0.45) eklendi |
| `gui_main.py` | `MicrophoneWorker` — Majority voting eklendi (deque maxlen=5) |
| `gui_main.py` | `from collections import deque, Counter` düzeltildi |
| `dataset_builder.py` | Colab/yerel çift ortam path bloğu |
| `train_efficientnet.py` | Colab/yerel çift ortam path bloğu, manifest path fix |

### v5.0 Değişiklikleri (Bu Sohbet)
| Bileşen | Değişiklik | Durum |
|---|---|---|
| `noise_detector.py` | `_classify_efficientnet`'e CMVN eklendi (3 satır) | ✅ Yapıldı |
| `noise_detector.py` | `_infer_efficientnet_chunk` CMVN doğrulandı — doğru | ✅ Onaylandı |
| `noise_detector.py` | `PRIOR_WEIGHTS["AIRCRAFT"]` 0.10 → 0.25 | ✅ Yapıldı |
| `gui_main.py` | `PendingClipManager` — klip dizini D'ye taşındı | ✅ Yapıldı |
| `gui_main.py` | `MicrophoneWorker` — 10s `clip_buffer` eklendi (uzun konuşmalar) | ✅ Yapıldı |
| `dataset_builder.py` | `load_live_records()` fonksiyonu eklendi | ✅ Yapıldı |
| `dataset_builder.py` | `import csv` eklendi | ✅ Yapıldı |
| `dataset_builder.py` | Cache path mantığı (features_v5.pkl live varsa) | ✅ Yapıldı |
| `dataset_builder.py` | LIVE_CLIPS_DIR, APPROVED_MANIFEST sabitleri eklendi | ✅ Yapıldı |

### v5.0 Kritik Bug Analizi (domain shift araştırması)

Bu sohbette şu analiz yapıldı:

**Yanlış bulunan sonra düzeltilen:** `_infer_efficientnet_chunk` CMVN'nin training'de olmadığı düşünüldü → train_efficientnet.py incelenince her iki yerde de CMVN olduğu doğrulandı. Canlı inference pipeline eğitimle tutarlı.

**Gerçek bug:** `_classify_efficientnet` (dosya analizi yolu) CMVN içermiyordu → eklendi.

**Gerçek domain shift kaynakları (pipeline değil):**
- Mikrofon frekans yanıtı ≠ internet kayıt zinciri (CMVN kısmen telafi ediyor)
- Ortam yankısı/reverb — eğitim setinde yok
- Arka plan gürültüsü — SpecAugment kısmen telafi ediyor
- Partial window — uçak/konuşma pencere ortasında başlıyorsa CMVN istatistikleri kayıyor

**Çözüm:** Gerçek ortam fine-tuning (sıradaki adım).

---

## 8. Mevcut Durum ve Bilinen Sorunlar

### Çalışan Özellikler
- Faz 1 dosya analizi tam çalışıyor
- Faz 2 mikrofon akışı çalışıyor
- 6 sınıflı EfficientNet v4.1 eğitildi
- Confidence threshold aktif (0.45)
- Majority voting aktif (buffer: 5 tahmin)
- Colab eğitimi çalışıyor
- Etiketleme + onaylama sistemi çalışıyor
- D:\Airport_Live_Clips klip deposu aktif
- manifest_v5.csv üretimi çalışıyor

### Bilinen Sorunlar

**1. AIRCRAFT Canlı Performansı — Orta Öncelik**
Recall=0.69. 59 örnek OTHER'a, 34'ü AMBIENT'e kaçıyor.
Geçici çözüm: PRIOR_WEIGHTS["AIRCRAFT"] = 0.25 yapıldı.
Kalıcı çözüm: Fine-tuning.

**2. Domain Shift — Fine-Tune Planlandı**
Gerçek mikrofon verileriyle fine-tune bekleniyor.
Önce SPEECH sınıfından başlanacak.

**3. Annotation Marker Layout (Faz 1) — Düşük Öncelik**
Şerit grafiğine annotation eklendiğinde yükseklik zaman zaman küçülüyor.

---

## 9. Yapılacaklar

### 🔴 Gerçek Ortam Fine-Tune (en önemli açık madde)

**Adımlar:**
1. GUI Faz 2 → Kayıt Başlat → Konuş → Etiketle & Gönder
2. DataReviewTab → Dinle → Onayla → approved_manifest.csv güncellenir
3. `dataset_builder.py` çalıştır → manifest_v5.csv + features_v5.pkl
4. `train_efficientnet.py` MANIFEST_CSV = manifest_v5.csv yap
5. Fine-tune eğit (8 epoch, lr=1e-4, sadece classifier)
6. Test et → başarılıysa `best_efficientnet.pt` ile değiştir

**Öncelik sırası:** SPEECH → AIRCRAFT → diğerleri

**Klip toplama stratejisi:**
- Her sınıf için 30-50 klip minimum
- Farklı mesafe ve gain ayarlarıyla kaydet
- Hem "pencere dolu" hem "pencere ortasında başlayan" klipler topla
- Kısa sessizlik + ses geçişleri içeren klipler ekle (partial window öğrenimi)

### 🟡 Modern Mimari Kanalı (BEATs) — Paralel Geliştirme
Bkz. Bölüm 10.

### 🟡 Orta Öncelik
- [ ] Mikrofon cihaz listesi yenileme butonu
- [ ] Annotation marker layout sorunu

---

## 10. Yeni Mimari — Modern Kanal (BEATs/CLAP)

### Genel Strateji

Mevcut EfficientNet sistemi korunur. GUI'deki model seçim combo box'ına yeni bir seçenek eklenir. Bu seçenek BEATs/CLAP tabanlı modern kanalı aktive eder. İki sistemin softmax output'u ensemble edilir.

```python
# gui_main.py MicrophoneTab içindeki MODEL_MAP'e eklenecek
MODEL_MAP = {
    "Otomatik (EfficientNet → CNN → SVM)": "auto",
    "EfficientNet-B0": "efficientnet",
    "CNN": "cnn",
    "SVM": "svm",
    "BEATs (Modern)": "beats",         # ← YENİ
    "Ensemble (EfficientNet + BEATs)": "ensemble",  # ← YENİ
}
```

### Modern Sistemin Mimarisi

```
Mikrofon (22050 Hz, 5s pencere)
         │
         ▼
    [VAD Filtresi]
    Silero-VAD veya WebRTC-VAD
    Sessiz pencereler → AMBIENT / atla
         │
         ▼
  ┌──────────────────────────────────┐
  │  Kanal A: Mevcut EfficientNet   │
  │  (pipeline değişmedi)           │
  │  → 6-sınıf softmax              │
  └──────────────────────────────────┘
         +
  ┌──────────────────────────────────┐
  │  Kanal B: BEATs-Small Encoder   │
  │  (frozen) → 768-dim embedding   │
  │  → Lightweight MLP (2-layer)    │
  │  → 6-sınıf softmax              │
  └──────────────────────────────────┘
         │
         ▼
    [Ensemble Layer]
    softmax_A * α + softmax_B * (1-α)
    α = 0.5 başlangıç (tune edilecek)
         │
         ▼
    [Temporal Smoother]
    Probability averaging (deque maxlen=10)
    + Event duration filter (min 2s)
         │
         ▼
       Output
```

### BEATs Entegrasyon Planı

**Model:** `microsoft/BEATs` — BEATs_iter3_plus_AS2M.pt (fine-tune versiyonu)
**Kullanım:** Frozen encoder — sadece embedding çıkarımı
**Classifier:** 768 → 256 → 6 (ReLU, Dropout 0.3)
**Eğitim:** approved live clips üzerinde

```python
# noise_detector.py'ye eklenecek yeni sınıf
class BEATsClassifier:
    def __init__(self, model_path: str, label_encoder_path: str):
        self.model = BEATs(...)          # microsoft/unilm'dan
        self.model.load_state_dict(...)
        self.model.eval()
        # Encoder dondur
        for param in self.model.parameters():
            param.requires_grad = False
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 6)
        )

    def classify(self, audio_chunk: np.ndarray) -> dict:
        with torch.no_grad():
            y_t = torch.FloatTensor(audio_chunk).unsqueeze(0)
            embedding = self.model.extract_features(y_t)[0]
            embedding = embedding.mean(dim=1)  # temporal pooling
            logits = self.classifier(embedding)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        return {cls: float(p) for cls, p in zip(self.classes_, probs)}
```

### Temporal Smoother (Mevcut majority voting'in gelişmiş versiyonu)

```python
class TemporalSmoother:
    def __init__(self, buffer_size=10, min_event_duration=2.0):
        self.prob_buffer = deque(maxlen=buffer_size)
        self.min_event_duration = min_event_duration
        self.current_event = None
        self.event_start = None

    def update(self, probs: dict, timestamp: float) -> str:
        self.prob_buffer.append(probs)
        avg_probs = {cls: np.mean([p.get(cls, 0) for p in self.prob_buffer])
                     for cls in probs}
        raw_label = max(avg_probs, key=avg_probs.get)

        if raw_label != self.current_event:
            if self.event_start is None:
                self.event_start = timestamp
            elif timestamp - self.event_start >= self.min_event_duration:
                self.current_event = raw_label
                self.event_start = None
        else:
            self.event_start = None

        return self.current_event or raw_label
```

### Fingerprint Modülü (Terminal anonsları ve alarm sesleri için)

```python
class AudioFingerprinter:
    """
    Shazam benzeri — sabit tonlar ve tekrarlayan sesler için.
    Kullanım: terminal anonsları, alarm sesleri, bagaj sistemi.
    Konuşma, uçak, rüzgar için KULLANMA.
    """
    def match(self, audio_chunk: np.ndarray,
              threshold: float = 0.85) -> str | None:
        # constellation map + hash eşleşmesi
        ...
```

```python
# Ensemble inference akışı
def classify_ensemble(audio_chunk):
    # 1. Fingerprint — hızlı, kesin (sadece bazı sınıflar)
    fp = fingerprinter.match(audio_chunk, threshold=0.85)
    if fp:
        return fp, 1.0

    # 2. EfficientNet
    probs_eff = efficientnet.classify(audio_chunk)

    # 3. BEATs
    probs_beats = beats_classifier.classify(audio_chunk)

    # 4. Ensemble
    ensemble = {cls: 0.5 * probs_eff.get(cls, 0) + 0.5 * probs_beats.get(cls, 0)
                for cls in probs_eff}
    label = max(ensemble, key=ensemble.get)
    return label, ensemble
```

### Geliştirme Öncelik Sırası

1. BEATs-Small'u yükle ve embedding kalitesini test et (offline)
2. Lightweight MLP'yi approved live clips üzerinde eğit
3. GUI'ye "BEATs (Modern)" seçeneğini ekle
4. Offline karşılaştırma: EfficientNet vs BEATs vs Ensemble
5. Ensemble katsayısını (α) validation set üzerinde optimize et
6. Fingerprint modülünü terminal anonsları için ekle (opsiyonel)

### Araştırma Notu

Bu hibrit yapı (mevcut CNN + Foundation Model Embedding + Fingerprint) konferans makalesi için güçlü bir katkı. Baseline olarak mevcut EfficientNet sistemi kullanılabilir, karşılaştırma tablosu hazırlanabilir. Domain shift azaltma etkisi özellikle ölçülmeli.

---

## 11. Sonraki Sohbet İçin Kurallar ve Bağlam

### Bu Dokümanı Nasıl Kullanacaksın

1. Bu dosyayı (`PROJE_DOKUMANTASYON_v12.md`) her yeni sohbetin başında paylaş
2. İlgili kod dosyalarını paylaş — hangileri gerektiği belirtilir
3. İstenen değişiklik için ilgili dosyanın tamamını ya da ilgili fonksiyonu iste

### Claude'dan Kod İsteme Kuralları

**Değişiklik istendiğinde:**
- Önce ilgili kod dosyasını paylaş
- "Şu değişikliği yap" veya "Şu özelliği ekle" diye belirt
- Claude önce değiştirilecek satırı/bloğu gösterecek (eski → yeni formatında)
- Sonra değişikliğin tam bağlamını verecek (fonksiyonun tamamı veya dosyanın tamamı)

**Her zaman tam kod verilmeli:**
- Kısmi kod bloğu değil, ilgili fonksiyonun veya sınıfın tamamı verilmeli
- Parametre değişirse tüm fonksiyon yeniden yazılmalı
- Birden fazla yerde değişiklik varsa her değişiklik için tam blok ayrı verilmeli

### Temel Kurallar

```
manifest_v3.csv ve features_v3.pkl  → DOKUNMA (yedek)
manifest_v4.csv                     → Temel veri seti (dokunma)
features_v4.pkl                     → Live klip yoksa kullan

num_workers = 0                     → Windows'ta sabit
num_workers = 4                     → Colab'da

dataset_builder.py çalıştırılmadan önce
features_v5.pkl SİLİNMELİ (eğer normalizasyon değiştiyse)

map_location="cpu" satırları YOK — GPU aktif

from collections import deque, Counter  → gui_main.py'de bu şekilde
```

### Hızlı Referans — Dosya → Görev Eşlemesi

| Görev | Dosya |
|---|---|
| Canlı inference değiştirme | `noise_detector.py` → `_infer_efficientnet_chunk` |
| Dosya analizi değiştirme | `noise_detector.py` → `_classify_efficientnet` |
| Veri seti ekleme/değiştirme | `dataset_builder.py` |
| Eğitim parametreleri | `train_efficientnet.py` |
| GUI değişikliği | `gui_main.py` |
| Klip kayıt dizini | `gui_main.py` → `MainWindow.__init__` → `LIVE_CLIPS_DIR` |
| Prior ağırlıkları | `noise_detector.py` → `PRIOR_WEIGHTS` |
| Confidence threshold | `noise_detector.py` → `classify_chunk_live` |

---

### Sistem Promptu — Yeni Mimari Geliştirme (Diğer Sohbet İçin)

Aşağıdaki prompt, BEATs/CLAP tabanlı modern kanalı geliştirmek için yeni bir sohbette kullanılabilir:

---

```
Sen bir ses makine öğrenmesi uzmanısın. Havalimanı akustik analiz sistemi
geliştiriyoruz. Proje dokümantasyonunu paylaşıyorum — önce onu dikkatlice oku.

MEVCUT SİSTEM:
- EfficientNet-B0 + CNN + SVM ensemble
- 6 sınıf: AIRCRAFT, AMBIENT, SPEECH, TRAFFIC, WIND, OTHER
- Pipeline: Mel Spectrogram (SR=22050, N_FFT=2048, HOP=512, N_MELS=128)
  → AmplitudeToDB → CMVN → min-max → RGB 224x224 → EfficientNet
- PyQt6 GUI, canlı mikrofon inference, klip etiketleme sistemi
- Çalışıyor ama domain shift (canlı mikrofon vs eğitim verisi) sorunu var

YENİ EKLENECEK MODÜL:
GUI'deki model seçim combo box'ına "BEATs (Modern)" ve
"Ensemble (EfficientNet + BEATs)" seçeneklerini ekleyeceğiz.

YENİ MODÜLİN GEREKSİNİMLERİ:
1. noise_detector.py'ye BEATsClassifier sınıfı eklenecek
   - microsoft/unilm BEATs-Small encoder (frozen)
   - 768-dim embedding → 2-layer MLP → 6 sınıf
   - Mevcut inference interface ile uyumlu
2. noise_detector.py'ye EnsembleClassifier sınıfı eklenecek
   - EfficientNet + BEATs softmax output'larını birleştirir
   - α katsayısı ayarlanabilir
3. train_beats.py — yeni eğitim scripti
   - approved_manifest.csv üzerinde MLP eğitir
   - BEATs encoder frozen, sadece MLP katmanları eğitilir
4. gui_main.py — MODEL_MAP güncelleme
   - "BEATs (Modern)": "beats"
   - "Ensemble (EfficientNet + BEATs)": "ensemble"

KISITLAR:
- Mevcut EfficientNet inference değişmemeli
- BEATs modeli D:\models\ altına kaydedilmeli
- CPU'da çalışabilmeli (GPU varsa GPU kullanmalı)
- Mevcut CMVN + min-max pipeline BEATs için de uygulanmalı
- Latency hedefi: ≤300ms (CPU'da)

BAŞLARKEN:
1. Önce noise_detector.py'yi paylaşacağım
2. Sonra gui_main.py'yi paylaşacağım
3. Sen önce BEATsClassifier sınıfını yaz, test edelim
4. Sonra eğitim scriptini, sonra GUI entegrasyonunu yapacağız

Bir değişiklik önerdiğinde:
- Hangi dosyanın hangi fonksiyonunu değiştireceğini söyle
- Değişiklik için tam fonksiyon/sınıf kodunu ver (kısmi değil)
```

---

*Doküman: PROJE_DOKUMANTASYON_v12.md | Güncelleme: Haziran 2026*
