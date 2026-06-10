# ✈ Havalimanı Çevresel Gürültü Tespit Sistemi
**Airport Environmental Noise Detection System**

> **Durum:** Aktif Geliştirme — v4.0
> **Son Güncelleme:** Mayıs 2026
> **Mimari:** PyQt6 GUI + Çok Modelli ML (EfficientNet-B0 / CNN / SVM) + Canlı Mikrofon + Etiketleme Sistemi

---

## İçindekiler

1. [Proje Özeti](#1-proje-özeti)
2. [Klasör Yapısı](#2-klasör-yapısı)
3. [Mimari — Temel Modüller](#3-mimari--temel-modüller)
4. [ML Modelleri](#4-ml-modelleri)
5. [GUI — Sekme Rehberi](#5-gui--sekme-rehberi)
6. [Veri Akışı](#6-veri-akışı)
7. [Tamamlanan İşler](#7-tamamlanan-işler)
8. [Mevcut Durum ve Bilinen Sorunlar](#8-mevcut-durum-ve-bilinen-sorunlar)
9. [Yapılacaklar](#9-yapılacaklar)
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
│   ├── manifest_v3.csv        # ESKİ — 5 sınıf (AIRCRAFT/AMBIENT/SPEECH/TRAFFIC/WIND)
│   ├── manifest_v4.csv        # AKTİF — 6 sınıf (+ OTHER), 26079 örnek
│   ├── features_v3.pkl        # ESKİ önbellek
│   └── features_v4.pkl        # AKTİF önbellek (RMS norm + CMVN uygulandı)
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

**Harici Veri Seti (D diski):**
```
D:\Downloads_2\DATASET\
├── Animals\   (CATS, DOGS, ELEPHANT, HORSE, LIONS, CAT, DOG, LION → OTHER)
├── Birds\     (CROWS, PARROT, PEACOCK, SPARROW, CROW → OTHER)
├── Environment\ (CROWD→SPEECH, MILITARY→OTHER, OFFICE→AMBIENT,
│                 RAINFALL→AMBIENT, TRAFFIC→TRAFFIC, WIND→WIND)
└── Vehicles\  (airplane,helicopter→AIRCRAFT | car,bus,truck,bike,bicycle,train→TRAFFIC)
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
| `AnnotationDialog` | Faz 1 annotation (şeride tıkla → etiket düzelt) — 6 sınıf |
| `LiveLabelDialog` | Faz 2 etiketleme (model tahmini göster → kullanıcı etiket seç) — 6 sınıf |
| `PendingClipManager` | WAV klip yönetimi, manifest CSV okuma/yazma |
| `LiveSpectrumWidget` | Gerçek zamanlı FFT spektrum görüntüleyici |
| `VUMeter` | Animasyonlu ses seviyesi metre |
| `RollingClassStrip` | Son 60 saniyelik sınıflandırma geçmişi |
| `DbHistoryWidget` | Son 60 saniyelik dBFS geçmişi |
| `MicrophoneWorker` | QThread — mikrofon akışı + inference |

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

### EfficientNet-B0 (Aktif Model — v4.0)
- 6 sınıf: `AIRCRAFT | AMBIENT | OTHER | SPEECH | TRAFFIC | WIND`
- Giriş: 224×224 RGB Mel Spectrogram
- **CMVN normalizasyonu** uygulanıyor (eğitim ve inference'da)
- 2 aşamalı eğitim: Freeze (backbone donuk) → FineTune (son 3 blok açık)
- Test sonuçları (son eğitim):
  ```
  Accuracy  : 91.5%
  F1 Macro  : 87.3%
  AIRCRAFT  : F1=0.74  Recall=0.66  ← zayıf, iyileştirme gerekiyor
  AMBIENT   : F1=0.89  Recall=0.92
  OTHER     : F1=0.96  Recall=0.96  ← mükemmel
  SPEECH    : F1=0.92  Recall=0.91
  TRAFFIC   : F1=0.96  Recall=0.95
  WIND      : F1=0.76  Recall=0.94
  ```
- Checkpoint: `models/best_efficientnet.pt`

### Prior Ağırlık Düzeltmesi
```python
PRIOR_WEIGHTS = {
    "AIRCRAFT": 0.10,
    "SPEECH":   3.5,
    "TRAFFIC":  2.0,
    "AMBIENT":  2.0,
    "WIND":     2.0,
    "OTHER":    1.5,
}
```

### Sınıf Renkleri (tüm dosyalarda tutarlı)
```python
CLASS_COLORS = {
    "AIRCRAFT": "#FF6B35",  # Turuncu
    "AMBIENT":  "#7EE8A2",  # Yeşil
    "SPEECH":   "#A8DADC",  # Açık mavi
    "TRAFFIC":  "#FFE66D",  # Sarı
    "WIND":     "#4ECDC4",  # Turkuaz
    "OTHER":    "#9E9E9E",  # Gri
    "UNKNOWN":  "#6C757D",  # Koyu gri
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
  └── load_generic_records()     → D:\Downloads_2\DATASET (23302 örnek)
      → cache/manifest_v4.csv   (26079 toplam, 6 sınıf)
      → cache/features_v4.pkl   (SVM için, RMS norm uygulanmış)

train_efficientnet.py
  → MANIFEST_CSV = cache/manifest_v4.csv
  → models/best_efficientnet.pt
```

**Önemli:** `features_v4.pkl` silindikten sonra `dataset_builder_v4.py`
yeniden çalıştırılmalıdır (~33 dakika).

---

## 6. Veri Akışı

### Faz 1 — Dosya Analizi
```
WAV/MP3/FLAC
  → AudioLoader → librosa yükle, 22050 Hz mono
  → AudioAnalyzer → waveform, spektrogram, dBFS
  → AirportNoiseSystem.classify_file()
      → EfficientNet: Mel→CMVN→RGB→224×224→softmax→PRIOR düzelt
      → {label, probs, db_rms} per window
  → GUI: sınıflandırma şeridi, grafikler
```

### Faz 2 — Canlı Mikrofon Inference
```
sounddevice InputStream
  → BLOCK_SIZE=1024 chunk'lar kuyruğa girer
  → 5s kayan pencere (WINDOW_SAMPLES=110250)
  → Her SLIDE_SAMPLES=22050 (1s) inference tetiklenir
  → AirportNoiseSystem.classify_chunk_live()
      → RMS Normalizasyonu (chunk üzerinde)
      → EfficientNet / CNN / SVM
      → CMVN → PRIOR_WEIGHTS düzeltmesi
      → {label, probs, db_rms}
  → GUI sinyal (result_signal) ile güncellenir
```

### Etiketleme Akışı
```
LiveLabelDialog / AnnotationDialog
  original_label = model tahmini
  corrected_label = kullanıcı seçimi
      ↓
PendingClipManager.save_pending_clip()
  → live_clips/pending/<CLASS>/<ID>_<CLASS>.wav
  → pending_manifest.csv (status=pending)
      ↓  [DataReviewTab: Dinle → Onayla]
PendingClipManager.approve_clip()
  → live_clips/approved/<CLASS>/
  → approved_manifest.csv
      ↓
dataset_builder_v4.py → manifest_v5.csv (gelecek)
```

---

## 7. Tamamlanan İşler

### v3 → v4 Geçişinde Yapılanlar
- [x] **OTHER sınıfı eklendi** (6. sınıf) — modelin tanımadığı sesleri AIRCRAFT yerine OTHER'a atar
- [x] `GENERIC_AUDIO_CLASSIFIER` (D:\Downloads_2\DATASET) veri seti entegre edildi
- [x] `dataset_builder_v4.py` yazıldı — klasör→sınıf eşleme tablosu ile
- [x] `noise_detector.py` güncellendi: `PRIOR_WEIGHTS`, `CLASS_COLORS`, `LABEL_COLORS`'a OTHER eklendi
- [x] `train_efficientnet.py` güncellendi: `MANUAL_CLASS_WEIGHTS`'e OTHER eklendi
- [x] `gui_main.py` güncellendi: `AnnotationDialog` ve `LiveLabelDialog` sınıflarına OTHER eklendi
- [x] **GPU desteği** eklendi: `map_location=_TORCH_DEVICE`, `.to(device)`, `torch.backends.cudnn.benchmark=False`
- [x] **CMVN normalizasyonu** eklendi: `train_efficientnet.py` ve `noise_detector.py`
- [x] **RMS normalizasyonu** eklendi ve kısmen geri alındı (AIRCRAFT F1'ini düşürdüğü için)
- [x] DataLoader: `num_workers=0` (Windows multiprocessing uyumluluğu için)

### Temel Altyapı (v3'ten devir)
- [x] Modüler ses işleme motoru
- [x] 264-boyutlu özellik vektörü (SVM için)
- [x] Model hiyerarşisi: EfficientNet → CNN → SVM → Kural Tabanlı
- [x] 2 aşamalı EfficientNet eğitimi (Freeze → FineTune)
- [x] Weighted loss, MANUAL_CLASS_WEIGHTS, early stopping
- [x] Confusion matrix, per-class F1, val/test raporları
- [x] GUI: tüm sekmeler, VU metre, rolling şerit, FFT spektrum
- [x] Etiketleme → pending → approved → CSV export akışı

---

## 8. Mevcut Durum ve Bilinen Sorunlar

### Çalışan Özellikler
- Faz 1 dosya analizi tam çalışıyor
- Faz 2 mikrofon akışı çalışıyor (CUDA ile)
- 6 sınıflı EfficientNet modeli eğitildi ve yüklendi
- Etiketleme sistemi çalışıyor

### Bilinen Sorunlar

**1. Domain Mismatch — Kritik**
Canlı mikrofon testleri zayıf. Model studio kayıtlarıyla eğitildi; mikrofon sesi farklı akustik özellikler taşıyor. Özellikle:
- `SPEECH` neredeyse hiç tahmin edilmiyor (raw prob ≈ 0.0)
- `AIRCRAFT` aşırı tahmin ediliyor (prior ile bastırılıyor ama yetmiyor)
- Farklı mikrofon takıldığında tahminler bozuluyor

**2. AIRCRAFT F1 Düşüşü**
RMS normalizasyonu eklendikten sonra AIRCRAFT F1: 0.94 → 0.74 geriledi. RMS normalizasyonu uçağın yüksek enerji özelliğini sildi. Çözüm: RMS normalizasyonunu kaldırıp sadece CMVN ile yeniden eğitmek.

**3. Kayan Pencere Sorunu**
5 saniyelik pencere, kısa sesleri (konuşma 1-2s) yakalamakta yetersiz. Her 1 saniyede inference tetikleniyor ama model 5 saniyelik temiz ses için eğitildi.

**4. Annotation Marker Layout Sorunu (Faz 1)**
Şerit grafiğine annotation eklendiğinde yükseklik zaman zaman küçülüyor. Tam çözüme ulaşılmadı.

---

## 9. Yapılacaklar

### 🔴 Öncelikli — Domain Mismatch ve Canlı Performans

#### A) RMS Normalizasyonunu Kaldır, CMVN Koru
AIRCRAFT F1'ini geri kazanmak için.

**Değişiklik için gerekli dosyalar:**
```
dataset_builder_v4.py
noise_detector.py
train_efficientnet.py
```
Kaldırılacak blok (her dosyada `load_audio_fixed` veya `_infer_efficientnet_chunk` içinde):
```python
rms = np.sqrt(np.mean(y ** 2))
if rms > 1e-8:
    y = y * (0.1 / rms)
y = np.clip(y, -1.0, 1.0)
```
Ardından `features_v4.pkl` silinecek, dataset_builder ve train_efficientnet yeniden çalıştırılacak.

---

#### B) Majority Voting / Smoothing (eğitim gerektirmez)
Ekranda her saniye değişen tahmin yerine son N tahminin çoğunluğunu göster.

**Değişiklik için gerekli dosya:**
```
noise_detector.py  (classify_chunk_live veya MicrophoneWorker)
```

---

#### C) Confidence Threshold (eğitim gerektirmez)
En yüksek olasılık eşiğin altındaysa `OTHER` veya `UNKNOWN` döndür.

**Değişiklik için gerekli dosya:**
```
noise_detector.py  (_apply_prior veya classify_chunk_live)
```

---

#### D) Pencere Süresini Kısalt (eğitim gerektirir)
5s → 2s. Kısa sesleri (konuşma) daha iyi yakalar.

**Değişiklik için gerekli dosyalar:**
```
dataset_builder_v4.py   (CLIP_DUR değişkeni)
train_efficientnet.py   (DURATION değişkeni)
noise_detector.py       (WINDOW, SLIDE_SAMPLES)
```
Tüm önbellekler silinecek, sıfırdan eğitim gerekecek.

---

#### E) Alan İçi Fine-Tune (en kalıcı çözüm)
Kullanılacak ortamda kayıt al, mevcut modeli fine-tune et.

Adımlar:
1. GUI'deki kayıt sistemiyle hedef ortamda ses topla
2. DataReviewTab'dan onayla → `approved_manifest.csv`
3. `dataset_builder_v5.py` yaz: approved_manifest + v4 verisini birleştir
4. `train_efficientnet.py`'de backbone'u dondur, sadece classifier eğit (5-10 epoch)

**Değişiklik için gerekli dosyalar:**
```
train_efficientnet.py
dataset_builder_v4.py  (referans)
```

---

### 🟡 Orta Öncelik

- [ ] Mikrofon cihaz listesi yenileme butonu
  - `sd._terminate()` + `sd._initialize()` → cihaz listesini yenile
  - **Gerekli dosya:** `gui_main.py`

- [ ] Annotation marker layout sorunu kalıcı çözüm
  - **Gerekli dosya:** `gui_main.py`

---

## 10. Sonraki Sohbet İçin Bağlam

Yeni bir sohbette devam ederken:

1. Bu dokümanı (`PROJE_DOKUMANTASYON_v10.md`) paylaş
2. Yapılacak işe göre ilgili dosyaları paylaş — **Claude ilk mesajda hangi dosyaları istediğini belirtecek**
3. Herhangi bir değişiklikten önce Claude ilgili kod bloklarını senden isteyecek

### Temel Kurallar
- `manifest_v3.csv` ve `features_v3.pkl`'ye **dokunma** — yedek olarak kal
- GPU değişiklikleri daha önce yapıldı, `map_location="cpu"` satırları **yok**
- `num_workers=0` Windows'ta kalmalı — 4 yapınca multiprocessing crash veriyor
- `dataset_builder_v4.py` çalıştırılmadan önce `features_v4.pkl` silinmeli (normalizasyon değişirse)

### Hızlı Başvuru — Temel Parametreler
```python
SR          = 22050    # Örnekleme frekansı
WINDOW      = 5.0      # Pencere süresi (saniye)
HOP_SEC     = 2.5      # Pencere atlama (%50 overlap)
N_FFT       = 2048
HOP_FFT     = 512
N_MELS      = 128
EFF_IMG     = 224      # EfficientNet giriş boyutu
FEAT_DIM    = 264      # SVM özellik vektörü boyutu
BATCH_SIZE  = 32       # EfficientNet eğitim batch boyutu
```
