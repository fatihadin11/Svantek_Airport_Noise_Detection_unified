# ✈ Havalimanı Çevresel Gürültü Tespit Sistemi
**Airport Environmental Noise Detection System**

> **Durum:** Aktif Geliştirme — v3.1  
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

**Sınıflar:** `AIRCRAFT` | `AMBIENT` | `SPEECH` | `TRAFFIC` | `WIND`

**İki temel çalışma modu:**
- **Faz 1 — Dosya Analizi:** WAV/MP3/FLAC dosyası yükle, analiz et, görselleştir
- **Faz 2 — Canlı Mikrofon:** Gerçek zamanlı ses akışı, anlık sınıflandırma, kayıt ve etiketleme

---

## 2. Klasör Yapısı

```
proje_kök/
│
├── noise_detector.py          # Tüm ML modelleri + ses işleme motoru
├── gui_main.py                # PyQt6 arayüzü (tek dosya)
├── mic_map.py                 # Harita sekmesi (opsiyonel, MapTab)
│
├── models/                    # Eğitilmiş model ağırlıkları
│   ├── best_efficientnet.pt   # EfficientNet-B0 checkpoint
│   ├── efficientnet_label_encoder.pkl
│   ├── best_cnn.pt            # CNN checkpoint
│   ├── cnn_label_encoder.pkl
│   ├── best_model.pkl         # SVM modeli (joblib)
│   └── label_encoder.pkl      # SVM label encoder
│
├── dataset/                   # Orijinal eğitim veri seti
│   ├── AIRCRAFT/
│   ├── AMBIENT/
│   ├── SPEECH/
│   ├── TRAFFIC/
│   └── WIND/
│
├── live_clips/                # Canlı mikrofon etiketleme çıktıları
│   ├── pending/               # Henüz onaylanmamış klipler
│   │   ├── AIRCRAFT/
│   │   ├── SPEECH/
│   │   └── ...
│   ├── approved/              # Onaylanmış, eğitime hazır klipler
│   │   ├── AIRCRAFT/
│   │   ├── SPEECH/
│   │   └── ...
│   ├── rejected/              # Reddedilen klipler
│   ├── pending_manifest.csv   # Pending kliplerin kaydı
│   └── approved_manifest.csv  # Onaylı klipler — dataset_builder okur
│
├── outputs_gui/               # Faz 1 dosya analizi çıktıları (PNG, CSV)
├── outputs_mic/               # Faz 2 kayıt çıktıları (WAV + CSV)
│   ├── session_YYYYMMDD_HHMMSS.wav
│   └── session_YYYYMMDD_HHMMSS.csv
│
├── dataset_builder_v3.py      # Dataset hazırlama scripti (approved_manifest okur)
├── train_efficientnet.py      # EfficientNet eğitim scripti
├── train_cnn.py               # CNN eğitim scripti
└── training_labels.csv        # GUI'den dışa aktarılan eğitim etiketi CSV'si
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
| `AnnotationDialog` | Faz 1 annotation (şeride tıkla → etiket düzelt) |
| `LiveLabelDialog` | Faz 2 etiketleme (model tahmini göster → kullanıcı etiket seç) |
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
`model_pref="auto"` seçilince bu sırada ilk bulunan çalışır.

### Ortak Parametreler
```python
SR          = 22050 Hz
WINDOW      = 5.0 saniye
HOP         = 2.5 saniye (%50 overlap)
N_FFT       = 2048
HOP_FFT     = 512
N_MELS      = 128
```

### EfficientNet-B0
- ImageNet ağırlıklarıyla başlatılan backbone, son sınıflandırıcı katmanı değiştirildi
- Giriş: 224×224 RGB, Mel spektrogram → normalize → 3 kanala kopyala
- Çıkış: 5 sınıf softmax
- Prior ağırlık düzeltmesi (`PRIOR_WEIGHTS`) uygulanır
- Checkpoint: `models/best_efficientnet.pt` → `{model_state, epoch, val_f1, phase}`

### CNN (AirportCNN)
- 3× Conv2D + BatchNorm + ReLU + MaxPool + Dropout bloğu
- AdaptiveAvgPool → 4×4 → Flatten → 256 → n_classes
- Giriş: (1, 128, T) Mel spektrogram
- Checkpoint: `models/best_cnn.pt` → `{model_state, epoch, n_classes}`

### SVM
- 264-boyutlu el-yapımı özellik vektörü
- `predict_proba` destekli, prior düzeltmesi uygulanır
- Checkpoint: `models/best_model.pkl` + `models/label_encoder.pkl`

### Prior Ağırlık Düzeltmesi
Eğitim verisi AIRCRAFT-ağırlıklı olduğundan model diğer sınıflara karşı önyargılıdır. Tüm modellerde ham softmax çıktısına şu ağırlıklar uygulanır:
```python
PRIOR_WEIGHTS = {
    "AIRCRAFT": 0.10,   # Baskıla — eğitimde aşırı temsil edildi
    "SPEECH":   2.0,
    "TRAFFIC":  2.0,
    "AMBIENT":  2.0,
}
```
Ardından normalize edilip yeniden olasılık vektörüne çevrilir.

---

## 5. GUI — Sekme Rehberi

### 📊 Sınıflandırma (Faz 1)
- Dosya aç → Model seç → Analiz Et
- Çıktılar: renk kodlu sınıflandırma şeridi, dBFS grafiği, softmax eğrileri
- **Annotation Modu:** "✏ Annotation" butonu açıkken şeride tıklayınca `AnnotationDialog` açılır, etiket düzeltilip `live_clips/pending/` klasörüne eklenir

### 🎨 Mel Spektrogram (Faz 1)
128-Mel spektrogram, üstte mini sınıflandırma şeridi

### 📈 Özellikler (Faz 1)
ZCR, RMS, Spectral Centroid zaman grafikleri

### 🎙 Canlı Mikrofon (Faz 2)
- Cihaz + Model seç → ▶ Başlat
- Sol panel: anlık sınıf rozeti, dBFS, VU metre, softmax güven barları
- Sağ panel: son 60s sınıflandırma şeridi, dBFS geçmişi, canlı FFT spektrum
- **⚫ Kayıt Başlat:** WAV + CSV `outputs_mic/` klasörüne kaydeder
- **🏷 Etiketle & Gönder:** `LiveLabelDialog` açılır:
  - Modelin anlık tahmini büyük font ile gösterilir
  - Kullanıcı açılır listeden kendi etiketini seçer
  - Eşleşmezse sarı uyarı gösterilir
  - "📥 Gönder" ile 5 saniyelik WAV `live_clips/pending/` klasörüne kaydedilir
  - `original_label` (model) ≠ `corrected_label` (kullanıcı) olarak CSV'ye yazılır

### 📋 Veri Review
İki bölümlü veri yönetim ekranı:

**Üst: Bekleyen Klipler**

| Buton | İşlev |
|---|---|
| 🔄 Yenile | Manifest yeniden oku |
| ▶ Dinle | Seçili klip WAV'ını çal |
| ⏹ Durdur | Oynatmayı durdur |
| ✅ Onayla | `pending/` → `approved/` taşı |
| ❌ Reddet | `pending/` → `rejected/` taşı |
| ✅✅ Tümünü Onayla | Tüm bekleyenleri onayla |

**Alt: Onaylanmış Veri Seti**

| Buton | İşlev |
|---|---|
| ▶ Dinle | Seçili onaylı klip WAV'ını çal |
| 🗑 Seçileni Sil | Manifest kaydını VE WAV dosyasını kalıcı sil |
| 🗑🗑 Tümünü Sil | Tüm onaylı veriyi sıfırla |
| 📤 Eğitim CSV Dışa Aktar | `training_labels.csv` oluştur |

**Tablo sütunları:** Kullanıcı Etiketi | Model Tahmini | Eşleşme (✓/≠) | Kaynak Dosya | Güven | Not

### 🗺 Harita (opsiyonel)
`mic_map.py` → `MapTab` yüklüyse görünür, aksi hâlde sekme oluşmaz.

---

## 6. Veri Akışı

### Etiketleme → Eğitim Akışı
```
Mikrofon / Dosya
      │
      ▼
LiveLabelDialog / AnnotationDialog
  original_label = model tahmini
  corrected_label = kullanıcı seçimi
      │
      ▼
PendingClipManager.save_pending_clip()
  → live_clips/pending/<CLASS>/<ID>_<CLASS>.wav
  → pending_manifest.csv (status=pending)
      │
      ▼  [DataReviewTab: Dinle → Onayla]
      │
      ▼
PendingClipManager.approve_clip()
  → live_clips/approved/<CLASS>/<ID>_<CLASS>.wav
  → approved_manifest.csv (status=approved)
      │
      ▼  [DataReviewTab: 📤 Eğitim CSV Dışa Aktar]
      │
      ▼
training_labels.csv
  clip_id | clip_path | user_label | model_label | label_match | confidence | note | timestamp
      │
      ▼
dataset_builder_v3.py  (approved_manifest.csv okur)
      │
      ▼
train_efficientnet.py / train_cnn.py
```

### Canlı Mikrofon Inference Akışı
```
sounddevice InputStream
  → BLOCK_SIZE=1024 chunk'lar kuyruk'a girer
  → 5s kayan pencere (WINDOW_SAMPLES=110250)
  → Her SLIDE_SAMPLES=22050 (1s) inference tetiklenir
  → AirportNoiseSystem.classify_chunk_live()
      → EfficientNet / CNN / SVM
      → PRIOR_WEIGHTS düzeltmesi
      → {label, probs, db_rms}
  → GUI sinyal (result_signal) ile güncellenir
```

---

## 7. Tamamlanan İşler

### Temel Altyapı
- [x] Modüler ses işleme motoru (`AudioLoader`, `AudioAnalyzer`, `FeatureExtractor`, `NoiseFilter`, `Visualizer`)
- [x] Kural tabanlı sınıflandırıcı (yedek sistem)
- [x] 264-boyutlu el-yapımı özellik vektörü (`extract_features_ml`)
- [x] SVM modeli eğitimi ve entegrasyonu
- [x] CNN (`AirportCNN`) eğitimi ve entegrasyonu
- [x] EfficientNet-B0 transfer learning eğitimi ve entegrasyonu
- [x] Model hiyerarşisi: EfficientNet → CNN → SVM → Kural Tabanlı
- [x] Prior ağırlık düzeltmesi (AIRCRAFT aşırı temsil sorunu)

### GUI (PyQt6)
- [x] Karanlık tema, renk paleti, stylesheet sistemi
- [x] Faz 1: Dosya analizi, 3 grafik sekmesi (sınıflandırma, spektrogram, özellikler)
- [x] Faz 1: CSV + PNG export
- [x] Faz 2: Canlı mikrofon akışı (`MicrophoneWorker` QThread)
- [x] Faz 2: VU metre, rolling sınıflandırma şeridi, dBFS geçmişi
- [x] Faz 2: Canlı FFT spektrum görüntüleyici
- [x] Faz 2: WAV + CSV kayıt sistemi

### Etiketleme & Staging Sistemi
- [x] `PendingClipManager` — pending/approved/rejected klasör yapısı
- [x] Faz 1 Annotation modu: şeride tıkla → `AnnotationDialog` → pending'e ekle
- [x] Faz 2 Etiketleme: `LiveLabelDialog` — model tahmini göster, kullanıcı etiket seç
  - `original_label` (model) ile `corrected_label` (kullanıcı) ayrı tutulur
  - Etiket uyuşmazlığında sarı uyarı
  - Model kendi tahminini onaylıyorsa yeşil onay
- [x] **Veri Review sekmesi yeniden tasarlandı (iki bölüm):**
  - Pending bölümü: mevcut işlevler
  - Approved bölümü: **Dinle, Seçileni Sil, Tümünü Sil** — hatalı etiketleri manifest'ten ve diskten kalıcı sil
  - Eğitim CSV export: `user_label`, `model_label`, `label_match` sütunları

---

## 8. Mevcut Durum ve Bilinen Sorunlar

### Çalışan Özellikler
- Faz 1 analiz ve görselleştirme tam çalışıyor
- Faz 2 mikrofon akışı çalışıyor
- Etiketleme → pending → approved → CSV export akışı çalışıyor
- Hatalı etiketler DataReviewTab'dan kalıcı silinebiliyor

### Bilinen Sorunlar (Ertelenmiş)
Bu sorunlar var olduğu bilinmekte ancak şu an düzeltilmemiştir:

1. **Annotation marker layout sorunu (Faz 1):** Şerit grafiğine annotation eklendiğinde grafiğin yüksekliği zaman zaman küçülüyor. `fig.set_layout_engine("none")` ile kısmen giderildi, tam çözüme ulaşılmadı.

2. **Model CPU'da çalışıyor, GPU kullanılmıyor:** EfficientNet ve CNN inference sırasında `map_location="cpu"` ile yükleniyor. CUDA kontrolü yok. Hem eğitim hem de inference yavaş.

3. **Mikrofon sekmesinde cihaz listesi sorunu:** Bazı sistemlerde cihaz listesi doğru dolmuyor (ses kütüphanesine bağımlı).

---

## 9. Yapılacaklar

### 🔴 Öncelikli — GPU Desteği

**Problem:** Model checkpoint'ler `map_location="cpu"` ile yükleniyor. CUDA varsa bile kullanılmıyor. Eğitim ve inference son derece yavaş.

**Hedef dosyalar:**
- `noise_detector.py` — `_load_cnn_model`, `_load_efficientnet_model`, `_infer_efficientnet_chunk`, `_infer_cnn_chunk`
- `train_efficientnet.py`, `train_cnn.py`

**Yapılacaklar:**
- [ ] `torch.device` tespiti: `cuda` varsa GPU, yoksa CPU
- [ ] Checkpoint yüklemede `map_location=device`
- [ ] Model `.to(device)` çağrısı
- [ ] Inference sırasında tensör `.to(device)` — sonuç `.cpu().numpy()`
- [ ] `train_efficientnet.py` ve `train_cnn.py` içinde de `device` otomasyonu
- [ ] `DataLoader` içinde `pin_memory=True` (GPU varsa)
- [ ] GUI'de hangi cihazın kullanıldığını status bar'da göster

**Değişiklik için gerekli kod blokları:**
Değişikliğe başlamadan önce şu dosyaların güncel hallerini paylaş:
```
noise_detector.py
train_efficientnet.py  (varsa)
train_cnn.py           (varsa)
```

---

### 🟡 Orta Öncelik — EfficientNet İyileştirme

**Problem:** Mevcut veri seti AIRCRAFT-ağırlıklı ve yeterince büyük değil. Prior ağırlıklarıyla geçici düzeltme yapılıyor ancak kök neden veri dengesizliği.

**Hedefler:**
- [ ] Ek dataset toplamak:
  - Ortam sesleri için `ESC-50`, `UrbanSound8K`, `FreeSound` gibi açık kaynaklar
  - Uçak sesleri için Freesound / Aviation audio datasets
  - Trafik ve rüzgar sesleri
- [ ] `dataset_builder_v3.py` güncelleme — dış kaynaklardan gelen sesleri entegre et
- [ ] Sınıf başına hedef: en az 500 klip (şu an muhtemelen AIRCRAFT hariç düşük)
- [ ] `train_efficientnet.py` güncelleme:
  - Weighted random sampler veya class weights ile dengesizlik düzelt
  - Data augmentation: zaman kaydırma, pitch shift, gürültü ekleme, mixup
  - Fine-tuning fazı: ilk N epoch backbone dondurulmuş, sonra açılmış
  - Learning rate scheduler (CosineAnnealing veya OneCycle)
- [ ] Validation metrikleri: confusion matrix, per-class F1, ROC eğrisi

**Değişiklik için gerekli kod blokları:**
```
train_efficientnet.py
dataset_builder_v3.py  (varsa)
```

---

### 🟢 Düşük Öncelik — Diğer İyileştirmeler

- [ ] Annotation marker layout sorunu kalıcı çözüm
- [ ] Mikrofon cihaz listesi yenileme butonu
- [ ] `approved_manifest.csv`'yi veri seti birleştirme scriptine bağla
- [ ] Model performans karşılaştırma ekranı (Confusion Matrix görselleştirmesi)
- [ ] Harita sekmesi (`mic_map.py`) entegrasyon testleri

---

## 10. Sonraki Sohbet İçin Bağlam

Yeni bir sohbette devam etmek için aşağıdaki bilgileri sağla:

### GPU Desteği Geliştirmesi İçin
1. Bu dokümanı paylaş
2. Şu dosyaların **güncel** hallerini ekle:
   - `noise_detector.py`
   - `train_efficientnet.py`
   - `train_cnn.py`
3. Sistemdeki GPU bilgisini paylaş:
   ```bash
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU yok')"
   ```

### EfficientNet İyileştirmesi İçin
1. Bu dokümanı paylaş
2. Şu dosyaların **güncel** hallerini ekle:
   - `train_efficientnet.py`
   - `dataset_builder_v3.py`
3. Mevcut veri seti dağılımını paylaş:
   ```bash
   # Her sınıf için kaç WAV var?
   find dataset/ -name "*.wav" | sed 's|/[^/]*$||' | sort | uniq -c
   ```

### Genel Not
`gui_main.py` değiştirilecekse her zaman güncel hali eklenmelidir — dosya büyük, bağlam kritik.

---

## Hızlı Başvuru — Sınıf Renkleri

```python
CLASS_COLORS = {
    "AIRCRAFT": "#FF6B35",  # Turuncu
    "AMBIENT":  "#7EE8A2",  # Yeşil
    "SPEECH":   "#A8DADC",  # Açık mavi
    "TRAFFIC":  "#FFE66D",  # Sarı
    "WIND":     "#4ECDC4",  # Turkuaz
    "UNKNOWN":  "#6C757D",  # Gri
}
```

## Hızlı Başvuru — Temel Parametreler

```python
# noise_detector.py ile dataset_builder senkronize olmalı
SR          = 22050    # Örnekleme frekansı
WINDOW      = 5.0      # Pencere süresi (saniye)
HOP_SEC     = 2.5      # Pencere atlama (saniye, %50 overlap)
N_MFCC      = 40       # MFCC katsayı sayısı
N_FFT       = 2048     # FFT boyutu
HOP_FFT     = 512      # FFT hop length
N_MELS      = 128      # Mel kanal sayısı
EFF_IMG     = 224      # EfficientNet giriş boyutu (224×224)
FEAT_DIM    = 264      # SVM özellik vektörü boyutu
```
