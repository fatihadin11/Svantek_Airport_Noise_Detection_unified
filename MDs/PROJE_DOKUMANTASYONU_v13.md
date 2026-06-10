# TÜBİTAK Havalimanı Gürültü Tespiti — Proje Dokümantasyonu v13

> **Önceki sürüm:** v12 — EfficientNet pipeline tamamlanmıştı  
> **Bu sürüm:** v13 — Bölüm 10 tamamlandı: BEATs/Modern Kanal, Group-Split, Augmentation  
> **Tarih:** Haziran 2025

---

## 🔁 Çalışma Kuralları (Önemli)

> Yeni bir sohbet başlatıldığında Claude'a şunu söyle:
> **"Projeye X. Aşama'dan devam etmek istiyorum, gerekli dosyaları iste."**

**Dosya isteme kuralı:** Herhangi bir dosyada değişiklik yapılmadan önce Claude ilgili dosyayı kullanıcıdan ister. Dosya paylaşılmadan kod yazılmaz. Küçük değişiklikler (< 20 satır) elle yapılabilir şekilde açıklanır; büyük değişiklikler tam dosya olarak teslim edilir.

---

## 📁 Klasör Yapısı (Güncel)

```
C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise\       ← PROJE KÖKÜ
│
├── BEATs.py                    ← microsoft/unilm (proje köküne kopyalanmalı)
├── backbone.py                 ← microsoft/unilm
├── modules.py                  ← microsoft/unilm
├── quantizer.py                ← microsoft/unilm
│
├── noise_detector.py           ← AirportNoiseSystem + tüm classifier sınıfları
├── gui_main.py                 ← PyQt5 GUI
├── train_beats.py              ← BEATs MLP eğitim scripti (v2 — güncel)
├── train_efficientnet.py       ← EfficientNet eğitim scripti
├── dataset_builder.py          ← Manifest oluşturucu
│
├── cache/
│   ├── manifest_v4.csv         ← ESC-50 + AeroSonicDB + env_audio + D:\DATASET
│   └── manifest_v5.csv         ← manifest_v4 + onaylı live klipler (varsa)
│
├── models/                     ← Lokal model çıktıları (opsiyonel)
│
└── outputs/
    └── training_beats/
        ├── beats_training_curves.png
        ├── beats_per_class_f1.png
        ├── beats_confusion_matrix_test.png
        └── beats_confusion_matrix_val.png

D:\models\                      ← AĞIRLIKLAR VE CACHE (D: diskinde)
├── BEATs_iter3_plus_AS2M.pt    ← BEATs frozen encoder (~90 MB)
├── beats_mlp.pt                ← Eğitilmiş MLP (güncel)
├── beats_embed_cache.pkl       ← BASE embeddingler (~tüm veri, temiz)
└── beats_aug_cache.pkl         ← Augmented embeddingler (sadece train seti)

D:\Downloads_2\DATASET\         ← Ham veri seti (değiştirilmez)
D:\Airport_Live_Clips\          ← Canlı kayıt klipleri + approved_manifest.csv
```

---

## ✅ Bu Sohbette Tamamlananlar (Bölüm 10)

### 10.1 noise_detector.py — Eklenen Metodlar

**Dosya paylaşılmadan değiştirilmedi.** Şu metodlar eksik tespit edilip eklendi:

#### `_classify_beats(audio_path)` — Satır ~1413'ten sonra

```python
def _classify_beats(self, audio_path: str):
    chunks, _ = _load_and_chunk_ml(audio_path)
    classes   = _BEATS_CLASSES
    frame_labels, frame_probs = [], []

    for chunk in chunks:
        try:
            label, probs_dict = self.beats_model.infer(chunk)
            probs_arr = np.array(
                [probs_dict.get(cls, 0.0) for cls in classes], dtype=np.float32)
            frame_labels.append(label); frame_probs.append(probs_arr)
        except Exception as e:
            print(f"  [!] BEATs pencere hatası: {e}")
            frame_labels.append("UNKNOWN")
            frame_probs.append(np.ones(len(classes), dtype=np.float32) / len(classes))

    label_times = np.array([i * _ML_HOP_SEC for i in range(len(frame_labels))])
    counts  = Counter(frame_labels); total = len(frame_labels)
    summary = {k: round(100*v/total, 1) for k,v in
               sorted(counts.items(), key=lambda x: -x[1])}
    return frame_labels, label_times, summary, frame_probs, list(classes)
```

#### `_classify_ensemble(audio_path)` — `_classify_beats`'in hemen ardına

```python
def _classify_ensemble(self, audio_path: str):
    chunks, _ = _load_and_chunk_ml(audio_path)
    classes   = _BEATS_CLASSES
    frame_labels, frame_probs = [], []

    for chunk in chunks:
        try:
            label, probs_dict = self.ensemble_model.infer(chunk)
            probs_arr = np.array(
                [probs_dict.get(cls, 0.0) for cls in classes], dtype=np.float32)
            frame_labels.append(label); frame_probs.append(probs_arr)
        except Exception as e:
            print(f"  [!] Ensemble pencere hatası: {e}")
            frame_labels.append("UNKNOWN")
            frame_probs.append(np.ones(len(classes), dtype=np.float32) / len(classes))

    label_times = np.array([i * _ML_HOP_SEC for i in range(len(frame_labels))])
    counts  = Counter(frame_labels); total = len(frame_labels)
    summary = {k: round(100*v/total, 1) for k,v in
               sorted(counts.items(), key=lambda x: -x[1])}
    return frame_labels, label_times, summary, frame_probs, list(classes)
```

#### `analyze_for_gui()` — SVM bloğundan sonra, `auto` bloğundan önce

```python
elif model_pref == "beats" and self.beats_model is not None:
    fl, lt, sm, fp, cn = self._classify_beats(audio_path)
    used = "BEATs (Modern)"
elif model_pref == "ensemble" and self.ensemble_model is not None:
    fl, lt, sm, fp, cn = self._classify_ensemble(audio_path)
    used = "Ensemble (EfficientNet + BEATs)"
```

---

### 10.2 gui_main.py — 4 Küçük Değişiklik

**Değişiklik 1** — `MicrophoneTab.MODEL_MAP` (~satır 1689):
```python
MODEL_MAP = {
    "Otomatik (EfficientNet → CNN → SVM)": "auto",
    "EfficientNet-B0": "efficientnet",
    "CNN": "cnn",
    "SVM": "svm",
    "BEATs (Modern)": "beats",                       # ← yeni
    "Ensemble (EfficientNet + BEATs)": "ensemble",   # ← yeni
}
```

**Değişiklik 2** — `MainWindow._MODEL_MAP` (~satır 1985):
```python
_MODEL_MAP = {0:"auto", 1:"efficientnet", 2:"cnn", 3:"svm",
              4:"beats", 5:"ensemble"}   # ← 4 ve 5 yeni
```

**Değişiklik 3** — `model_combo.addItems()` (~satır 2014):
```python
self.model_combo.addItems([
    "Otomatik (EfficientNet → CNN → SVM)",
    "EfficientNet-B0", "CNN", "SVM",
    "BEATs (Modern)",                       # ← yeni
    "Ensemble (EfficientNet + BEATs)",      # ← yeni
])
```

**Değişiklik 4 (Bug Fix)** — Duplike `tab_mic` satırı (~satır 2040):
```python
# SİL:
self.tab_mic = MicrophoneTab(self.system)        ← bu satırı sil
# KALS N:
self.tab_mic = MicrophoneTab(self.system, self._clip_mgr)   ← bu doğru
```

---

### 10.3 train_beats.py — Sıfırdan Yazıldı (v1 → v2)

Tamamı yeni yazılan dosya. **İki versiyon:**

| Versiyon | Açıklama | Durum |
|---|---|---|
| v1 | Temel pipeline, random split | Çalışır, metrikleri şişirilmiş |
| v2 | Group-split + Augmentation | **Güncel, bunu kullan** |

#### Manifest Formatı (manifest_v4/v5.csv)
```
Sütunlar: path, label
Örnek:    C:\...\1-101296-A-19.wav, WIND
```

#### Checkpoint Formatı (beats_mlp.pt)
```python
{
    "epoch":       int,
    "model_state": model.net.state_dict(),  # keys: 0.weight/0.bias/3.weight/3.bias
    "label_names": ["AIRCRAFT","AMBIENT","OTHER","SPEECH","TRAFFIC","WIND"],
    "n_classes":   6,
    "val_f1":      float,
    "embed_dim":   768,
}
```

> ⚠️ `model.net.state_dict()` — `model.state_dict()` DEĞİL.  
> noise_detector.py `beats.mlp.load_state_dict(ckpt["model_state"])` ile yükler.  
> Key uyumu zorunlu: `0.weight / 3.weight` (net. prefix'siz).

#### BEATs Encoder Yükleme (Kritik Düzeltme)
```python
# YANLIŞ (v1'deki bug):
cfg = BEATsConfig()          # input_patch_size=-1 → Conv2d patlar!
enc = BEATs(cfg)

# DOĞRU (v2):
ckpt = torch.load(path, ...)
cfg  = BEATsConfig(ckpt["cfg"])   # config checkpoint'ten okunmalı
enc  = BEATs(cfg)
enc.load_state_dict(ckpt["model"])
```

---

## 🔧 Kurulum Adımları (BEATs)

BEATs her yeni ortamda kurulması gereken bağımlılık:

### 1. Python modülleri (proje kökünde olmalı)
```powershell
$root = "C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise"
$base = "https://raw.githubusercontent.com/microsoft/unilm/master/beats"
@("BEATs.py","backbone.py","modules.py","quantizer.py") | ForEach-Object {
    Invoke-WebRequest "$base/$_" -OutFile "$root\$_"
}
```

### 2. Encoder model dosyası
```powershell
New-Item -ItemType Directory -Force -Path "D:\models"
Invoke-WebRequest `
  "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt" `
  -OutFile "D:\models\BEATs_iter3_plus_AS2M.pt"
```

### 3. Import testi
```powershell
cd "C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise"
python -c "from BEATs import BEATs, BEATsConfig; print('OK')"
```

---

## 🐛 Bu Sohbette Çözülen Sorunlar

### Sorun 1: BEATsConfig() Conv2d RuntimeError
```
RuntimeError: Trying to create tensor with negative dimension -1: [512, 1, -1, -1]
```
**Neden:** `BEATsConfig()` boş oluşturulunca `input_patch_size=-1` geliyor.  
**Çözüm:** Config checkpoint içinden okunmalı → `BEATsConfig(ckpt["cfg"])`

---

### Sorun 2: Fan Sesi → SPEECH Yanlış Sınıflandırması
**Neden:** Canlı kayıt sırasında mikrofon fan sesini de aldı, klipler SPEECH etiketiyle eğitildi. Model "fan sesi = SPEECH" öğrendi.  
**Çözüm:**
1. Konuşmadan fan sesi klipleri kaydet (GUI Mikrofon sekmesi)
2. Data Review'da **AMBIENT** olarak etiketle ve onayla
3. `dataset_builder.py` → `train_beats.py` sırasıyla çalıştır

---

### Sorun 3: Data Leakage — ESC-50 A/B Varyantları
**Tespit:** `manifest_v5.csv` içinde `1-101296-A-19.wav` ve `1-101296-B-19.wav` gibi aynı kaydın farklı segmentleri rastgele split ile train ve val'a ayrılıyordu.

**Etki:** Val F1 Macro %99.56 (şişirilmiş), gerçek genelleme bundan düşük.

**ESC-50 Dosya Formatı:**
```
{fold}-{clip_id}-{take}-{class}.wav
1     -101296   -A    -19   .wav   ← aynı kayıt
1     -101296   -B    -19   .wav   ← aynı kayıt (farklı segment)
```

**Çözüm:** `train_beats.py v2`'de `GroupShuffleSplit` + `extract_source_id()`:
```python
def extract_source_id(path):
    fname = os.path.basename(path)
    m = re.match(r'^\d+-(\d+)-[A-Z]-\d+\.wav$', fname)
    if m:
        return f"esc50_{m.group(1)}"   # aynı clip_id → aynı grup
    return os.path.splitext(fname)[0]  # diğerleri → her dosya kendi grubu
```

---

### Sorun 4: Domain Shift (Farklı Mikrofon/Oda)
**Gözlem:**
- YouTube dosyaları → ✅ Mükemmel
- Kendi sessiz odası (eğitimde kullanılan) → ✅ Doğru OTHER
- Farklı sessiz oda → ❌ Sürekli AIRCRAFT

**Neden:** O odanın HVAC sistemi 80–200 Hz düşük frekanslı bileşen üretiyor. BEATs bunu uçak motoruyla ilişkilendiriyor. Training seti mikrofon çeşitliliği içermiyor.

**Uygulanan çözüm:** Waveform augmentation (`train_beats.py v2`):
```
• Gaussian noise  (SNR 10–25 dB, P=0.65)  — mikrofon gürültüsü
• Random gain     (±4 dB, P=0.50)         — kayıt seviyesi
• Pitch shift     (±1.5 yarı ton, P=0.35) — ses tonu kayması
Her eğitim klibinin N_AUGMENTS=2 versiyonu → eğitim seti ~3× büyür
```

**Kısmen çözüldü.** Tam çözüm için ek adımlar gerekli (bakınız: Kalan Görevler).

---

## 📊 Model Performans Özeti

| Model | F1 Macro | Split | Not |
|---|---|---|---|
| EfficientNet-B0 | 0.8859 | Random | Val F1, Epoch 26 |
| BEATs MLP v1 | 0.9921 | Random | Leakage var, şişirilmiş |
| BEATs MLP v2 | ? | Group-aware | Çalıştırılmadı, gerçek metrik |

---

## 🔄 Yeniden Eğitim Pipeline Sırası

Fan sesi / yeni klip eklendiğinde:

```
1. GUI → Data Review → klipleri AMBIENT/OTHER olarak onayla
            ↓
2. python dataset_builder.py    ← manifest_v5.csv güncellenir
            ↓
3. python train_beats.py        ← cache otomatik yenilenir, MLP eğitilir
            ↓
4. (Gerekirse) python train_efficientnet.py
            ↓
5. GUI başlatılır → D:\models\beats_mlp.pt otomatik yüklenir
```

> **NOT:** `noise_detector.py` ayrıca çalıştırılmaz. GUI başlatıldığında
> `AirportNoiseSystem.__init__` içinde model otomatik yüklenir.

### Cache Davranışı
```
beats_embed_cache.pkl:
  manifest path seti değişmişse → otomatik yenilenir
  Zorla yenilemek → python train_beats.py --rebuild-cache

beats_aug_cache.pkl:
  train seti path seti veya N_AUGMENTS değişmişse → otomatik yenilenir
  group-split değişince train seti değişir → aug cache de yenilenir
```

---

## 📋 Kalan Görevler (Bölüm 10 — Sıradaki)

### Adım 4: Offline Karşılaştırma (Henüz yapılmadı)
EfficientNet vs BEATs vs Ensemble performans karşılaştırması.  
**Gerekli dosya:** `train_efficientnet.py` ve kayıtlı EfficientNet model yolu.

### Adım 5: Ensemble α Optimizasyonu (Henüz yapılmadı)
`noise_detector._ENSEMBLE_ALPHA = 0.5` sabitini validation setinde tune et.  
Optimal α için grid search: 0.0 → 1.0 (0.1 adımlarla).

### Adım 6: Fingerprint Modülü (Opsiyonel)
Terminal anonsları için ses parmak izi (shazam-benzeri eşleştirme).

### Domain Shift — Ek Önlemler (Önerilen)
- [ ] Farklı ortamlardan (3+ farklı oda/mik) ambient klip topla
- [ ] Telefon mikrofonu kaydı → domain shift sayısal ölçümü
- [ ] `train_efficientnet.py`'e de group-split + augmentation ekle

### Makale için Karşılaştırma Tablosu
```
Tablo 1: Window-level split  → inflated baseline (v1 sonuçları)
Tablo 2: Group-level split   → gerçek metrikler (v2 sonuçları)
Tablo 3: Group + Augmentation → domain shift azalması
Bulgu: EfficientNet vs BEATs domain shift direnci karşılaştırması
```

---

## 🗂️ Dosya Değişiklik Özeti

| Dosya | Durum | Değişiklik |
|---|---|---|
| `noise_detector.py` | Elle değiştirildi | `_classify_beats`, `_classify_ensemble`, `analyze_for_gui` 2 elif |
| `gui_main.py` | Elle değiştirildi | MODEL_MAP, _MODEL_MAP, addItems, duplike satır silindi |
| `train_beats.py` | Yeni / Güncellendi | v1→v2: group-split + augmentation |
| `BEATs.py` | Yeni | microsoft/unilm'den indirildi |
| `backbone.py` | Yeni | microsoft/unilm'den indirildi |
| `modules.py` | Yeni | microsoft/unilm'den indirildi |
| `quantizer.py` | Yeni | microsoft/unilm'den indirildi |
| `D:\models\BEATs_iter3_plus_AS2M.pt` | Yeni | Microsoft sunucusundan indirildi |
| `D:\models\beats_mlp.pt` | Yeni | train_beats.py v1 ile eğitildi |

---

## 🏗️ Sistem Mimarisi (Güncel)

```
                    ┌─────────────────────────────────────────┐
                    │           AirportNoiseSystem             │
                    │                                          │
  Ses Dosyası ──►   │  ┌────────────┐  ┌────────────────────┐ │
  (analyze_for_gui) │  │EfficientNet│  │  BEATsClassifier   │ │
                    │  │   B0       │  │  (frozen encoder   │ │
  Mikrofon ──────►  │  │            │  │   + MLP 768→256→6) │ │
  (classify_chunk   │  └────────────┘  └────────────────────┘ │
   _live)           │        │                  │              │
                    │        └──────┬───────────┘              │
                    │               ▼                          │
                    │      ┌─────────────────┐                 │
                    │      │EnsembleClassifier│                │
                    │      │ α·BEATs +        │                │
                    │      │ (1-α)·EfficientNet│               │
                    │      └─────────────────┘                 │
                    └─────────────────────────────────────────┘

GUI Model Seçimi:
  0: auto          → EfficientNet → CNN → SVM sırasıyla dener
  1: efficientnet  → EfficientNet-B0
  2: cnn           → CNN
  3: svm           → SVM
  4: beats         → BEATs (Modern)   ← YENİ
  5: ensemble      → Ensemble         ← YENİ
```

---

## ⚙️ train_beats.py v2 — Önemli Parametreler

```python
# Augmentation — domain shift direnci
N_AUGMENTS     = 2      # Artırılabilir (3-4), aug cache yenilenir
AUG_NOISE_PROB = 0.65
AUG_GAIN_PROB  = 0.50
AUG_PITCH_PROB = 0.35   # 0.0 yapılırsa pitch shift devre dışı (hız için)

# Embedding batch — GPU belleğine göre ayarla
EMBED_BATCH = 32  # GPU ≥8GB → 64 | GPU 4GB → 32 | CPU → 16

# Split
TEST_SIZE   = 0.15
VAL_SIZE    = 0.15
RANDOM_SEED = 42
```

---

## 🚀 Hızlı Başlangıç (Yeni Sohbet)

Bir sonraki sohbette şu bölümden devam edilecek:

**"Bölüm 10 Adım 4'ten devam: EfficientNet vs BEATs offline karşılaştırma scripti yazılacak."**

Gerekli dosyalar istenecek:
- `train_efficientnet.py` (EfficientNet model yolu için)
- EfficientNet checkpoint yolu (örn. `D:\models\efficientnet_best.pt`)

---

*v13 — Bu sohbet sonu itibarıyla güncel durum*
