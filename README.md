# Airport Noise — Gerçek Zamanlı Çevresel Ses Sınıflandırma Sistemi

Havalimanı ortamında çalışan, uçtan uca gerçek zamanlı ses sınıflandırma sistemi.  
EfficientNet-B0 ve BEATs (Microsoft) foundation model'ini paralel olarak çalıştırır.

## Sınıflandırılan Kategoriler

| Sınıf | Açıklama |
|---|---|
| `AIRCRAFT` | Uçak motoru, kalkış, iniş sesleri |
| `AMBIENT` | Havalimanı ortam sesi |
| `SPEECH` | Konuşma, anons sesleri |
| `TRAFFIC` | Kara taşıtı sesleri |
| `WIND` | Rüzgar sesleri |
| `OTHER` | Yukarıdaki kategorilere girmeyen sesler |

---

## Sistem Gereksinimleri

- **OS:** Windows 10/11 (64-bit)
- **GPU:** CUDA destekli NVIDIA GPU (önerilir — BEATs encoder GPU olmadan çok yavaş çalışır)
- **RAM:** 16 GB+
- **Disk:** C: sürücüsünde ~2 GB, D: sürücüsünde ~500 MB (model ağırlıkları için)
- **Python:** 3.9 – 3.11

> ⚠️ Proje bazı yolları `D:\` sürücüsünde sabit kodlanmış olarak bekler.  
> `D:` sürücünüz yoksa ilgili yolları `noise_detector.py`, `train_beats.py` ve `dataset_builder.py` içinde arayıp güncelleyin (`D:\models`, `D:\Airport_Live_Clips`).

---

## Kurulum

### 1. Repoyu klonla

```bash
git clone https://github.com/<kullanici>/<repo>.git
cd Airport_Noise
```

### 2. Sanal ortam oluştur ve bağımlılıkları kur

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> PyTorch'u CUDA ile kurmak için önce [pytorch.org](https://pytorch.org/get-started/locally/) adresinden sisteminize uygun komutu alın:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```

### 3. BEATs Encoder'ı İndir

BEATs frozen encoder ağırlıklarını (~90 MB) Microsoft'un resmi kaynağından indir:

```
https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt
```

İndirilen dosyayı şu konuma yerleştir:

```
D:\models\BEATs_iter3_plus_AS2M.pt
```

### 4. D:\ Klasör Yapısını Oluştur

```
D:\
├── models\                    ← Yukarıda indirilen BEATs checkpoint buraya
│   └── BEATs_iter3_plus_AS2M.pt
└── Airport_Live_Clips\        ← Canlı kayıt oturumları için (GUI tarafından otomatik kullanılır)
    ├── pending\
    │   └── AIRCRAFT\  AMBIENT\  OTHER\  SPEECH\  TRAFFIC\  WIND\
    ├── approved\
    │   └── AIRCRAFT\  AMBIENT\  OTHER\  SPEECH\  TRAFFIC\  WIND\
    └── rejected\
        └── AIRCRAFT\  AMBIENT\  OTHER\  SPEECH\  TRAFFIC\  WIND\
```

Klasörleri hızlıca oluşturmak için PowerShell:

```powershell
$base = "D:\Airport_Live_Clips"
$classes = "AIRCRAFT","AMBIENT","OTHER","SPEECH","TRAFFIC","WIND"
foreach ($folder in "pending","approved","rejected") {
    foreach ($cls in $classes) {
        New-Item -ItemType Directory -Force -Path "$base\$folder\$cls"
    }
}
New-Item -ItemType Directory -Force -Path "D:\models"
```

---

## Harici Veri Setleri (Eğitim için)

Modeli sıfırdan eğitmek istiyorsan aşağıdaki veri setlerine ihtiyacın var.  
Sadece GUI'yi çalıştırıp inference yapacaksan bu adımı atlayabilirsin — eğitilmiş ağırlıklar repoda mevcut.

| Veri Seti | Kaynak | Hedef Klasör |
|---|---|---|
| ESC-50 | [github.com/karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50) | `Dataset_ESC50/` |
| AeroSonicDB | Proje sahibinden temin et | `Dataset_Airplane/` |
| Generic Audio Classifier | [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/audio-dataset) | `D:\Downloads_2\DATASET\` |

---

## Çalıştırma

### GUI (Canlı Sınıflandırma + Dosya Analizi)

```bash
python gui_main.py
```

Arayüz iki ana sekme içerir:
- **Faz 1 — Dosya Analizi:** Ses dosyası yükle, sınıflandır, haritada görselleştir
- **Faz 2 — Canlı Kayıt:** Mikrofondan gerçek zamanlı sınıflandırma ve aktif öğrenme

---

## Eğitim

Eğitim scriptlerini bu sırayla çalıştır:

### 1. Manifest Oluştur

```bash
python dataset_builder.py
```

`cache/manifest_v4.csv` ve `cache/manifest_v5.csv` oluşturur.

### 2. BEATs MLP Eğit (Önerilen)

```bash
python train_beats.py
```

- İlk çalıştırmada BASE embedding cache'i oluşturur (~45–50 dk, GPU gerekli)
- Cache oluştuktan sonraki çalıştırmalar çok daha hızlı
- Eğitilmiş MLP: `D:\models\beats_mlp.pt`

### 3. EfficientNet Eğit (İsteğe Bağlı)

```bash
python train_efficientnet.py
```

### 4. CNN / SVM Eğit (İsteğe Bağlı)

```bash
python train_cnn.py
```

---

## Proje Yapısı

```
Airport_Noise/
│
├── BEATs.py                    # Microsoft/unilm BEATs model tanımı
├── backbone.py                 # BEATs backbone
├── modules.py                  # BEATs yardımcı modüller
├── quantizer.py                # BEATs quantizer
│
├── noise_detector.py           # Ana sistem sınıfı — tüm model inference burada
├── gui_main.py                 # PyQt6 arayüzü
├── mic_map.py                  # Harita bileşeni
│
├── dataset_builder.py          # Manifest oluşturucu
├── env_audio_processor.py      # AMBIENT klip üretici
├── train_beats.py              # BEATs MLP eğitim scripti (v2 — aktif)
├── train_efficientnet.py       # EfficientNet eğitim scripti
├── train_cnn.py                # CNN eğitim scripti
│
├── cache/
│   ├── manifest_v4.csv         # Temel eğitim manifestosu
│   └── manifest_v5.csv         # v4 + onaylı canlı klipler (aktif)
│
├── models/                     # Eğitilmiş model ağırlıkları
│   ├── beats_mlp.pt            # BEATs MLP (aktif)
│   ├── best_efficientnet.pt    # EfficientNet-B0
│   ├── best_efficientnet_finetune.pt
│   ├── efficientnet_label_encoder.pkl
│   ├── efficientnet_meta.pkl
│   ├── best_cnn.pt
│   ├── cnn_label_encoder.pkl
│   ├── best_model.pkl          # SVM
│   └── label_encoder.pkl
│
└── outputs/
    └── training_beats/         # Eğitim grafikleri ve confusion matrix'ler
```

---

## Model Performansı

| Model | F1 Macro | Split Yöntemi |
|---|---|---|
| SVM | — | — |
| CNN | — | — |
| EfficientNet-B0 | 0.8859 | Random |
| **BEATs MLP v2** | **0.9876** | Group-aware (leakage-free) |

---

## Mimari Özeti

```
Mikrofon / Dosya
      │
      ▼
  Rolling Buffer (5s pencere)
      │
      ├──► EfficientNet-B0 ──► Softmax
      │         (Mel Spectrogram)
      │
      └──► BEATs Encoder (frozen) ──► MLP ──► Softmax
                (768-dim embedding)
                       │
                       ▼
                Ensemble (α=0.5)
                       │
                       ▼
            Majority Voting (n=5)
                       │
                       ▼
              Tahmin + Güven Skoru
```

---

## Notlar

- `manifest_v5.csv` içindeki canlı klip yolları (`D:\Airport_Live_Clips\approved\...`) bu makineye özgüdür. Başka bir makinede eğitim yapılacaksa `manifest_v4.csv` kullanılması önerilir veya canlı klipler yeniden toplanmalıdır.
- BEATs embedding cache dosyaları (`.pkl`, toplam ~186 MB) repoya dahil edilmemiştir. `train_beats.py` ilk çalıştırmada otomatik oluşturur.
- CUDA bulunamazsa sistem CPU moduna düşer; BEATs embedding hesaplama çok uzun sürer.
