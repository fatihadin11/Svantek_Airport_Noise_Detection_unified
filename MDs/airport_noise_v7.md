# Havalimanı Gürültü Tespit Sistemi — Devam Promptu v7

> ⚠️ YENİ SOHBETTE KULLANIM TALİMATI:
> 1. Bu MD'yi yapıştır
> 2. Ne yapmak istediğini söyle
> 3. Asistan senden **ilgili dosyayı** isteyecek — dosyayı gördükten sonra kod yazar
> 4. **Dosyayı görmeden asla kod yazma**
> 5. Değişiklik önerirken **tüm dosyayı** ver, sadece bloğu değil

---

## Proje Özeti

TÜBİTAK projesi. Havalimanı ortamında gerçek zamanlı çevresel ses sınıflandırması.

**Hedef:** F1 Macro ≥ 0.858 (CNN için)
**Sonraki adım:** F1 yeterli düzeye gelince → pretrained model entegrasyonu (EfficientNet / VGGish / BirdNET tabanlı transfer learning)

**Aktif Sınıflar (5):**
`AIRCRAFT` / `AMBIENT` / `SPEECH` / `TRAFFIC` / `WIND`

**Mevcut Durum:**
- SVM pipeline: ✅ Sağlıklı, F1 Macro ~0.897
- CNN pipeline: ⚠️ Çalışıyor ama F1 henüz 0.858 hedefine ulaşmadı
- EfficientNet: ✅ Eğitildi, val F1=0.877 — dış testte F1=0.709
- CNN dış test: ❌ F1=0.204, negatif recall=0.00 (domain shift sorunu)
- AMBIENT veri sorunu: ✅ Çözüldü (env_audio_processor.py overflow fix + per-file CSV mapping)
- train_cnn.py: ✅ Kararlı versiyon aktif

---

## Klasör Yapısı

```
C:\Users\Fatih\Desktop\TUBITAK\
├── Airport_Noise\
│   ├── noise_detector.py          ✅ CNN entegreli
│   ├── main.py                    ✅
│   ├── env_audio_processor.py     ✅ Overflow fix + per-file CSV mapping
│   ├── dataset_builder_v3.py      ✅ WIND ayrı sınıf
│   ├── train_model_v3.py          ✅ SVM
│   ├── train_cnn.py               ✅ Kararlı versiyon
│   ├── train_efficientnet.py      ✅ EfficientNet fine-tune
│   ├── test_external.py           ✅ Dış dataset testi (binary: aircraft/negative)
│   ├── compare_models.py          ✅
│   ├── models\
│   │   ├── best_model.pkl              ✅ SVM (5 sınıf)
│   │   ├── label_encoder.pkl           ✅
│   │   ├── training_meta.pkl           ✅
│   │   ├── best_cnn.pt                 ⚠️ val F1=0.856, dış test zayıf
│   │   ├── cnn_label_encoder.pkl       ✅
│   │   ├── cnn_meta.pkl                ✅
│   │   ├── best_efficientnet.pt        ✅ val F1=0.877
│   │   └── efficientnet_label_encoder.pkl ✅
│   ├── cache\
│   │   ├── features_v3.pkl        264-boyut, 5 sınıf
│   │   └── manifest_v3.csv        path + label
│   └── outputs\
│       ├── training_v3\
│       ├── training_cnn\
│       ├── training_efficientnet\
│       └── external_test\         ← test_external.py çıktıları
│
├── Dataset_Airplane\
│   ├── audio\audio\               AIRCRAFT WAV (1895 dosya)
│   ├── env_audio\env_audio\       6 sürekli kayıt WAV (AMBIENT hammadde)
│   ├── env_clips\                 AMBIENT klipleri (~322 klip, -65 dBFS eşiği)
│   ├── env_audio_manifest.csv
│   ├── environment_class_mappings.csv   ← 720 satır × 6 dosya = 120 satır/dosya
│   └── sample_meta.csv
│
├── Dataset_ESC50\
│   ├── audio\audio\
│   └── esc50.csv
│
└── Airport_Noise\Test_Folder\     ← Dış test dataseti
    ├── aircraft-test\             54 WAV  (uçak, test split)
    ├── aircraft-train\            54 WAV  (uçak, train split)
    ├── negative-test\             139 WAV (negatif, test split)
    ├── negative-train\            139 WAV (negatif, train split)
    └── labels.csv                 386 kayıt (filename, class, duration, sample_rate, dtype, split)
```

---

## Mevcut Performans

### SVM ✅
| Sınıf | F1 |
|---|---|
| AIRCRAFT | 0.965 |
| AMBIENT | 0.969 |
| SPEECH | 0.960 |
| TRAFFIC | 0.791 |
| **Macro** | **0.897** |

### CNN ⚠️
| Değerlendirme | Sonuç |
|---|---|
| Val F1 Macro (iç) | ~0.856 — hedef 0.858 |
| Dış Test F1 | 0.204 |
| Dış Test Aircraft Recall | 0.917 |
| Dış Test Negative Recall | 0.000 |

### EfficientNet ✅ (iç) / ⚠️ (dış)
| Değerlendirme | Sonuç |
|---|---|
| Val F1 Macro (iç) | 0.877 |
| Dış Test F1 | 0.709 |
| Dış Test Aircraft Recall | 0.972 |
| Dış Test Aircraft Precision | 0.498 |
| Önerilen eşik | 0.17 (sweep ile bulundu) |

---

## Dataset Dağılımı (Mevcut)

```
AIRCRAFT    1975
AMBIENT      322  (322 klip, -65 dBFS, env_audio_processor.py ile üretildi)
WIND         160
SPEECH       160
TRAFFIC      160
TOPLAM      2777
```

---

## Özellik Vektörü — KRİTİK (264 boyut, SVM için)

```python
# dataset_builder_v3.py ve noise_detector.py'de BİREBİR AYNI olmalı
SR = 22050;  N_MFCC = 40;  N_FFT = 2048;  HOP_FFT = 512;  CLIP_DUR = 5.0
# Boyut: MFCC×3×2=240 + chroma=12 + spectral×4×2=8 + zcr+rms×2=4 = 264
```

---

## ✅ Kararlı train_cnn.py — Kritik Parametreler

```python
SR        = 22050
N_MELS    = 128
N_FFT     = 2048
HOP_FFT   = 512
DURATION  = 5.0

EPOCHS        = 50
PATIENCE      = 10
BATCH_SIZE    = 32
LR            = 1e-3        # ← 1e-4 değil, 1e-3
WEIGHT_DECAY  = 1e-4

MANUAL_CLASS_WEIGHTS = {
    "AIRCRAFT": 0.5,
    "SPEECH":   2.0,
    "TRAFFIC":  2.5,
    "WIND":     2.5,
    "AMBIENT":  2.0,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Model Mimarisi

```python
class AirportCNN(nn.Module):
    def __init__(self, n_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )
```

### DataLoader — WeightedRandomSampler YOK

```python
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0, pin_memory=False)
```

### Loss & Scheduler

```python
weights   = torch.FloatTensor([MANUAL_CLASS_WEIGHTS[labels[i]] for i in range(len(labels))]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
```

---

## 🔴 Öncelikli Görev: CNN Geliştirme

### Dış Test Bulgusu (386 dosya: 108 aircraft, 278 negative)

CNN 5-sınıf çıktısında negatif örneklerin büyük çoğunluğu WIND'e düşüyor.
AIRCRAFT softmax skoru sabit yüksek kalıyor → her şeyi aircraft sayıyor.
Sorun: eğitim AMBIENT/WIND örnekleri çok spesifik kaynaklardan geldi,
dış dünyanın geniş ses evrenini tanımıyor → **domain shift**.

### Yapılacaklar (sıraya göre)

- [ ] Dış dataset'ten negatif örnekleri AMBIENT/WIND eğitim setine ekle → fine-tune
- [ ] Waveform augmentation dene: time stretch, pitch shift (dikkatli test et)
---

## ⛔ GEÇMİŞTE YAPILAN HATALAR — BİR DAHA YAPMA

### Hata 1 — WeightedRandomSampler + BatchNorm

WeightedRandomSampler eklendi → train/val batch dağılımı farklılaştı → BatchNorm
running stats tutarsızlaştı → val loss epoch 3'te 5.6'ya fırladı → 20 epoch kurtarılamadı.

**Kural:** BatchNorm varken Sampler KULLANMA. Ya Sampler → GroupNorm, ya da
BatchNorm → sadece loss weight. ✅ Mevcut: loss weight kullanılıyor.

### Hata 2 — env_audio_processor.py overflow

Global index biriktirme → dosya 2-6 tamamen atlandı → sadece dosya 1'den AMBIENT geldi.
**Düzeltme (aktif):** `seg_flags[i*120 : (i+1)*120]`, `global_seg_offset=0`.

### Hata 3 — Overflow segmentleri "temiz say"

Overflow segmentler `is_clean = True` → 2131 doğrulanmamış AMBIENT klip → model collapse.
**Kural:** CSV kapsamı dışı → her zaman `continue`.

### Hata 4 — AMBIENT manuel 4x duplikasyon + Sampler

Duplikasyon VE Sampler aynı anda → AMBIENT %50+ → model collapse.
**Kural:** İkisi aynı anda kullanılmaz.

### Hata 5 — Warmup çok küçük LR

`1e-4 / 8 = 1.25e-5` → BatchNorm güncelleme sorunu.
**Kural:** Warmup eklenecekse ≥ `LR/4` ile başla, epoch ≤ 4.

### Hata 6 — WIND → AMBIENT birleştirmesi

Performans düştü, geri alındı.
**Kural:** WIND ayrı sınıf kalır, `ESC50_LABEL_MAP`'e dokunma.

### Hata 7 — SVM bozuk pickle

`best_model.pkl` üzerine yazıldı → `invalid load key` hatası.
**Kural:** SVM kullanılacaksa önce `train_model_v3.py` çalıştır.

### Hata 8 — torch.load weights_only eksik

**Düzeltme (aktif):**
```python
ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
```

---

## Sonraki Hedefler (Sıraya Göre)

### 1. CNN F1 ≥ 0.858 — ŞU AN AKTİF
Domain shift sorunu öncelikli. Dış test negatif örnekleri eğitime eklemek
en hızlı kazanımı sağlar, sıfırdan eğitim gerekmez.

### 2. EfficientNet Eşik Optimizasyonu
Threshold sweep 0.17 öneriyor. `test_external.py --threshold 0.17` ile doğrula.

### 3. Ensemble (SVM + CNN veya SVM + EfficientNet) — CNN Stabil Olunca
Soft voting: SVM prob + model prob → ağırlıklı ortalama.
SVM'in negatif tanıma kabiliyeti yüksek, precision sorununu dengeler.

### 4. Pretrained Model Entegrasyonu — CNN F1 ≥ 0.858 OLUNCA
EfficientNet-B0 veya VGGish tabanlı transfer learning.

---

## Kurulum & Çalıştırma

```bash
cd C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise
venv\Scripts\activate

# Sıfırdan başlarken:
del cache\features_v3.pkl
del cache\manifest_v3.csv
rmdir /s /q Dataset_Airplane\env_clips

python env_audio_processor.py    # AMBIENT klipleri (~322 klip)
python dataset_builder_v3.py     # özellik matrisi (2777 örnek, 264 boyut)
python train_model_v3.py         # SVM (~2 dk)
python train_cnn.py              # CNN (CPU: ~1 saat)

# Dış dataset testi:
python test_external.py
python test_external.py --threshold 0.17

# GPU (GTX 1650, CUDA 13.1 driver):
pip uninstall torch torchaudio -y
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Sorun Geçmişi (Özet)

| Sorun | Kök Neden | Çözüm | Durum |
|---|---|---|---|
| AMBIENT recall=0.00 | env_audio_processor global idx overflow | Per-file CSV dilimi | ✅ |
| Val loss spike (1.9→5.6) | WeightedRandomSampler + BatchNorm | Sampler kaldırıldı | ✅ |
| 2131 kirli AMBIENT | Overflow → temiz say | Overflow → atla | ✅ |
| WIND→AMBIENT birleşme | ESC50_LABEL_MAP güncelleme | Geri alındı | ✅ |
| torch.load uyarısı | weights_only eksik | weights_only=False | ✅ |
| SVM bozuk pickle | Üzerine yazıldı | train_model_v3.py tekrar çalıştır | ⚠️ |
| CNN dış test F1=0.20 | Domain shift, negatif örnek yok | Dış dataset ekleme planlandı | 🔄 |
| EfficientNet precision=0.50 | Eşik düşük | Threshold 0.17'ye çek | 🔄 |

---

## Asistan İçin Protokol

1. Kod değişikliği önerilmeden önce **ilgili dosyayı kullanıcıdan iste**
2. Değişiklik yaparken **tüm dosyayı** ver, sadece bloğu değil
3. Her değişiklikten sonra **beklenen çıktıyı** açıkça yaz
4. Confusion matrix veya training log geldiğinde **önce AMBIENT recall'a bak**
5. WeightedRandomSampler + BatchNorm kombinasyonunu **asla önerme**
6. Overflow segmentleri için asla `is_clean = True` yazma
7. Pretrained model entegrasyonunu CNN F1 ≥ 0.858 olmadan başlatma
8. Dış test sonucu geldiğinde **önce 5-sınıf dağılımına bak** (domain shift tespiti)
