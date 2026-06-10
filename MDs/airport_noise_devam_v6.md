# Havalimanı Gürültü Tespit Sistemi — Devam Promptu v6

> ⚠️ YENİ SOHBETTE KULLANIM TALİMATI:
> 1. Bu MD'yi yapıştır
> 2. Ne yapmak istediğini söyle
> 3. Asistan senden **ilgili dosyayı** isteyecek — dosyayı gördükten sonra kod yazar
> 4. **Dosyayı görmeden asla kod yazma**
> 5. Değişiklik önerirken **tüm dosyayı** ver, sadece bloğu değil

---

## Proje Özeti

TÜBİTAK projesi. Havalimanı ortamında gerçek zamanlı çevresel ses sınıflandırması.

**Hedef:** F1 Macro ≥ 0.858 (eski çalışan CNN versiyonunda ulaşılan değer)
**Sonraki adım:** F1 yeterli düzeye gelince → pretrained model entegrasyonu (EfficientNet / VGGish / BirdNET tabanlı transfer learning)

**Aktif Sınıflar (5):**
`AIRCRAFT` / `AMBIENT` / `SPEECH` / `TRAFFIC` / `WIND`

**Mevcut Durum:**
- SVM pipeline: ✅ Sağlıklı, F1 Macro ~0.897
- CNN pipeline: ⚠️ Çalışıyor ama F1 henüz 0.858 hedefine ulaşmadı
- AMBIENT veri sorunu: ✅ Çözüldü (env_audio_processor.py overflow fix + per-file CSV mapping)
- train_cnn.py: ✅ Kararlı versiyon bulundu ve aktif (aşağıda detay)

---

## Klasör Yapısı

```
C:\Users\Fatih\Desktop\TUBITAK\
├── Airport_Noise\
│   ├── noise_detector.py          ✅ CNN entegreli
│   ├── main.py                    ✅
│   ├── env_audio_processor.py     ✅ Overflow fix + per-file CSV mapping uygulandı
│   ├── dataset_builder_v3.py      ✅ WIND ayrı sınıf
│   ├── train_model_v3.py          ✅ SVM
│   ├── train_cnn.py               ✅ Kararlı versiyon (v1 temelli, aşağıda açıklandı)
│   ├── compare_models.py          ✅
│   ├── models\
│   │   ├── best_model.pkl         ✅ SVM (5 sınıf)
│   │   ├── label_encoder.pkl      ✅
│   │   ├── training_meta.pkl      ✅
│   │   ├── best_cnn.pt            ⚠️ Eğitiliyor — henüz hedef F1'e ulaşmadı
│   │   ├── cnn_label_encoder.pkl
│   │   └── cnn_meta.pkl
│   ├── cache\
│   │   ├── features_v3.pkl        264-boyut, 5 sınıf
│   │   └── manifest_v3.csv        path + label
│   └── outputs\
│       ├── training_v3\
│       ├── training_cnn\
│       └── comparison\
│
├── Dataset_Airplane\
│   ├── audio\audio\               AIRCRAFT WAV (1895 dosya)
│   ├── env_audio\env_audio\       6 sürekli kayıt WAV (AMBIENT hammadde)
│   ├── env_clips\                 AMBIENT klipleri (~322 klip, -65 dBFS eşiği)
│   ├── env_audio_manifest.csv
│   ├── environment_class_mappings.csv   ← 720 satır × 6 dosya = 120 satır/dosya
│   └── sample_meta.csv
│
└── Dataset_ESC50\
    ├── audio\audio\
    └── esc50.csv
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

### CNN — Hedef
```
Test F1 Macro: 0.858   ← eski çalışan versiyonda ulaşılan, bu hedef
```

### CNN — Mevcut Durum
Kararlı eğitim var, AMBIENT sorunları çözüldü. Henüz 0.858'e ulaşılmadı.

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

### Model Mimarisi (BatchNorm korundu — nedeni aşağıda)

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
# shuffle=True ile normal DataLoader — WeightedRandomSampler kullanılmıyor
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0, pin_memory=False)
```

### Loss

```python
# MANUAL_CLASS_WEIGHTS ile CrossEntropyLoss — Sampler YOK
weights   = torch.FloatTensor([MANUAL_CLASS_WEIGHTS[labels[i]] for i in range(len(labels))]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
```

### Scheduler — Warmup YOK

```python
# CosineAnnealingLR — warmup yok, doğrudan başlar
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
```

---

## ⛔ GEÇMİŞTE YAPILAN HATALAR — BİR DAHA YAPMA

### Hata 1 — WeightedRandomSampler + BatchNorm kombinasyonu

**Ne oldu:** WeightedRandomSampler eklendi → train batch dağılımı (~%44 AIRCRAFT)
ile val dağılımı (~%71 AIRCRAFT) farklılaştı → BatchNorm running stats tutarsızlaştı
→ val loss epoch 1'de 1.9, epoch 3'te 5.6'ya fırladı → AIRCRAFT ve AMBIENT recall
sıfıra düştü, 20 epoch boyunca kurtarılamadı.

**Kural:** BatchNorm ile WeightedRandomSampler bir arada KULLANMA.
İkisinden biri seçilmeli:
- Sampler kullanacaksan → GroupNorm'a geç (running stats tutmaz)
- BatchNorm tutacaksan → Sampler kullanma, sadece loss weight kullan ✅ (mevcut)

### Hata 2 — env_audio_processor.py overflow yönetimi

**Ne oldu:** 6 WAV dosyası toplam 8634 hop üretiyor, CSV sadece 720 satır
(6 dosya × 120 satır). Eski kod global index biriktiriyordu:
- Dosya 1: idx 0-1438, CSV 0-719 → idx 720-1438 overflow → atla ✓
- Dosya 2: idx 1439-2877 → TAMAMI overflow → tüm dosya atlandı ❌
- Dosya 3-6: Tamamen atlandı → sadece dosya 1'den AMBIENT geldi

**Düzeltme:** Her dosya için CSV dilimi = `seg_flags[i*120 : (i+1)*120]`
`global_seg_offset=0` olarak geçirilir. Bu fix uygulandı ve aktif.

### Hata 3 — AMBIENT veri kirliliği (overflow → "temiz say")

**Ne oldu:** Overflow segmentleri `is_clean = True` olarak işaretlendi
→ 2131 AMBIENT klip üretildi ama büyük çoğunluğu doğrulanmamış kayıtlardı
→ model AMBIENT=AIRCRAFT öğrendi.

**Kural:** CSV kapsamı dışındaki segmentler her zaman `continue` (atla).
Asla "temiz say" yapma.

### Hata 4 — AMBIENT manuel 4x duplikasyon + Sampler çakışması

**Ne oldu:** `train_records = train_records + ambient_records * 3` satırı eklendi.
Sampler zaten AMBIENT'i dengeliyordu. Üstüne 4x duplikasyon → eğitim
setinde AMBIENT oranı %50+'ye çıktı → model collapse.

**Kural:** Duplikasyon VE Sampler aynı anda kullanılmaz.

### Hata 5 — Warmup LR çok küçük başlatıldı

**Ne oldu:** `LR / WARMUP_EPOCHS = 1e-4 / 8 = 1.25e-5` ile başlandı.
BatchNorm güncelleme sorunlarına yol açtı, val loss ilk epoch'larda dengesiz kaldı.

**Kural:** Kararlı versiyonda warmup yok, doğrudan `LR=1e-3` ile başlanıyor.
Warmup eklenecekse en az `LR/4` ile başlamalı, epoch sayısı ≤ 4 olmalı.

### Hata 6 — WIND → AMBIENT birleştirmesi

**Ne oldu:** ESC50_LABEL_MAP'te WIND sesleri AMBIENT'e taşındı.
Model performansı düştü, geri alındı.

**Kural:** WIND ayrı sınıf olarak kalır. `dataset_builder_v3.py`'de
`ESC50_LABEL_MAP`'e dokunma.

### Hata 7 — SVM modeli bozuk pickle

**Ne oldu:** `env_audio_processor.py --svm` çalıştırıldığında
`invalid load key, '\x0c'` hatası → SVM yüklenemedi.
Modeli yeniden eğitmeden `best_model.pkl` üzerine yazılmış olabilir.

**Kural:** SVM ikinci katman doğrulama kullanılacaksa önce
`train_model_v3.py` çalıştırılarak model yenilenmeli.

### Hata 8 — torch.load weights_only parametresi eksik

**Ne oldu:** `torch.load(path, map_location=device)` → yeni PyTorch
sürümlerinde FutureWarning veya hata.

**Düzeltme (aktif):**
```python
ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
```

---

## Sonraki Hedefler (Sıraya Göre)

### 1. F1 Macro artırma — ŞU AN AKTİF
Yapılacaklar (asistan dosyayı görmeden öneri vermemeli):
- [ ] Ctrl+C → anlık confusion matrix + eğitim durdurma
- [ ] Her N epoch'ta terminale per-class recall, val/train loss delta
- [ ] Eğitimi ortadan bölme (checkpoint'ten devam)
- [ ] Waveform augmentation (stretch, pitch) — dikkatli test edilmeli
- [ ] LR tuning denemeleri (1e-3 → cosine min 1e-6)
- [ ] Dropout ayarı (şu an conv=0.1, fc=0.5)

### 2. Pretrained Model Entegrasyonu — F1 ≥ 0.858 OLUNCA
- EfficientNet-B0 veya VGGish tabanlı transfer learning
- Mel spectrogram yerine ham waveform → VGGish feature extraction
- Freeze → fine-tune stratejisi

### 3. Ensemble (SVM + CNN) — CNN Stabil Olunca
- Soft voting: SVM prob + CNN prob → ağırlıklı ortalama
- compare_models.py ile kıyaslama

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

# GPU (GTX 1650, CUDA 13.1 driver):
pip uninstall torch torchaudio -y
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Bu Sohbette Çözülen Sorunlar (Özet)

| Sorun | Kök Neden | Çözüm | Durum |
|---|---|---|---|
| AMBIENT recall=0.00 | env_audio_processor global idx overflow | Per-file CSV dilimi | ✅ |
| Val loss spike (1.9→5.6) | WeightedRandomSampler + BatchNorm çakışması | Sampler kaldırıldı, loss weight korundu | ✅ |
| 2131 kirli AMBIENT | Overflow → temiz say | Overflow → atla | ✅ |
| WIND→AMBIENT birleşme | ESC50_LABEL_MAP güncelleme | Geri alındı | ✅ |
| torch.load uyarısı | weights_only parametresi eksik | weights_only=False eklendi | ✅ |
| SVM bozuk pickle | Üzerine yazıldı | train_model_v3.py yeniden çalıştır | ⚠️ |
| F1 Macro < 0.858 | Henüz devam ediyor | Sıradaki hedef | 🔄 |

---

## Asistan İçin Protokol

1. Kod değişikliği önerilmeden önce **ilgili dosyayı kullanıcıdan iste**
2. Değişiklik yaparken **tüm dosyayı** ver, sadece bloğu değil
3. Her değişiklikten sonra **beklenen çıktıyı** açıkça yaz
4. Confusion matrix veya training log geldiğinde **önce AMBIENT recall'a bak**
5. WeightedRandomSampler + BatchNorm kombinasyonunu **asla önerme**
6. Overflow segmentleri için asla `is_clean = True` yazma
7. Pretrained model entegrasyonunu F1 ≥ 0.858 olmadan başlatma
