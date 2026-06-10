# Havalimanı Gürültü Tespit Sistemi — Devam Promptu v5

> ⚠️  YENİ SOHBETTE KULLANMA TALİMATI:
> 1. Bu MD'yi yapıştır
> 2. Ne yapmak istediğini söyle
> 3. Asistan senden **ilgili dosyaları** isteyecek — önce dosyaları gör, sonra kod yaz
> 4. **Dosyaları görmeden kod yazma**

---

## Proje Özeti

TÜBİTAK projesi. Havalimanı ortamında gerçek zamanlı çevresel ses sınıflandırma.
**SVM pipeline tamamlandı ve sağlıklı.** CNN eğitimi uzun süredir AMBIENT sınıfı
yüzünden sorunlu — kök neden: `env_audio_processor.py`'deki overflow sorunu
(aşağıda detaylı açıklandı).

**Aktif Sınıflar (5):**
`AIRCRAFT` / `AMBIENT` / `SPEECH` / `TRAFFIC` / `WIND`

> NOT: Geçmişte WIND→AMBIENT birleştirmesi denendi, olumsuz etki yaptı.
> WIND tekrar ayrı sınıf olarak geri alındı. `dataset_builder_v3.py`'de
> `ESC50_LABEL_MAP` buna göre güncellenmiş durumda.

---

## Klasör Yapısı

```
C:\Users\Fatih\Desktop\TUBITAK\
├── Airport_Noise\
│   ├── noise_detector.py          ← CNN entegreli ✅
│   ├── main.py                    ✅
│   ├── env_audio_processor.py     ⚠️  SORUNLU — bkz. Ana Sorun
│   ├── dataset_builder_v3.py      ✅ (WIND ayrı sınıf)
│   ├── train_model_v3.py          ✅ (sadece SVM)
│   ├── train_cnn.py               ⚠️  SORUNLU — bkz. Ana Sorun
│   ├── compare_models.py          ✅
│   ├── models\
│   │   ├── best_model.pkl         ← SVM (5 sınıf) ✅
│   │   ├── label_encoder.pkl      ✅
│   │   ├── training_meta.pkl      ✅
│   │   ├── best_cnn.pt            ← CNN ⚠️ (AMBIENT sorunu var)
│   │   ├── cnn_label_encoder.pkl
│   │   └── cnn_meta.pkl
│   ├── cache\
│   │   ├── features_v3.pkl        ← 264-boyut, 5 sınıf
│   │   └── manifest_v3.csv        ← path + label
│   └── outputs\
│       ├── training_v3\
│       ├── training_cnn\
│       └── comparison\
│
├── Dataset_Airplane\
│   ├── audio\audio\               ← AIRCRAFT WAV (1895 dosya)
│   ├── env_audio\env_audio\       ← 6 sürekli kayıt WAV (AMBIENT hammadde)
│   ├── env_clips\                 ← AMBIENT klipleri (sayı değişken, bkz. sorun)
│   ├── env_audio_manifest.csv
│   ├── environment_class_mappings.csv   ← ⚠️ CSV formatı kritik, bkz. aşağısı
│   └── sample_meta.csv
│
└── Dataset_ESC50\
    ├── audio\audio\
    └── esc50.csv
```

---

## Mevcut Performans

### SVM — Sağlıklı ✅
| Sınıf    | F1    |
|----------|-------|
| AIRCRAFT | 0.965 |
| AMBIENT  | 0.969 |
| SPEECH   | 0.960 |
| TRAFFIC  | 0.791 |
| **Macro**| **0.897** |

### CNN — Hedef (eski çalışan versiyonda ulaşılan)
```
Toplam örnek: 3434
Sınıflar: AIRCRAFT(1975) AMBIENT(979) WIND(160) SPEECH(160) TRAFFIC(160)
Test F1 Macro: 0.858   ← bu hedeftir, bu sohbette ulaşılamadı
```

### CNN — Şu Anki Durum ❌
AMBIENT her denemede ya 0.00 ya da 0.99 recall veriyor ama ikisi de yanlış
(model tüm AMBIENT'i AIRCRAFT'a atıyor). Tüm denemelerde val F1 < 0.25.

---

## ⚠️  ANA SORUN: AMBIENT Veri Kirliliği + Overflow

### environment_class_mappings.csv Formatı
```
720 satır × 6 sütun
Satır 0: "0,1,2,3,4,5" (başlık)
Col 0, Col 1: 0=temiz, 1=uçak, "ignore"=kirli
Col 2-5: diğer sınıflar (kullanılmıyor)

Temiz segment (col0=0 AND col1=0): ~412 / 720 (%57)
Kirli segment: ~308 / 720 (%43)
```

### env_audio_processor.py — Sorunun Kökü

CSV sadece 720 satır kapsıyor. Ama 6 env_audio dosyasının toplam hop sayısı
bundan çok daha fazla (dosya başına değişiyor, henüz ölçülmedi).

**Bu sohbette yapılan overflow düzeltmesi yanlış çalıştı:**
```python
# YANLIŞ (bu sohbette eklendi — overflow'u "temiz" sayıyor):
if global_idx >= len(seg_flags):
    is_clean = True   # ← CSV kapsamı dışındaki segmentleri temiz kabul etti
                       # Sonuç: 2131 AMBIENT klip, büyük çoğunluğu doğrulanmamış
                       # Model AMBIENT=AIRCRAFT öğrendi, val F1 ~0.17

# DOĞRU davranış (eski v1'de):
if global_idx >= len(seg_flags):
    segments_consumed += 1
    start_sample += hop_samples
    continue   # CSV dışını atla
```

**Neden CSV 720 satır ama daha fazla hop üretiliyor?**
Bu sohbette ölçülmedi. Yeni sohbette ÖNCE şu komutu çalıştır:
```bash
cd C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise
python -c "
import librosa, os
audio_dirs = [
    r'Dataset_Airplane\env_audio\env_audio',
    r'Dataset_Airplane\env_audio',
]
audio_dir = next(p for p in audio_dirs if os.path.isdir(p))
files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
SR, HOP, CLIP = 22050, 2.5, 5.0
total_hops = 0
for f in files:
    path = os.path.join(audio_dir, f)
    dur = librosa.get_duration(path=path)
    hops = max(0, int((dur - CLIP) / HOP) + 1)
    print(f'  {f:35s}  {dur/60:5.1f}dk  {hops:4d} hop')
    total_hops += hops
print(f'\n  Toplam hop : {total_hops}')
print(f'  CSV satir  : 720')
print(f'  Overflow   : {max(0, total_hops - 720)}')
"
```
Bu çıktı olmadan env_audio_processor'a dokunma.

### Mevcut Manifest Durumu (şu an)
```
AIRCRAFT    1975
AMBIENT      210   ← doğru sayı belirsiz, son çalıştırmaya göre değişiyor
WIND         160
SPEECH       160
TRAFFIC      160
```
Eski başarılı versiyonda AMBIENT=979 idi. Şu an 87-2131 arasında gidip geliyor.

---

## Özellik Vektörü — KRİTİK (264 boyut)

```python
# dataset_builder_v3.py ve noise_detector.py'de BİREBİR AYNI olmalı
SR = 22050;  N_MFCC = 40;  N_FFT = 2048;  HOP_FFT = 512;  CLIP_DUR = 5.0
# Boyut: MFCC×3×2=240 + chroma=12 + spectral×4×2=8 + zcr+rms×2=4 = 264
```

---

## CNN Mimarisi — AirportCNN

```python
# Giriş: (B, 1, 128, ~216)  — Mel Spectrogram
# train_cnn.py içinde, şu an bu halde:

class AirportCNN(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2), nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2), nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128*4*4, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )
```

---

## train_cnn.py — Güncel Kritik Parametreler

```python
SR=22050; N_MELS=128; N_FFT=2048; HOP_FFT=512; DURATION=5.0

EPOCHS=60; PATIENCE=12; BATCH_SIZE=64
LR=1e-4; WEIGHT_DECAY=1e-4
WARMUP_EPOCHS=8         # ilk 8 epoch LR kademeli artar
SAMPLER_MAX_RATIO=6.0   # sqrt-capped WeightedRandomSampler
DEBUG_CM_EVERY=5        # her 5 epoch'ta per-class recall yazdırır

# Scheduler: CosineAnnealingLR (warmup sonrası)
# Ctrl+C → anlık confusion matrix kaydedip çıkar
# Her 5 epoch'ta per-class recall terminale yazdırılır
```

### Augmentation (şu an aktif)
```python
ops = np.random.choice(["noise", "gain"], size=2, replace=False)
# "stretch" ve "pitch" kodda var ama ops listesinde yok → çalışmıyor
# + SpecAugment (time mask + freq mask)
```

### WeightedRandomSampler (sqrt + max_ratio=6.0)
```python
# 1/count yerine sqrt(max_cnt/cnt), max 6x ile kırpılmış
# Son çalıştırmada batch payları:
#   AIRCRAFT ~39.9%   AMBIENT ~26.0%   SPEECH/TRAFFIC/WIND ~9.8% her biri
```

---

## Bilinen Sorunlar (Öncelik Sırasıyla)

### 1. ⛔ AMBIENT Veri Sorunu — ÇÖZÜLMEDİ

**Sorun:** env_audio_processor.py'nin overflow davranışı tutarsız.
- Strict mode (overflow atla): 87 AMBIENT klip → model AIRCRAFT öğreniyor
- Overflow dahil (temiz say): 2131 AMBIENT klip → hepsi kirli, model yine AIRCRAFT öğreniyor

**Yapılacak ilk iş:**
1. Yukarıdaki hop sayısı scriptini çalıştır
2. Sonuca göre env_audio_processor'da per-file CSV satır aralığı hesapla
3. Her dosya için CSV'nin gerçekten kaçıncı-kaçıncı satırları kapsadığını belirle
4. Doğrulanmış temiz klip sayısı 300'ün altındaysa augmentation ile telafi et

**Alternatif:** env_clips/ içindeki tüm .wav dosyalarını mevcut SVM modeli ile
tarayıp AIRCRAFT tahmin edilenleri sil (SVM ikinci katman doğrulama).

### 2. ⚠️  Train/Val Kararsızlığı

Val loss ilk epoch'larda 5-10'a fırlıyor. Warmup LR çok küçük başlıyor (1.25e-5),
ama bu batch norm güncelleme sorunlarına yol açıyor.

**Olası düzeltme:** Warmup'ı LR/8 yerine LR/4 ile başlat; ya da warmup epoch sayısını
8'den 4'e düşür.

### 3. ⚠️  SPEECH/TRAFFIC/WIND Karışıklığı

Her biri 160 örnek. F1: SPEECH~0.62-0.88, TRAFFIC~0.62-0.71, WIND~0.71-0.75.
AMBIENT sorunu çözüldükten sonra ele alınacak.

---

## dataset_builder_v3.py — ESC50_LABEL_MAP (güncel)

```python
ESC50_LABEL_MAP = {
    "airplane":     "AIRCRAFT",
    "helicopter":   "AIRCRAFT",
    "car_horn":     "TRAFFIC",
    "engine":       "TRAFFIC",
    "train":        "TRAFFIC",
    "siren":        "TRAFFIC",
    "wind":         "WIND",       # AMBIENT'e birleştirilmişti, geri alındı
    "rain":         "WIND",
    "thunderstorm": "WIND",
    "sea_waves":    "WIND",
    "clapping":     "SPEECH",
    "laughing":     "SPEECH",
    "crying_baby":  "SPEECH",
    "crowd":        "SPEECH",
    "footsteps":    "SPEECH",
}
```

---

## Sonraki Adımlar (Sıraya Göre)

- [ ] **ÖNCE:** Hop sayısı scriptini çalıştır → çıktıyı paylaş
- [ ] **env_audio_processor.py** dosyasını asistana gönder → per-file CSV mapping düzelt
- [ ] Temiz AMBIENT sayısı yeterliyse (≥300): cache sil → dataset yeniden oluştur
- [ ] Temiz AMBIENT < 300 ise: SVM ikinci katman doğrulama VEYA agresif augmentation
- [ ] **train_cnn.py** dosyasını asistana gönder → warmup düzelt → yeniden eğit
- [ ] Confusion matrix'te tüm sınıflar ≥0.75 recall olunca: compare_models.py çalıştır
- [ ] Ensemble (SVM + CNN) — CNN stabil olduktan sonra

---

## Kurulum & Çalıştırma

```bash
cd C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise
venv\Scripts\activate

# Sıfırdan başlarken:
del cache\features_v3.pkl
del cache\manifest_v3.csv
rmdir /s /q Dataset_Airplane\env_clips

python env_audio_processor.py    # AMBIENT klipleri üret
python dataset_builder_v3.py     # özellik matrisi
python train_model_v3.py         # SVM (~2 dk)
python train_cnn.py              # CNN (CPU: ~2 saat)

# Mevcut terminal dizini sorunu — HATA ALIYORSAN:
# "No such file or directory: 'cache/manifest_v3.csv'"
# → Önce cd komutu gerekli:
cd C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise

# GPU (GTX 1650, CUDA 13.1 driver):
pip uninstall torch torchaudio -y
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Bu Sohbette Yapılan Değişikliklerin Özeti

| Dosya | Değişiklik | Sonuç |
|-------|-----------|-------|
| env_audio_processor.py | Zaman tabanlı → segment-index tabanlı filtre | ✅ Doğru yaklaşım |
| env_audio_processor.py | Overflow → "temiz say" | ❌ 2131 kirli klip |
| env_audio_processor.py | MIN_SILENCE_DB -45→-52, BUFFER 2s→1s | ✅ Mantıklı |
| dataset_builder_v3.py | WIND→AMBIENT birleştirme | ❌ Geri alındı |
| dataset_builder_v3.py | WIND tekrar ayrı sınıf | ✅ |
| train_cnn.py | WeightedRandomSampler 1/cnt → sqrt capped | ✅ |
| train_cnn.py | LR 3e-4 → 1e-4, warmup 8 epoch | ✅ |
| train_cnn.py | ReduceLROnPlateau → CosineAnnealingLR | ✅ |
| train_cnn.py | Dropout2d 0.1 → 0.25 | ✅ |
| train_cnn.py | Ctrl+C → anlık CM | ✅ |
| train_cnn.py | Her 5 epoch per-class recall | ✅ |

---

## Yeni Sohbet Notu

Asistan şu sırayı izlemeli:
1. Hop sayısı scriptini çalıştırmasını iste, çıktıyı al
2. `env_audio_processor.py` dosyasını iste → per-file mapping düzelt
3. `train_cnn.py` dosyasını iste → warmup sorunu düzelt
4. Her değişiklikten sonra sadece değiştirilen bloğu göster, tüm dosyayı yeniden yazma
5. Confusion matrix ve training curves geldiğinde önce AMBIENT recall'a bak
