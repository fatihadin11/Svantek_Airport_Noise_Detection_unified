# Havalimanı Gürültü Tespit Sistemi — Devam Promptu v4

> ⚠️ YENİ SOHBETTE KULLANILACAKSA:
> 1. Bu MD dosyasını yapıştır
> 2. Hangi değişikliği yapacağını söyle
> 3. Asistan senden ilgili dosyaları isteyecek — o zaman paylaş
> 4. **Dosyaları görmeden kod yazma**

---

## Proje Özeti

Havalimanı ortamında gerçek zamanlı çevresel gürültü sınıflandırma sistemi. TÜBİTAK projesi.
SVM pipeline tamamlandı. CNN eğitimi yapıldı, iyileştirme aşamasındayız.

**Aktif Sınıflar (4):** AIRCRAFT / AMBIENT / SPEECH / TRAFFIC
*(WIND → AMBIENT'e birleştirildi — dataset_builder'da ESC50_LABEL_MAP güncellendi)*

---

## Klasör Yapısı

```
C:\Users\Fatih\Desktop\TUBITAK\
├── Airport_Noise\
│   ├── noise_detector.py          ← CNN entegreli, CNN>SVM>kural önceliği ✅
│   ├── main.py                    ✅
│   ├── env_audio_processor.py     ✅
│   ├── dataset_builder_v3.py      ← WIND→AMBIENT değişikliği yapıldı ✅
│   ├── train_model_v3.py          ← RF/GBM/GridSearch kaldırıldı, sadece SVM ✅
│   ├── train_cnn.py               ← WeightedRandomSampler + augmentation ✅
│   │                                 ⚠️ Overfitting sorunu var (bkz. Bilinen Sorunlar)
│   ├── compare_models.py          ✅
│   ├── models\
│   │   ├── best_model.pkl         ← SVM (4 sınıf) ✅
│   │   ├── label_encoder.pkl      ✅
│   │   ├── training_meta.pkl      ✅
│   │   ├── best_cnn.pt            ← CNN (4 sınıf, kirli AMBIENT ile) ⚠️
│   │   ├── cnn_label_encoder.pkl  ✅
│   │   └── cnn_meta.pkl           ✅
│   ├── cache\
│   │   ├── features_v3.pkl        ← 264-boyut, 4 sınıf ✅
│   │   └── manifest_v3.csv        ✅
│   └── outputs\
│       ├── training_v3\           ← SVM confusion matrix
│       ├── training_cnn\          ← CNN eğitim eğrileri + confusion matrix
│       └── comparison\            ← SVM vs CNN karşılaştırma
│
├── Dataset_Airplane\
│   ├── audio\audio\               ← AIRCRAFT WAV (1895 dosya)
│   ├── env_audio\env_audio\       ← 6 sürekli kayıt WAV (AMBIENT hammadde)
│   ├── env_clips\                 ← 979 AMBIENT klip ✅
│   ├── env_audio_manifest.csv     ✅
│   ├── environment_class_mappings.csv  ← ⚠️ BAŞLIKSIZ CSV (ana sorun, bkz. aşağısı)
│   └── sample_meta.csv
│
└── Dataset_ESC50\
    ├── audio\audio\
    └── esc50.csv
```

---

## Mevcut Performans

### SVM (referans, sağlıklı)
| Sınıf    | F1    |
|----------|-------|
| AIRCRAFT | 0.965 |
| AMBIENT  | 0.969 |
| SPEECH   | 0.960 |
| TRAFFIC  | 0.791 |
| **Macro**| **0.897** |

### CNN v2 (sorunlu — bkz. Bilinen Sorunlar)
| Sınıf    | F1    |
|----------|-------|
| AIRCRAFT | —     |
| AMBIENT  | ~0.15 recall |
| SPEECH   | 0.88  |
| TRAFFIC  | 0.83  |
| **Macro**| **~0.55** |

---

## Özellik Vektörü (264 boyut) — KRİTİK

```python
# dataset_builder_v3.py ve noise_detector.py'de BİREBİR AYNI olmalı
N_MFCC = 40;  N_FFT = 2048;  HOP_FFT = 512;  SR = 22050
```

---

## Bilinen Sorunlar

### 1. ⚠️ AMBIENT Veri Kirliliği — ANA HEDEF

**Sorun:** `env_clips/` içindeki 979 AMBIENT klibinin ~50-100 tanesi uçak sesi içeriyor.
`environment_class_mappings.csv` başlıksız olduğu için hangi env_audio segmentlerinin
uçak sesi içerdiği tespit edilemedi. CNN bunu öğrendi:
AMBIENT görünce → AIRCRAFT tahmin ediyor (recall: 0.15).

**Çözüm planı:**
```
1. environment_class_mappings.csv içeriğini incele
   → sütun anlamlarını çıkar
   → uçak sesi içeren segmentleri filtrele
2. env_audio_processor.py'yi güncelle
   → kirli klipler env_clips'ten çıkarılsın
3. cache/ klasörünü sil, dataset_builder_v3.py yeniden çalıştır
4. train_cnn.py yeniden çalıştır
```

**Yapılacak ilk iş:** `environment_class_mappings.csv` içeriğini paylaş.
İlk birkaç satır + sütun sayısı yeterli.

### 2. ⚠️ CNN Overfitting

Train acc %90, Val acc %63'te sabit kalıyor. Val loss gürültülü.
AMBIENT kirliliği düzelince bu da büyük ölçüde çözülecek.
Düzelmezse uygulanacak değişiklikler aşağıda.

### 3. ⚠️ TRAFFIC/SPEECH Az Veri

160'ar örnek. WeightedRandomSampler ile telafi ediliyor ama
daha fazla veri bulunabilirse iyileşir.

---

## train_cnn.py — Güncel Kritik Bloklar

### WeightedRandomSampler (çalışıyor ✅)

```python
label_counts   = Counter(r["label_enc"] for r in train_records)
sample_weights = [1.0 / label_counts[r["label_enc"]] for r in train_records]
sampler        = WeightedRandomSampler(
    weights     = sample_weights,
    num_samples = len(train_records),
    replacement = True
)
# shuffle=False — sampler ile çelişir
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          sampler=sampler, num_workers=0, pin_memory=False)
```

### Loss — weight KALDIRILDI (sampler ile çift ceza yaratıyordu)

```python
criterion = nn.CrossEntropyLoss()   # weight yok — sampler zaten dengeliyor
```

### Augmentation — sadece noise+gain (stretch/pitch kaldırıldı — çok yavaş)

```python
def augment_waveform(y, sr=SR):
    ops = np.random.choice(["noise", "gain"], size=2, replace=False)
    for op in ops:
        if op == "noise":
            sigma = np.random.uniform(0.002, 0.008)
            y = y + (np.random.randn(len(y)) * sigma).astype(np.float32)
        elif op == "gain":
            gain = np.random.uniform(0.71, 1.41)
            y = y * gain
    max_val = np.abs(y).max()
    if max_val > 1e-6:
        y = y / max_val
    return y.astype(np.float32)
```

### Overfitting için bekleyen değişiklikler (AMBIENT düzeltmesinden sonra uygula)

```python
# Hiper-parametre değişiklikleri:
LR         = 3e-4    # 1e-3'ten düşürüldü
BATCH_SIZE = 64      # 32'den artırıldı — val loss gürültüsünü azaltır

# AirportCNN.classifier — ekstra dropout:
self.classifier = nn.Sequential(
    nn.Flatten(),
    nn.Dropout(0.3),           # ← YENİ — flatten'dan hemen sonra
    nn.Linear(128 * 4 * 4, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(256, n_classes),
)
```

---

## env_audio_processor.py — Yapılacak Değişiklik

`environment_class_mappings.csv` incelendikten sonra buraya filtre eklenecek.
Mevcut yapı (değiştirilecek kısım):

```python
# ŞU AN: tüm env_audio segmentleri AMBIENT olarak alınıyor
# OLACAK: uçak sesi içeren segmentler (aircraft_flag veya benzeri sütun)
#          env_clips'e dahil edilmeyecek

# Değiştirilecek satır env_audio_processor.py'de:
# → environment_class_mappings.csv yüklenir
# → uçak segmentleri filtrelenir
# → kalan segmentler kliplenir
```

**Bu dosyanın içeriği görülmeden değişiklik yapılmamalı.**

---

## Sonraki Adımlar (Sıraya Göre)

- [ ] **ÖNCE:** `environment_class_mappings.csv` içeriğini incele → sütunları anlamlandır
- [ ] `env_audio_processor.py` güncelle → uçak segmentleri filtrele
- [ ] `cache/` sil → `dataset_builder_v3.py` yeniden çalıştır
- [ ] `train_cnn.py` yeniden çalıştır (overfitting değişiklikleriyle birlikte)
- [ ] `compare_models.py` çalıştır → temiz karşılaştırma
- [ ] Ensemble (SVM + CNN) — CNN stabil olduktan sonra

---

## Kurulum & Çalıştırma

```bash
cd C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise
venv\Scripts\activate

# Sıfırdan başlamak gerekirse:
del cache\features_v3.pkl
del cache\manifest_v3.csv
python env_audio_processor.py    # AMBIENT klipler
python dataset_builder_v3.py     # özellik matrisi
python train_model_v3.py         # SVM (~2 dk)
python train_cnn.py              # CNN

# GPU kurulumu (GTX 1650, CUDA 13.1 driver):
pip uninstall torch torchaudio -y
pip cache purge
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
# → import torch; print(torch.cuda.is_available()) → True beklenir
```

---

## Yeni Sohbet Notu

Değişiklik yapmadan önce asistan şu dosyaları isteyecek (hangisi değişiyorsa):
- `environment_class_mappings.csv` — AMBIENT filtresi için **şart**
- `env_audio_processor.py` — filtre eklenecek
- `train_cnn.py` — overfitting düzeltmesi için
- `noise_detector.py` — sınıf sayısı veya prior değişirse
