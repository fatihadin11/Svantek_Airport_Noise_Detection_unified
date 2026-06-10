# Havalimanı Gürültü Tespit Sistemi — Devam Promptu v3

> ⚠️ BAŞKA SOHBETTE KULLANILACAKSA: Önce hangi dosyaları değiştireceğini söyle,
> ardından kullanıcıdan ilgili dosyanın güncel içeriğini iste. Dosyaları görmeden
> kod yazma — özellikle noise_detector.py ve train_model_v3.py kritik.

---

## Proje Özeti

Havalimanı ortamında gerçek zamanlı çevresel gürültü sınıflandırma sistemi.
TÜBİTAK projesi. Klasik ML (SVM) pipeline tamamlandı, sırada CNN implementasyonu var.

**5 Sınıf:** AIRCRAFT / AMBIENT / SPEECH / TRAFFIC / WIND

---

## Klasör Yapısı

```
C:\Users\Fatih\Desktop\TUBITAK\
├── Airport_Noise\                      ← Ana proje klasörü
│   ├── noise_detector.py               ← Ana sistem (ML entegreli) ✅
│   ├── main.py                         ← CLI çalıştırıcı ✅
│   ├── env_audio_processor.py          ← env_audio → AMBIENT klip üretici ✅
│   ├── dataset_builder_v3.py           ← ESC50 + AeroSonic + AMBIENT → features ✅
│   ├── train_model_v3.py               ← SVM eğitim scripti (temizlenecek) ✅
│   ├── train_cnn.py                    ← ⏳ YAZILACAK
│   ├── models\
│   │   ├── best_model.pkl              ← SVM (Pipeline: scaler + clf) ✅
│   │   ├── label_encoder.pkl           ✅
│   │   └── training_meta.pkl           ✅
│   ├── cache\
│   │   ├── features_v3.pkl             ← 264-boyut özellik matrisi ✅
│   │   └── manifest_v3.csv             ✅
│   └── outputs\
│       └── training_v3\
│           └── confusion_matrix_v3.png ✅
│
├── Dataset_Airplane\                   ← AeroSonicDB
│   ├── audio\audio\                    ← AIRCRAFT WAV (1895 dosya)
│   ├── env_audio\env_audio\            ← 6 sürekli kayıt WAV
│   ├── env_clips\                      ← 979 AMBIENT klip (işlenmiş) ✅
│   ├── env_audio_manifest.csv          ✅
│   ├── environment_class_mappings.csv  ← Başlıksız CSV (sorun var, not'a bak)
│   └── sample_meta.csv
│
└── Dataset_ESC50\
    ├── audio\audio\                    ← WAV dosyaları
    └── esc50.csv
```

---

## Mevcut Durum — SVM Pipeline (TAMAMLANDI)

### Eğitim Sonuçları (v3 — 5 sınıf)

| Sınıf    | Precision | Recall | F1   | Destek |
|----------|-----------|--------|------|--------|
| AIRCRAFT | 0.957     | 0.973  | 0.965| 297    |
| AMBIENT  | 0.966     | 0.973  | 0.969| 147    |
| SPEECH   | 0.923     | 1.000  | 0.960| 24     |
| TRAFFIC  | 0.895     | 0.708  | 0.791| 24     |
| WIND     | 0.857     | 0.750  | 0.800| 24     |

- **Test Accuracy:** %95.2 | **F1 Macro:** %89.7
- **Model:** SVM_RBF (C=50, gamma=scale) — sklearn Pipeline (scaler içinde)
- **SMOTE:** 2918 → 8390 eğitim örneği
- **Naive baseline:** %57.5 (sağlıklı, AIRCRAFT dominasyonu kırıldı)
- **Prior düzeltmesi:** noise_detector.py içinde PRIOR_WEIGHTS["AIRCRAFT"] = 0.10

### Veri Kaynakları

| Kaynak | Sınıf | Örnek |
|--------|-------|-------|
| AeroSonicDB audio/ | AIRCRAFT | 1895 (625 dosya → chunk) |
| ESC-50 | AIRCRAFT,SPEECH,TRAFFIC,WIND | 560 |
| AeroSonicDB env_audio/ | AMBIENT | 979 klip |
| **Toplam** | | **3434** |

### Özellik Vektörü (264 boyut) — KRİTİK

```python
# dataset_builder_v3.py ve noise_detector.py'de BİREBİR AYNI olmalı
N_MFCC = 40;  N_FFT = 2048;  HOP_FFT = 512;  SR = 22050

def extract_features(y, sr=SR):
    mfcc    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_FFT)
    d_mfcc  = librosa.feature.delta(mfcc)
    d2_mfcc = librosa.feature.delta(mfcc, order=2)
    chroma  = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_FFT)
    sc  = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_FFT)
    sb  = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_FFT)
    sr_ = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_FFT)
    sf  = librosa.feature.spectral_flatness(y=y, n_fft=N_FFT, hop_length=HOP_FFT)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=HOP_FFT)
    rms = librosa.feature.rms(y=y, hop_length=HOP_FFT)
    features = []
    for m in [mfcc, d_mfcc, d2_mfcc]:
        features.extend([m.mean(axis=1), m.std(axis=1)])
    features.append(chroma.mean(axis=1))
    for feat in [sc, sb, sr_, sf, zcr, rms]:
        features.extend([feat.mean(axis=1), feat.std(axis=1)])
    return np.concatenate(features).astype(np.float32)  # → (264,)
```

### Prior Düzeltmesi (noise_detector.py)

```python
# AirportNoiseSystem sınıfı içinde — AIRCRAFT dominasyonunu kırar
PRIOR_WEIGHTS = {
    "AIRCRAFT": 0.10,   # ← Ana ayar. Düşük = az AIRCRAFT tahmini
    "SPEECH":   2.0,
    "TRAFFIC":  2.0,
    "WIND":     2.0,
    "AMBIENT":  2.0,
}

def _apply_prior(self, probs):
    classes  = list(self.ml_le.classes_)
    adjusted = probs.copy()
    for i, cls in enumerate(classes):
        adjusted[i] *= self.PRIOR_WEIGHTS.get(cls, 1.0)
    return adjusted / (adjusted.sum() + 1e-9)
```

---

## train_model_v3.py — Yapılacak Temizlik

SVM her seferinde kazanıyor, diğer modelleri test etmeye gerek yok.
Aşağıdaki değişiklik yeterli:

```python
# KALDIRILACAK — compare_models() fonksiyonu içinden:
# RandomForest ve GradientBoosting döngüsü silinecek

# YERİNE — direkt SVM eğit:
def train_svm(X_tr, y_enc, le):
    labels = list(le.classes_)
    cw_enc = {i: MANUAL_CLASS_WEIGHTS.get(lbl, 1.0) for i, lbl in enumerate(labels)}
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=50, gamma="scale",
                    probability=True, class_weight=cw_enc))
    ])
    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1s = cross_val_score(pipe, X_tr, y_enc, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"  SVM CV F1: {f1s.mean():.4f} ± {f1s.std():.4f}")
    pipe.fit(X_tr, y_enc)
    return pipe

# GridSearch da kaldırılabilir — C=50, gamma=scale zaten optimize edildi
```

---

## Bilinen Sorunlar

| Sorun | Durum | Not |
|-------|-------|-----|
| env_audio CSV başlıksız | ⚠️ Açık | Uçak seg. filtrelenemedi, 979 klipte ~50-100 uçak sesi olabilir |
| TRAFFIC/WIND recall düşük | ⚠️ Açık | 160'ar örnek az, CNN ile iyileşmesi bekleniyor |
| WIND↔AMBIENT örtüşme | ⚠️ Açık | env_audio rüzgar içeriyor, sınır bulanık |

---

## Sonraki Adımlar — Plan

### AŞAMA 1: train_model_v3.py Temizliği (Kısa, Hemen)

- RF ve GBM karşılaştırmasını kaldır
- GridSearch kaldır (C=50 sabit)
- Sadece SVM eğit, kaydet, bitir

### AŞAMA 2: CNN + Mel Spectrogram (Ana Hedef)

**Neden CNN?**
SVM 264 boyutlu el yapımı özelliklerle çalışıyor — bu bilgi kaybı demek.
CNN ham Mel spektrogramdan (128×216 piksel = 27.648 boyut) kendi özelliklerini öğreniyor.
Avantajları:
- Frekans-zaman örüntülerini doğrudan yakalar (uçak motorunun harmonik yapısı gibi)
- SpecAugment ile kolayca augment edilir
- TÜBİTAK raporunda "klasik ML vs derin öğrenme" karşılaştırması güçlü akademik içerik

**Mimari Plan (train_cnn.py):**

```python
# Mel Spectrogram parametreleri
N_MELS    = 128
N_FFT     = 2048
HOP_FFT   = 512
SR        = 22050
DURATION  = 5.0
# Çıkan boyut: (1, 128, 216) — (kanal, mel_bin, zaman_frame)

# CNN Mimarisi
class AirportCNN(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            # Blok 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),          # → (32, 64, 108)
            nn.Dropout2d(0.1),
            # Blok 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),          # → (64, 32, 54)
            nn.Dropout2d(0.1),
            # Blok 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) # → (128, 4, 4) — sabit çıktı
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
```

**SpecAugment (CNN ile birlikte):**

```python
import torchaudio.transforms as T

class SpecAugment(nn.Module):
    """
    Eğitim sırasında Mel spektrogramı maskele.
    time_mask:  zaman ekseninde rastgele T frame sıfırla
    freq_mask:  frekans ekseninde rastgele F bin sıfırla
    """
    def __init__(self, time_mask=30, freq_mask=15, n_time=2, n_freq=2):
        super().__init__()
        self.time_masks = nn.ModuleList([
            T.TimeMasking(time_mask_param=time_mask) for _ in range(n_time)
        ])
        self.freq_masks = nn.ModuleList([
            T.FrequencyMasking(freq_mask_param=freq_mask) for _ in range(n_freq)
        ])

    def forward(self, x):
        for m in self.time_masks: x = m(x)
        for m in self.freq_masks: x = m(x)
        return x

# Dataset sınıfında eğitim modunda uygula:
if self.augment:
    spec = self.spec_augment(spec)
```

**Dataset sınıfı:**

```python
class MelSpectrogramDataset(Dataset):
    def __init__(self, records, sr=SR, n_mels=N_MELS,
                 n_fft=N_FFT, hop_length=HOP_FFT,
                 duration=DURATION, augment=False):
        self.records    = records
        self.augment    = augment
        self.mel_tf     = T.MelSpectrogram(sr, n_fft=n_fft,
                                            hop_length=hop_length, n_mels=n_mels)
        self.db_tf      = T.AmplitudeToDB()
        self.spec_aug   = SpecAugment() if augment else None
        self.target_len = int(sr * duration)

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        y, _  = librosa.load(rec["path"], sr=SR, mono=True)
        # Sabit uzunluk
        if len(y) >= self.target_len:
            start = (len(y) - self.target_len) // 2
            y     = y[start:start + self.target_len]
        else:
            y = np.pad(y, (0, self.target_len - len(y)))
        y_tensor = torch.FloatTensor(y).unsqueeze(0)          # (1, samples)
        spec     = self.mel_tf(y_tensor)                       # (1, n_mels, frames)
        spec     = self.db_tf(spec)                            # dB normalize
        if self.augment and self.spec_aug:
            spec = self.spec_aug(spec)
        return spec, rec["label_enc"]
```

**Eğitim döngüsü (özet):**

```python
# Class weights — manuel ağırlıklar
weights = torch.FloatTensor([
    MANUAL_CLASS_WEIGHTS[le.classes_[i]] for i in range(len(le.classes_))
]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# Eğitim: 50 epoch, early stopping (patience=10)
# Kayıt: models/best_cnn.pt
```

**Gereksinimler:**

```bash
pip install torch torchaudio  # CPU: ~200MB, GPU: ~2GB
# GPU yoksa CPU'da ~5-10 dakika/epoch (50 epoch = 4-8 saat)
# Colab veya GPU önerilir
```

### AŞAMA 3: Karşılaştırma Modülü

SVM ve CNN sonuçlarını yan yana karşılaştıran tablo + grafik.
`compare_models.py` — ayrı script, her ikisini de yükler, aynı test seti üzerinde değerlendirir.

### AŞAMA 4: Pretrained (İleride — Seninle Tartışılacak)

**PANNs (Pretrained Audio Neural Networks):**
- 527 sınıf AudioSet üzerinde eğitilmiş
- CNN14 mimarisi embedding çıkarır (2048 boyut)
- Üstüne küçük MLP ekle → fine-tune
- Çevresel sesler için en güçlü pretrained seçenek

**YAMNet:**
- Google'ın AudioSet modeli
- TensorFlow Hub'dan kolay yükleme
- Daha hafif ama PANNs'tan zayıf

**VGGish:**
- YouTube-8M üzerinde eğitilmiş
- 128 boyut embedding
- Eski ama yaygın kullanılan baseline

Hangisini seçeceğimizi CNN sonuçlarına bakarak karar vereceğiz.

---

## Kurulum & Çalıştırma

```bash
cd C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise
venv\Scripts\activate

# Normal kullanım
python main.py --audio "sounds\ses.wav"
python main.py --demo

# Model yeniden eğitim (sadece SVM artık)
python dataset_builder_v3.py   # özellikler zaten cachede, atlanır
python train_model_v3.py       # ~2 dakika

# CNN eğitim (yazılacak)
python train_cnn.py            # GPU önerilir

# Kütüphaneler
pip install numpy librosa scikit-learn pandas matplotlib seaborn
            tqdm joblib imbalanced-learn soundfile
pip install torch torchaudio   # CNN için
```

---

## Sohbet Geçişi Notu

Yeni sohbette şu adımları izle:
1. Bu MD dosyasını yapıştır
2. Hangi değişikliği yapacağını söyle
3. Asistan senden ilgili dosyaları isteyecek — o zaman dosya içeriklerini paylaş
4. **noise_detector.py, train_model_v3.py, dataset_builder_v3.py** en kritik dosyalar

Bir sonraki sohbette yapılacaklar (sıraya göre):
- [ ] train_model_v3.py temizliği (RF/GBM/GridSearch kaldır)
- [ ] train_cnn.py yazımı
- [ ] noise_detector.py CNN entegrasyonu (SVM ile aynı _classify_ml mantığı)
- [ ] compare_models.py — yan yana karşılaştırma
