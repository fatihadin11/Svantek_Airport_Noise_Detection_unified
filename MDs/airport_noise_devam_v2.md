# Havalimanı Gürültü Tespit Sistemi — Devam Promptu v2

> Yeni sohbette projeye eksiksiz devam için hazırlanmıştır.
> v1'den fark: Model eğitimi tamamlandı, sıradaki adım entegrasyon.

---

## Projenin Amacı

Havalimanı ortamında gerçek zamanlı çevresel gürültü tespit ve sınıflandırma sistemi.
4 sınıf: **AIRCRAFT / SPEECH / TRAFFIC / WIND**

---

## Klasör Yapısı

```
C:\Users\Fatih\Desktop\TUBITAK\
├── Airport_Noise\                  ← Ana proje
│   ├── noise_detector.py           ← Mevcut sistem (kural tabanlı sınıflandırıcı)
│   ├── ml_classifier.py            ← Eski ML taslağı (artık kullanılmıyor)
│   ├── train_model.py              ← Eğitim scripti (TAMAMLANDI)
│   ├── main.py                     ← CLI çalıştırıcı
│   ├── models\
│   │   ├── best_model.pkl          ← Eğitilmiş SVM modeli ✅
│   │   ├── scaler.pkl              ← StandardScaler ✅
│   │   ├── label_encoder.pkl       ← LabelEncoder ✅
│   │   └── training_meta.pkl       ← Metadata (sr, duration, vb.) ✅
│   ├── cache\
│   │   └── features.pkl            ← Özellik önbelleği (2535 örnek)
│   └── outputs\
│       └── training\               ← Confusion matrix, karşılaştırma grafikleri
├── Dataset_Airplane\               ← AeroSonicDB
│   └── audio\audio\                ← WAV dosyaları (625 örnek → 1895 chunk)
└── Dataset_ESC50\                  ← ESC-50
    ├── audio\                      ← WAV dosyaları
    └── esc50.csv
```

---

## Şu Anki Durum

```
✅ noise_detector.py     — 7 modüler sınıf (Loader, Analyzer, Extractor, Filter, Classifier, Visualizer, System)
✅ Kural tabanlı sistem  — çalışıyor ama zayıf (trafik→uçak hatası vardı)
✅ Dataset hazırlama     — ESC-50 (640) + AeroSonicDB (1895) = 2535 örnek
✅ Model eğitimi         — SVM_RBF kazandı, F1 Macro: %88.44
✅ Model kaydı           — best_model.pkl, scaler.pkl, label_encoder.pkl hazır
⏳ SIRADAKI ADIM         — Eğitilmiş modeli noise_detector.py'e entegre et
⏳ Gerçek zamanlı sistem — sounddevice (ileride)
```

---

## Model Eğitimi Sonuçları

| Sınıf    | Precision | Recall | F1   | Destek |
|----------|-----------|--------|------|--------|
| AIRCRAFT | 0.98      | 0.99   | 0.99 | 395    |
| SPEECH   | 0.95      | 1.00   | 0.98 | 40     |
| TRAFFIC  | 0.92      | 0.69   | 0.79 | 32     |
| WIND     | 0.78      | 0.80   | 0.79 | 40     |

- **Test Accuracy:** %95.86 | **F1 Macro:** %88.44 | **Kazanan:** SVM_RBF (C=5, gamma=scale)
- **SMOTE** uygulandı (2028 → 6320 eğitim örneği)
- Zayıf nokta: TRAFFIC (%69 recall) — broadband ses, uçak/rüzgarla karışıyor

---

## Özellik Çıkarım Fonksiyonu (264 boyut)

> **KRİTİK:** Bu fonksiyon train_model.py ve noise_detector.py'de AYNI olmalı.
> Değiştirilirse model yeniden eğitilmeli.

```python
import librosa
import numpy as np

SR = 22050
DURATION = 5.0
N_MFCC = 13

def load_audio_fixed(path, sr=SR, duration=DURATION):
    """Ses yükle + sabit uzunluğa getir (pad veya center-trim)."""
    y, _ = librosa.load(path, sr=sr, mono=True, duration=duration + 0.5)
    target = int(sr * duration)
    if len(y) >= target:
        start = (len(y) - target) // 2
        y = y[start:start + target]
    else:
        y = np.pad(y, (0, target - len(y)))
    return y.astype(np.float32)

def extract_features(y, sr=SR):
    """264 boyutlu özellik vektörü."""
    mfcc    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    d_mfcc  = librosa.feature.delta(mfcc)
    d2_mfcc = librosa.feature.delta(mfcc, order=2)
    sc      = librosa.feature.spectral_centroid(y=y, sr=sr)
    sb      = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    sr_     = librosa.feature.spectral_rolloff(y=y, sr=sr)
    sf      = librosa.feature.spectral_flatness(y=y)
    zcr     = librosa.feature.zero_crossing_rate(y)
    rms     = librosa.feature.rms(y=y)
    chroma  = librosa.feature.chroma_stft(y=y, sr=sr)

    def ms(x): return np.array([x.mean(), x.std()])

    return np.hstack([
        mfcc.mean(1), mfcc.std(1),
        d_mfcc.mean(1), d_mfcc.std(1),
        d2_mfcc.mean(1), d2_mfcc.std(1),
        ms(sc), ms(sb), ms(sr_), ms(sf), ms(zcr), ms(rms),
        chroma.mean(1), chroma.std(1),
    ]).astype(np.float32)
```

---

## Entegrasyon Sınıfı (train_model.py içinde mevcut)

```python
import joblib

class TrainedClassifier:
    """Eğitilen modeli yükleyip ses sınıflandır."""

    def __init__(self, models_dir=r"C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise\models"):
        self.model  = joblib.load(os.path.join(models_dir, "best_model.pkl"))
        self.scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
        self.le     = joblib.load(os.path.join(models_dir, "label_encoder.pkl"))
        self.meta   = joblib.load(os.path.join(models_dir, "training_meta.pkl"))

    def predict_file(self, wav_path):
        y = load_audio_fixed(wav_path, sr=self.meta["sr"], duration=self.meta["duration"])
        return self.predict_array(y) if y is not None else ("UNKNOWN", 0.0)

    def predict_array(self, y):
        feat = extract_features(y, sr=self.meta["sr"]).reshape(1, -1)
        # SVM için scale et, RF/GBM için ham kullan
        X_in = self.scaler.transform(feat) if self.meta["model_name"] == "SVM_RBF" else feat
        pred  = self.model.predict(X_in)[0]
        label = self.le.inverse_transform([pred])[0]
        conf  = float(self.model.predict_proba(X_in)[0].max()) if hasattr(self.model, "predict_proba") else 1.0
        return label, conf
```

---

## SIRADAKI ADIM: noise_detector.py Entegrasyonu

### Yapılacak Değişiklikler

**1. AirportNoiseSystem.__init__() — ML modeli yükle**

```python
# noise_detector.py → AirportNoiseSystem sınıfı içine ekle
from train_model import TrainedClassifier, extract_features, load_audio_fixed

class AirportNoiseSystem:
    def __init__(self, target_sr=22050, output_dir="outputs"):
        self.target_sr  = target_sr
        self.output_dir = output_dir
        # Diğer mevcut init kodları...
        
        # ML modeli yükle (models/ klasörü varsa)
        models_dir = os.path.join(os.path.dirname(__file__), "models")
        try:
            self.ml_clf = TrainedClassifier(models_dir=models_dir)
            print("[ML] Model yüklendi ✅")
        except FileNotFoundError:
            self.ml_clf = None
            print("[ML] Model bulunamadı — kural tabanlı kullanılacak")
```

**2. AirportNoiseSystem.run() — Sınıflandırma kısmını güncelle**

```python
# run() içindeki mevcut kural tabanlı classify() çağrısını şununla değiştir:
def _classify_ml(self, audio_path):
    """Tüm dosyayı 5 saniyelik pencerelere böl, her pencereyi sınıflandır."""
    y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
    window  = int(sr * 5.0)    # 5 saniyelik pencere
    hop     = int(sr * 2.5)    # %50 overlap
    labels  = []

    for start in range(0, len(y) - window + 1, hop):
        chunk = y[start:start + window].astype(np.float32)
        label, conf = self.ml_clf.predict_array(chunk)
        labels.append(label)

    if not labels:
        label, conf = self.ml_clf.predict_file(audio_path)
        labels = [label]

    # Dağılım hesapla
    from collections import Counter
    dist = Counter(labels)
    total = len(labels)
    return {lbl: (cnt / total * 100) for lbl, cnt in dist.items()}
```

**3. run() içinde koşullu kullanım**

```python
# run() metodunda:
if self.ml_clf is not None:
    classification = self._classify_ml(audio_path)
else:
    # Eski kural tabanlı
    rule_labels, _, _ = self.classifier.classify(features)
    from collections import Counter
    dist = Counter(rule_labels)
    total = len(rule_labels)
    classification = {lbl: cnt/total*100 for lbl, cnt in dist.items()}
```

---

## Eğitim Parametreleri (tekrar eğitim gerekirse)

```python
# train_model.py içindeki ayarlar
AIRPLANE_PATH = r"C:\Users\Fatih\Desktop\TUBITAK\Dataset_Airplane"
ESC50_PATH    = r"C:\Users\Fatih\Desktop\TUBITAK\Dataset_ESC50"
PROJECT_ROOT  = r"C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise"
SR            = 22050
DURATION      = 5.0
N_MFCC        = 13
CV_FOLDS      = 5
TEST_SIZE     = 0.15

# ESC-50 kategori eşlemesi
ESC50_LABEL_MAP = {
    "airplane": "AIRCRAFT", "helicopter": "AIRCRAFT",
    "car_horn": "TRAFFIC",  "engine": "TRAFFIC",
    "train": "TRAFFIC",     "siren": "TRAFFIC",
    "wind": "WIND",         "rain": "WIND",
    "thunderstorm": "WIND", "sea_waves": "WIND",
    "clapping": "SPEECH",   "laughing": "SPEECH",
    "crying_baby": "SPEECH","crowd": "SPEECH",
    "footsteps": "SPEECH",
}
```

---

## Bilinen Sorunlar ve Notlar

| Sorun | Durum | Çözüm |
|-------|-------|-------|
| TRAFFIC recall %69 | ⚠️ Açık | UrbanSound8K eklenebilir (sonraki iterasyon) |
| WIND recall %80 | ⚠️ Açık | Daha fazla çeşitli WIND örneği |
| Band-pass filtre etkisiz | ⚠️ Açık | `lowcut=100, highcut=2000` yapılmalı |
| Model her seferinde sıfırdan eğitiliyordu | ✅ Çözüldü | joblib ile kaydedildi |
| Kural tabanlı trafik→uçak hatası | ✅ Çözüldü | ML model devreye giriyor |

---

## Kurulum ve Çalıştırma

```bash
cd C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise
venv\Scripts\activate

# Normal çalıştırma (entegrasyon sonrası)
python main.py --audio "sounds\Flying Plane Sound Effect.wav"
python main.py --demo

# Modeli yeniden eğitmek gerekirse
python train_model.py              # Tüm pipeline
python train_model.py --phase 1   # Sadece özellik çıkar (önbellekle)

# Gerekli kütüphaneler
pip install numpy scipy matplotlib librosa soundfile scikit-learn pandas seaborn tqdm joblib imbalanced-learn
```

---

## Sonraki Sohbet Öncelik Sırası

1. **[ŞİMDİ]** `noise_detector.py` entegrasyonu — `AirportNoiseSystem` içine ML modeli ekle
2. **[ŞİMDİ]** `main.py` ile uçtan uca test
3. **[SONRA — gerekirse]** TRAFFIC/WIND için UrbanSound8K ekleme
4. **[SONRA]** Gerçek zamanlı mikrofon sistemi (`sounddevice`)
