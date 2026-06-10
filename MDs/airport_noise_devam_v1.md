# Havalimanı Çevresel Gürültü Tespit Sistemi — Devam Promptu v1

> Bu dosya yeni bir sohbette projeye eksiksiz devam edebilmek için hazırlanmıştır.
> Tüm kodlar, kararlar, sorunlar ve sonraki adımlar burada belgelenmiştir.

---

## Projenin Amacı

Havalimanı ortamlarında kullanılmak üzere bir çevresel gürültü tespit sistemi geliştirmek.
Nihai hedef: Gerçek zamanlı havalimanı gürültü izleme sistemine dönüştürülebilecek bir prototip.

---

## Klasör Yapısı

```
airport_noise/
├── noise_detector.py      ← Ana modüller (Loader, Analyzer, Filter, Classifier, Visualizer)
├── ml_classifier.py       ← ML tabanlı sınıflandırıcılar (GNB, KNN, CNN taslak)
├── main.py                ← Çalıştırıcı & CLI
├── venv/                  ← Virtual environment (git'e ekleme)
├── sounds/                ← Test ses dosyaları
│   └── Flying Plane Sound Effect.wav
└── outputs/               ← Otomatik oluşturulur, grafikler buraya kaydedilir
    ├── airport_noise_dashboard.png
    ├── airport_noise_classification_pie.png
    ├── airport_noise_1_waveform.png
    ├── airport_noise_2_db_over_time.png
    ├── airport_noise_3_spectrogram.png
    ├── airport_noise_4_mel_spectrogram.png
    ├── airport_noise_5_mfcc.png
    ├── airport_noise_6_features.png
    └── airport_noise_7_filter_comparison.png
```

---

## Kurulum

```bash
# 1. Klasör oluştur ve gir
cd C:\Users\Fatih\Desktop\TUBITAK\airport_noise

# 2. Virtual environment oluştur (bir kez)
python -m venv venv

# 3. Aktive et
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

# 4. Kütüphaneleri kur
pip install numpy scipy matplotlib librosa soundfile

# 5. Çalıştır
python main.py --audio sounds/Flying Plane Sound Effect.wav
python main.py --demo    # sentetik ses ile test
```

> ÖNEMLİ: Klasör adında Türkçe karakter KULLANMA (TÜBİTAK → TUBITAK).
> VS Code'da Ctrl+F5 veya Terminal > New Terminal → python main.py

---

## Bu Sohbette Yapılanlar

### ✅ Adım 1 — Temel Sistem Kurulumu
`noise_detector.py` içinde 7 modüler sınıf yazıldı:

| Sınıf | Görev |
|---|---|
| `AudioLoader` | WAV/MP3 yükleme, mono dönüşüm, yeniden örnekleme |
| `AudioAnalyzer` | Waveform, STFT spektrogram, Mel spektrogram, dBFS/zaman |
| `FeatureExtractor` | MFCC (13), Spectral Centroid, Bandwidth, Rolloff, ZCR, RMS |
| `NoiseFilter` | Band-pass (Butterworth), Spectral Gating, Wiener filtresi |
| `NoiseClassifier` | Kural tabanlı: AIRCRAFT / WIND / SPEECH / TRAFFIC / UNKNOWN |
| `Visualizer` | 9 panelli dashboard + ayrı ayrı grafikler + pasta grafiği |
| `AirportNoiseSystem` | Tüm pipeline'ı yöneten orkestratör |

### ✅ Adım 2 — MP3 / Librosa Desteği
`AudioLoader.load()` metodu `librosa` ile güncellendi → hem WAV hem MP3 çalışıyor.

### ✅ Adım 3 — Grafikleri Ayrı Ayrı Kaydetme
`Visualizer` sınıfına 7 ayrı grafik metodu eklendi:
- `plot_waveform()` → `_1_waveform.png`
- `plot_db_over_time()` → `_2_db_over_time.png`
- `plot_spectrogram()` → `_3_spectrogram.png`
- `plot_mel_spectrogram()` → `_4_mel_spectrogram.png`
- `plot_mfcc()` → `_5_mfcc.png`
- `plot_features()` → `_6_features.png` (SC, ZCR, RMS)
- `plot_filter_comparison()` → `_7_filter_comparison.png`

### ✅ Adım 4 — Gerçek Ses Dosyası Testi
YouTube'dan indirilen uçak sesi ile test edildi:
- Uçak sesi → %73 AIRCRAFT (kısmen doğru)
- Trafik sesi → %100 AIRCRAFT ❌ (yanlış — sınıflandırma sorunu tespit edildi)

---

## Mevcut Sorunlar ve Kararlar

### Sorun 1 — Sınıflandırma Kural Tabanlı ve Zayıf

Şu an `NoiseClassifier` sabit eşik değerleriyle karar veriyor:
```python
"aircraft_sc_min": 400,   # Spectral centroid 400 Hz üzerindeyse
"aircraft_sc_max": 2000,
"aircraft_rms_min": 0.05  # ve enerji yeterliyse → AIRCRAFT
```
Trafik sesi de broadband ve yüksek enerjili olduğu için AIRCRAFT olarak etiketleniyor.
**Karar:** Gerçek ML modeli gerekiyor.

### Sorun 2 — ML Modeli Kural Tabanlıyı Taklit Ediyor

`ml_classifier.py` içindeki `MLNoiseClassifier.bootstrap_train()` kural tabanlı etiketlerden öğreniyor.
Yani kural ne diyorsa onu öğreniyor — bağımsız bir model değil, gerçekten işe yaramıyor.

### Sorun 3 — Band-pass Grafikte Görsel Etki Yok

Mevcut ayar: `lowcut=80, highcut=8000` Hz — neredeyse tüm spektrum geçiyor, filtre etkisiz.
**Çözüm:** Anlamlı aralıklar kullanılmalı (örn. `100–2000 Hz` uçak motoru için).

### Sorun 4 — Model Kaydedilmiyor

Her çalıştırmada model sıfırdan başlıyor. Gerçek bir model eğitilince `joblib` ile kaydedilmeli.

---

## Sonraki Adımlar (Sırasıyla)

### 🔲 Adım 1 — Veri Seti Seçimi ve İndirme (EN ÖNCELİKLİ)

**Karar:** UrbanSound8K yerine **ESC-50** kullanılacak.

Gerekçe:
- UrbanSound8K'da "airplane" kategorisi yok, havalimanında olmayacak sesler var (gun_shot, dog_bark vb.)
- ESC-50'de "airplane" kategorisi mevcut, 50 kategori, 2000 klip, akademik kalite, sadece 600 MB

**İndirme:** https://github.com/karolpiczak/ESC-50
```
ESC-50/
├── audio/          ← 2000 WAV dosyası (5'er saniyelik)
└── meta/
    └── esc50.csv   ← etiketler burada
```

ESC-50 kategorilerinden kullanacaklarımız:

| ESC-50 Kategori | Bizim Etiket |
|---|---|
| airplane | AIRCRAFT |
| helicopter | AIRCRAFT |
| car_horn | TRAFFIC |
| engine | TRAFFIC |
| wind | WIND |
| rain | WIND (arka plan) |
| crowd | SPEECH |
| clapping | SPEECH |

Eksik olan uçak sesleri için:
- YouTube'dan indirilen mevcut kayıt kullanılacak
- **Audacity** ile uzun kaydı 4–5 saniyelik parçalara böl
- `kendi_veri/airplane/` klasörüne koy

### 🔲 Adım 2 — Veri Hazırlama ve Özellik Çıkarımı

```python
# train_model.py — yazılacak
import pandas as pd
import librosa
import numpy as np

ESC50_PATH = r"C:\...\ESC-50"
LABEL_MAP  = {
    "airplane":   "AIRCRAFT",
    "helicopter": "AIRCRAFT",
    "car_horn":   "TRAFFIC",
    "engine":     "TRAFFIC",
    "wind":       "WIND",
    "crowd":      "SPEECH",
}

def extract_features(wav_path, sr=22050):
    """Her ses dosyasından 1 özellik vektörü çıkar."""
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    sc    = librosa.feature.spectral_centroid(y=y, sr=sr)
    sb    = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zcr   = librosa.feature.zero_crossing_rate(y)
    rms   = librosa.feature.rms(y=y)
    # Her özelliğin ortalaması → sabit boyutlu vektör
    vec = np.hstack([
        mfcc.mean(axis=1),        # 13 boyut
        mfcc.std(axis=1),         # 13 boyut
        sc.mean(), sc.std(),       # 2 boyut
        sb.mean(), sb.std(),       # 2 boyut
        zcr.mean(), zcr.std(),     # 2 boyut
        rms.mean(), rms.std(),     # 2 boyut
    ])  # toplam: 34 boyut
    return vec
```

### 🔲 Adım 3 — Model Eğitimi (Random Forest + SVM)

CNN'den önce bu iki model daha hızlı sonuç verir, az veriyle iyi çalışır:

```python
# train_model.py devamı
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import joblib

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
scores = cross_val_score(rf, X, y, cv=5)
print(f"RF CV doğruluğu: {scores.mean():.3f} ± {scores.std():.3f}")

# SVM
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
scores = cross_val_score(svm, X_scaled, y, cv=5)
print(f"SVM CV doğruluğu: {scores.mean():.3f} ± {scores.std():.3f}")

# En iyi modeli kaydet
joblib.dump(rf, "models/random_forest.pkl")
joblib.dump(scaler, "models/scaler.pkl")
```

### 🔲 Adım 4 — Modeli Sisteme Entegre Et

`AirportNoiseSystem.run()` içinde kural tabanlı sınıflandırıcı yerine eğitilmiş model kullanılacak:

```python
# noise_detector.py → AirportNoiseSystem.__init__ güncellenmeli
import joblib
self.ml_model  = joblib.load("models/random_forest.pkl")
self.ml_scaler = joblib.load("models/scaler.pkl")
```

### 🔲 Adım 5 — Band-pass Parametrelerini Düzelt

```python
# noise_detector.py → AirportNoiseSystem.run() içinde
filtered_bp = self.filter.bandpass_filter(samples, lowcut=100, highcut=2000)
# Uçak motoru: 100–2000 Hz
# Bu aralık dışındaki rüzgar (düşük) ve yüksek frekanslı sesler kesilir
```

### 🔲 Adım 6 — Özellik Mühendisliği İyileştirmeleri

Şu an eksik olan güçlü özellikler:

```python
# FeatureExtractor'a eklenecek

# 1. Spectral Flatness — uçak sesi düz (broadband), konuşma sivri
def spectral_flatness(magnitude):
    geom_mean = np.exp(np.mean(np.log(magnitude + 1e-10), axis=0))
    arith_mean = np.mean(magnitude, axis=0)
    return geom_mean / (arith_mean + 1e-10)

# 2. Frekans Bandı Enerji Oranları — uçak 100–800 Hz'de baskın
def band_energy_ratio(samples, sr, bands=[(0,200),(200,800),(800,4000),(4000,11025)]):
    freqs, _, Zxx = signal.stft(samples, fs=sr, nperseg=2048)
    power = np.abs(Zxx)**2
    total = power.sum(axis=0) + 1e-10
    ratios = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        ratios.append(power[mask].sum(axis=0) / total)
    return ratios

# 3. Delta MFCC — konuşma hızlı değişir, uçak sesi yavaş
delta_mfcc = np.diff(mfcc, axis=1)
```

### 🔲 Adım 7 — Gerçek Zamanlı Sistem (Son Aşama)

```python
# realtime.py — yazılacak
import sounddevice as sd

class RealtimeAirportNoise:
    def __init__(self, sr=22050, block_size=22050):  # 1 saniyelik bloklar
        self.sr = sr
        self.block_size = block_size
        self.system = AirportNoiseSystem(target_sr=sr)

    def callback(self, indata, frames, time, status):
        samples = indata[:, 0]
        features = self.system.extractor.extract_all(samples)
        labels, _, summary = self.system.classifier.classify(features)
        dominant = max(summary, key=summary.get)
        print(f"[{time.inputBufferAdcTime:.1f}s] {dominant}: {summary}")

    def start(self):
        with sd.InputStream(samplerate=self.sr, channels=1,
                            blocksize=self.block_size,
                            callback=self.callback):
            print("Gerçek zamanlı izleme başladı... (Ctrl+C ile dur)")
            while True:
                sd.sleep(1000)
```

---

## Teknik Kararlar ve Gerekçeler

| Konu | Karar | Gerekçe |
|---|---|---|
| FFT vs STFT | STFT kullanıyoruz | STFT hem frekans hem zaman bilgisi verir, spektrogram için zorunlu |
| UrbanSound8K vs ESC-50 | ESC-50 | "airplane" kategorisi var, domain daha uygun, daha küçük |
| kural tabanlı vs ML | ML'e geçiyoruz | Trafik→uçak hataları sabit eşiklerle çözülemiyor |
| GNB/KNN vs RF/SVM | RF + SVM | Daha güçlü, az veriyle iyi çalışır, CNN'den hızlı |
| CNN ne zaman? | RF/SVM yeterli değilse | Daha fazla veri ve GPU gerektirir |
| librosa | Kuruldu, kullanıyoruz | WAV+MP3 desteği, MFCC hesabı |
| model kaydetme | joblib ile .pkl | Standart, hızlı, sklearn uyumlu |

---

## Kritik API / Davranış Notları

- `sys.path.insert()` ses dosyası bulmak için **kullanılmaz** — Python modül yolu içindir
- Klasör adında Türkçe karakter VS Code'da `İ` harfini düşürüyor → `TUBITAK` kullan
- `[main.py](http://main.py)` görünümü Markdown link formatı — terminalde `python main.py` yaz
- librosa kurulmadan `import librosa` satırı hata verir → `pip install librosa`
- Band-pass `80–8000 Hz` aralığı pratikte hiçbir şey kesmez, grafikte etki görünmez
- `matplotlib.use("Agg")` — headless mod, dosyaya kaydeder, ekranda göstermez

---

## Mevcut Kodlar

### `noise_detector.py`

```python
"""
=============================================================
 HAVALIMANL ÇEVRESEL GÜRÜLTÜ TESPİT SİSTEMİ
 Airport Environmental Noise Detection System
=============================================================
Modüler yapı:
  1. AudioLoader       – WAV/MP3 yükleme & normalize
  2. AudioAnalyzer     – Waveform, spektrogram, dB analizi
  3. FeatureExtractor  – MFCC, spectral centroid, ZCR, RMS
  4. NoiseFilter       – Band-pass, spectral gating
  5. NoiseClassifier   – Kural tabanlı + ML-ready sınıflandırıcı
  6. Visualizer        – Tüm grafikleri çizer
  7. AirportNoiseSystem – Orkestratör (hepsini birleştirir)
"""

import struct
import wave
import numpy as np
import scipy.signal as signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


def _read_wav(path: str):
    """stdlib wave modülü ile WAV oku → (samples_float32, sample_rate)"""
    with wave.open(path, "rb") as wf:
        sr       = wf.getframerate()
        n_ch     = wf.getnchannels()
        sampw    = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw      = wf.readframes(n_frames)
    fmt_map = {1: "b", 2: "h", 4: "i"}
    fmt     = f"<{n_frames * n_ch}{fmt_map[sampw]}"
    samples = np.array(struct.unpack(fmt, raw), dtype=np.float32)
    if n_ch > 1:
        samples = samples.reshape(-1, n_ch).mean(axis=1)
    max_val = float(2 ** (8 * sampw - 1))
    samples /= max_val
    return samples, sr


def _synth_wav(path: str, duration=5.0, sr=22050):
    """Test amaçlı sentetik WAV üretir: uçak + rüzgar + konuşma."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    aircraft  = 0.5 * np.sin(2 * np.pi * 200 * t)
    aircraft += 0.3 * np.sin(2 * np.pi * 400 * t)
    aircraft += 0.2 * np.sin(2 * np.pi * 800 * t)
    aircraft += 0.15 * np.random.randn(len(t))
    wind_noise = np.random.randn(len(t))
    b, a       = signal.butter(2, [20 / (sr / 2), 300 / (sr / 2)], btype="band")
    wind       = 0.3 * signal.lfilter(b, a, wind_noise)
    speech_noise = np.random.randn(len(t))
    b2, a2       = signal.butter(3, [300 / (sr / 2), 3400 / (sr / 2)], btype="band")
    speech_sim   = 0.1 * signal.lfilter(b2, a2, speech_noise)
    speech_sim  *= (0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    composite = aircraft + wind + speech_sim
    composite /= np.max(np.abs(composite) + 1e-9)
    pcm = (composite * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    print(f"[AudioLoader] Sentetik WAV oluşturuldu: {path}  ({duration:.1f}s, {sr} Hz)")
    return path


# ═══════════════════════════════════════════════════════
#  MODÜL 1 – AudioLoader
# ═══════════════════════════════════════════════════════

class AudioLoader:
    """WAV ve MP3 ses dosyalarını yükler (librosa ile)."""

    def __init__(self, target_sr: int = 22050):
        self.target_sr = target_sr

    def load(self, path: str):
        import librosa
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dosya bulunamadı: {path}")
        samples, sr = librosa.load(str(path), sr=self.target_sr, mono=True)
        duration = len(samples) / sr
        print(f"[AudioLoader] '{p.name}' yüklendi | Süre: {duration:.2f}s | SR: {sr} Hz | Örnekler: {len(samples):,}")
        return samples.astype(np.float32), sr

    @staticmethod
    def _resample(samples, orig_sr, target_sr):
        ratio = target_sr / orig_sr
        n_out = int(len(samples) * ratio)
        return signal.resample(samples, n_out).astype(np.float32)


# ═══════════════════════════════════════════════════════
#  MODÜL 2 – AudioAnalyzer
# ═══════════════════════════════════════════════════════

class AudioAnalyzer:
    """Zaman-frekans domeni analizleri."""

    def __init__(self, n_fft=2048, hop_length=512, sr=22050):
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.sr         = sr

    def waveform(self, samples):
        t = np.linspace(0, len(samples) / self.sr, len(samples))
        return t, samples

    def spectrogram(self, samples):
        freqs, times, Zxx = signal.stft(
            samples, fs=self.sr, window="hann",
            nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length)
        S_db = 20 * np.log10(np.abs(Zxx) + 1e-10)
        return freqs, times, S_db

    def mel_spectrogram(self, samples, n_mels=128):
        freqs, times, Zxx = signal.stft(
            samples, fs=self.sr, window="hann",
            nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length)
        power  = np.abs(Zxx) ** 2
        mel_fb = self._mel_filterbank(n_mels, self.n_fft, self.sr)
        mel_S  = mel_fb @ power
        mel_db = 10 * np.log10(mel_S + 1e-10)
        return freqs, times, mel_db

    @staticmethod
    def _mel_filterbank(n_mels, n_fft, sr):
        def hz_to_mel(hz):  return 2595 * np.log10(1 + hz / 700)
        def mel_to_hz(mel): return 700 * (10 ** (mel / 2595) - 1)
        fmin_mel = hz_to_mel(0)
        fmax_mel = hz_to_mel(sr / 2)
        mel_pts  = np.linspace(fmin_mel, fmax_mel, n_mels + 2)
        hz_pts   = mel_to_hz(mel_pts)
        bin_pts  = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
        fb = np.zeros((n_mels, n_fft // 2 + 1))
        for m in range(1, n_mels + 1):
            f_m_minus = bin_pts[m - 1]
            f_m       = bin_pts[m]
            f_m_plus  = bin_pts[m + 1]
            for k in range(f_m_minus, f_m):
                if f_m - f_m_minus > 0:
                    fb[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus):
                if f_m_plus - f_m > 0:
                    fb[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
        return fb

    def db_over_time(self, samples, frame_length=2048, hop_length=512):
        frames = []
        for start in range(0, len(samples) - frame_length, hop_length):
            frame = samples[start: start + frame_length]
            rms   = np.sqrt(np.mean(frame ** 2))
            frames.append(rms)
        rms_arr = np.array(frames, dtype=np.float32)
        db_arr  = 20 * np.log10(rms_arr + 1e-10)
        times   = np.arange(len(db_arr)) * hop_length / self.sr
        return times, db_arr


# ═══════════════════════════════════════════════════════
#  MODÜL 3 – FeatureExtractor
# ═══════════════════════════════════════════════════════

class FeatureExtractor:
    """MFCC, Spectral Centroid, Bandwidth, Rolloff, ZCR, RMS."""

    def __init__(self, sr=22050, n_fft=2048, hop_length=512, n_mfcc=13):
        self.sr         = sr
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.n_mfcc     = n_mfcc

    def extract_all(self, samples):
        features = {}
        freqs, times_stft, Zxx = signal.stft(
            samples, fs=self.sr, window="hann",
            nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length)
        magnitude = np.abs(Zxx)
        power     = magnitude ** 2
        times = times_stft

        sc = np.sum(freqs[:, None] * magnitude, axis=0) / (np.sum(magnitude, axis=0) + 1e-10)
        features["spectral_centroid"] = (times, sc)

        sb = np.sqrt(
            np.sum(((freqs[:, None] - sc[None, :]) ** 2) * magnitude, axis=0) /
            (np.sum(magnitude, axis=0) + 1e-10))
        features["spectral_bandwidth"] = (times, sb)

        cumpower    = np.cumsum(power, axis=0)
        thresh      = 0.85 * cumpower[-1, :]
        rolloff_idx = np.argmax(cumpower >= thresh[None, :], axis=0)
        rolloff_idx = np.clip(rolloff_idx, 0, len(freqs) - 1)
        features["spectral_rolloff"] = (times, freqs[rolloff_idx])

        zcr_list, t_zcr = [], []
        for start in range(0, len(samples) - self.n_fft, self.hop_length):
            frame = samples[start: start + self.n_fft]
            zcr   = np.mean(np.abs(np.diff(np.sign(frame)))) / 2
            zcr_list.append(zcr)
            t_zcr.append(start / self.sr)
        features["zcr"] = (np.array(t_zcr), np.array(zcr_list))

        rms_list, t_rms = [], []
        for start in range(0, len(samples) - self.n_fft, self.hop_length):
            frame = samples[start: start + self.n_fft]
            rms   = np.sqrt(np.mean(frame ** 2))
            rms_list.append(rms)
            t_rms.append(start / self.sr)
        features["rms"] = (np.array(t_rms), np.array(rms_list))

        features["mfcc"] = self._mfcc(samples)
        return features

    def _mfcc(self, samples):
        analyzer = AudioAnalyzer(self.n_fft, self.hop_length, self.sr)
        _, _, mel_db = analyzer.mel_spectrogram(samples, n_mels=40)
        n_mels, n_frames = mel_db.shape
        mfcc = np.zeros((self.n_mfcc, n_frames))
        for m in range(self.n_mfcc):
            mfcc[m] = np.sum(
                mel_db * np.cos(np.pi * m / n_mels *
                                (np.arange(n_mels)[:, None] + 0.5)), axis=0)
        return mfcc

    def feature_summary(self, features):
        summary = {}
        for name, val in features.items():
            arr = val if name == "mfcc" else val[1]
            summary[name] = {"mean": float(np.mean(arr)), "std": float(np.std(arr)),
                             "min": float(np.min(arr)), "max": float(np.max(arr))}
        return summary


# ═══════════════════════════════════════════════════════
#  MODÜL 4 – NoiseFilter
# ═══════════════════════════════════════════════════════

class NoiseFilter:
    """Band-pass, Spectral Gating, Wiener filtresi."""

    def __init__(self, sr=22050):
        self.sr = sr

    def bandpass_filter(self, samples, lowcut=100.0, highcut=2000.0, order=5):
        nyq  = self.sr / 2
        low  = np.clip(lowcut / nyq,  1e-6, 0.9999)
        high = np.clip(highcut / nyq, 1e-6, 0.9999)
        b, a = signal.butter(order, [low, high], btype="band")
        filtered = signal.filtfilt(b, a, samples)
        print(f"[NoiseFilter] Band-pass: {lowcut}–{highcut} Hz, derece={order}")
        return filtered.astype(np.float32)

    def spectral_gating(self, samples, n_fft=2048, hop_length=512,
                         prop_decrease=0.9, n_std_thresh=1.5, noise_clip_fraction=0.1):
        n_noise    = max(n_fft, int(len(samples) * noise_clip_fraction))
        noise_clip = samples[:n_noise]
        _, _, noise_stft = signal.stft(noise_clip, fs=self.sr, window="hann",
                                        nperseg=n_fft, noverlap=n_fft - hop_length)
        noise_power = np.mean(np.abs(noise_stft) ** 2, axis=1, keepdims=True)
        noise_std   = np.std(np.abs(noise_stft), axis=1, keepdims=True)
        _, times_out, sig_stft = signal.stft(samples, fs=self.sr, window="hann",
                                              nperseg=n_fft, noverlap=n_fft - hop_length)
        sig_magnitude = np.abs(sig_stft)
        sig_phase     = np.angle(sig_stft)
        noise_thresh  = np.sqrt(noise_power) + n_std_thresh * noise_std
        mask          = sig_magnitude > noise_thresh
        mask_smoothed = np.clip((sig_magnitude - noise_thresh) / (noise_thresh + 1e-10), 0, 1)
        mask_final    = mask * (1 - prop_decrease) + mask_smoothed * prop_decrease
        sig_filtered_stft = mask_final * sig_magnitude * np.exp(1j * sig_phase)
        _, filtered = signal.istft(sig_filtered_stft, fs=self.sr, window="hann",
                                    nperseg=n_fft, noverlap=n_fft - hop_length)
        print(f"[NoiseFilter] Spectral gating: prop_decrease={prop_decrease}")
        return filtered[:len(samples)].astype(np.float32)

    def wiener_filter(self, samples, n_fft=2048, hop_length=512,
                       noise_power_estimate_frames=10):
        _, _, stft = signal.stft(samples, fs=self.sr, window="hann",
                                  nperseg=n_fft, noverlap=n_fft - hop_length)
        S    = np.abs(stft) ** 2
        N    = np.maximum(np.mean(S[:, :noise_power_estimate_frames], axis=1, keepdims=True), 1e-10)
        gain = np.maximum((S - N) / S, 0.0)
        _, filtered = signal.istft(stft * np.sqrt(gain), fs=self.sr, window="hann",
                                    nperseg=n_fft, noverlap=n_fft - hop_length)
        print("[NoiseFilter] Wiener filtresi uygulandı")
        return filtered[:len(samples)].astype(np.float32)


# ═══════════════════════════════════════════════════════
#  MODÜL 5 – NoiseClassifier (Kural Tabanlı)
# ═══════════════════════════════════════════════════════

class NoiseClassifier:
    """Kural tabanlı sınıflandırıcı. Gerçek ML modeli gelene kadar kullanılır."""

    LABELS = ["AIRCRAFT", "WIND", "TRAFFIC", "SPEECH", "UNKNOWN"]

    THRESHOLDS = {
        "aircraft_sc_min":  400,
        "aircraft_sc_max":  2000,
        "aircraft_rms_min": 0.05,
        "wind_sc_max":      500,
        "wind_rms_max":     0.10,
        "traffic_sc_min":   200,
        "traffic_sc_max":   1500,
        "speech_sc_min":    500,
        "speech_sc_max":    3500,
        "speech_zcr_min":   0.05,
    }

    def classify_frame(self, sc, zcr, rms):
        th = self.THRESHOLDS
        if th["aircraft_sc_min"] < sc < th["aircraft_sc_max"] and rms > th["aircraft_rms_min"]:
            return "AIRCRAFT"
        if sc < th["wind_sc_max"] and rms < th["wind_rms_max"]:
            return "WIND"
        if th["speech_sc_min"] < sc < th["speech_sc_max"] and zcr > th["speech_zcr_min"]:
            return "SPEECH"
        if th["traffic_sc_min"] < sc < th["traffic_sc_max"]:
            return "TRAFFIC"
        return "UNKNOWN"

    def classify(self, features):
        times_sc, sc = features["spectral_centroid"]
        times_zcr, zcr = features["zcr"]
        times_rms, rms = features["rms"]
        n = min(len(sc), len(zcr), len(rms))
        sc = sc[:n]; zcr = zcr[:n]; rms = rms[:n]; t = times_sc[:n]
        rms_norm = rms / (np.max(rms) + 1e-10)
        labels = [self.classify_frame(sc[i], zcr[i], rms_norm[i]) for i in range(n)]
        from collections import Counter
        counts  = Counter(labels)
        summary = {k: round(100 * v / len(labels), 1) for k, v in counts.items()}
        print(f"[NoiseClassifier] {n} çerçeve: {summary}")
        return labels, t, summary


# ═══════════════════════════════════════════════════════
#  MODÜL 6 – Visualizer
# ═══════════════════════════════════════════════════════

class Visualizer:
    """Tüm grafikleri matplotlib ile çizer ve PNG olarak kaydeder."""

    LABEL_COLORS = {
        "AIRCRAFT": "#FF6B35",
        "WIND":     "#4ECDC4",
        "TRAFFIC":  "#FFE66D",
        "SPEECH":   "#A8DADC",
        "UNKNOWN":  "#6C757D",
    }

    def __init__(self, output_dir="outputs"):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def _style(self, ax):
        ax.set_facecolor("#161B22")
        ax.tick_params(colors="#8B949E", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")

    def plot_waveform(self, samples, sr, filename_prefix="airport_noise"):
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor("#0D1117")
        self._style(ax)
        t = np.linspace(0, len(samples) / sr, len(samples))
        ax.plot(t, samples, color="#00D4FF", lw=0.5, alpha=0.85)
        ax.fill_between(t, samples, 0, color="#00D4FF", alpha=0.08)
        ax.set_title("Waveform (Dalga Formu)", color="white", fontsize=13)
        ax.set_xlabel("Zaman (s)", color="#8B949E")
        ax.set_ylabel("Genlik", color="#8B949E")
        ax.grid(True, color="#21262D", linewidth=0.5)
        path = self.out / f"{filename_prefix}_1_waveform.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[Visualizer] Waveform → {path}")
        return str(path)

    def plot_db_over_time(self, samples, sr, filename_prefix="airport_noise"):
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor("#0D1117")
        self._style(ax)
        analyzer = AudioAnalyzer(sr=sr)
        t_db, db = analyzer.db_over_time(samples)
        ax.plot(t_db, db, color="#FF6B35", lw=1.2)
        ax.fill_between(t_db, db, np.min(db) - 5, color="#FF6B35", alpha=0.15)
        ax.set_title("Ses Seviyesi (dBFS / Zaman)", color="white", fontsize=13)
        ax.set_xlabel("Zaman (s)", color="#8B949E")
        ax.set_ylabel("dBFS", color="#8B949E")
        ax.grid(True, color="#21262D", linewidth=0.5)
        path = self.out / f"{filename_prefix}_2_db_over_time.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[Visualizer] dB/Zaman → {path}")
        return str(path)

    def plot_spectrogram(self, samples, sr, filename_prefix="airport_noise"):
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#0D1117")
        self._style(ax)
        analyzer = AudioAnalyzer(sr=sr)
        f, t, S_db = analyzer.spectrogram(samples)
        im = ax.pcolormesh(t, f, S_db, shading="auto", cmap="inferno",
                           vmin=np.percentile(S_db, 10))
        ax.set_ylim(0, min(8000, sr / 2))
        ax.set_title("Spektrogram (STFT)", color="white", fontsize=13)
        ax.set_xlabel("Zaman (s)", color="#8B949E")
        ax.set_ylabel("Frekans (Hz)", color="#8B949E")
        cb = plt.colorbar(im, ax=ax)
        cb.set_label("dB", color="#8B949E")
        cb.ax.yaxis.set_tick_params(color="#8B949E")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8B949E")
        path = self.out / f"{filename_prefix}_3_spectrogram.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[Visualizer] Spektrogram → {path}")
        return str(path)

    def plot_mel_spectrogram(self, samples, sr, filename_prefix="airport_noise"):
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#0D1117")
        self._style(ax)
        analyzer = AudioAnalyzer(sr=sr)
        _, _, mel_db = analyzer.mel_spectrogram(samples, n_mels=64)
        im = ax.imshow(mel_db, aspect="auto", origin="lower", cmap="magma",
                       extent=[0, len(samples) / sr, 0, mel_db.shape[0]])
        ax.set_title("Mel Spektrogram", color="white", fontsize=13)
        ax.set_xlabel("Zaman (s)", color="#8B949E")
        ax.set_ylabel("Mel Kanalı", color="#8B949E")
        cb = plt.colorbar(im, ax=ax)
        cb.set_label("dB", color="#8B949E")
        cb.ax.yaxis.set_tick_params(color="#8B949E")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8B949E")
        path = self.out / f"{filename_prefix}_4_mel_spectrogram.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[Visualizer] Mel Spektrogram → {path}")
        return str(path)

    def plot_mfcc(self, features, samples, sr, filename_prefix="airport_noise"):
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#0D1117")
        self._style(ax)
        mfcc = features["mfcc"]
        im = ax.imshow(mfcc, aspect="auto", origin="lower", cmap="coolwarm",
                       extent=[0, len(samples) / sr, 0, mfcc.shape[0]])
        ax.set_title("MFCC (13 Katsayı)", color="white", fontsize=13)
        ax.set_xlabel("Zaman (s)", color="#8B949E")
        ax.set_ylabel("MFCC Katsayısı", color="#8B949E")
        cb = plt.colorbar(im, ax=ax)
        cb.ax.yaxis.set_tick_params(color="#8B949E")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8B949E")
        path = self.out / f"{filename_prefix}_5_mfcc.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[Visualizer] MFCC → {path}")
        return str(path)

    def plot_features(self, features, filename_prefix="airport_noise"):
        plt.style.use("dark_background")
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
        fig.patch.set_facecolor("#0D1117")
        fig.suptitle("Spektral Özellikler", color="white", fontsize=14)
        datasets = [
            ("spectral_centroid", "Spectral Centroid (Hz)", "#7EE8A2"),
            ("zcr",               "Zero Crossing Rate",     "#FFE66D"),
            ("rms",               "RMS Enerji",              "#FF6B9D"),
        ]
        for ax, (key, title, color) in zip(axes, datasets):
            t, vals = features[key]
            self._style(ax)
            ax.plot(t, vals, color=color, lw=1.0)
            ax.fill_between(t, vals, alpha=0.12, color=color)
            ax.set_title(title, color="white", fontsize=11)
            ax.set_xlabel("Zaman (s)", color="#8B949E", fontsize=9)
            ax.grid(True, color="#21262D", linewidth=0.5)
        plt.tight_layout()
        path = self.out / f"{filename_prefix}_6_features.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[Visualizer] Özellikler → {path}")
        return str(path)

    def plot_filter_comparison(self, samples, sr, filtered_bp, filtered_sg,
                                filename_prefix="airport_noise"):
        plt.style.use("dark_background")
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.patch.set_facecolor("#0D1117")
        fig.suptitle("Filtre Karşılaştırması", color="white", fontsize=14)
        signals = [
            (samples,      "Orijinal Sinyal",         "#00D4FF"),
            (filtered_bp,  "Band-pass Filtreli",       "#FF6B35"),
            (filtered_sg,  "Spectral Gating Filtreli", "#7EE8A2"),
        ]
        for ax, (sig, title, color) in zip(axes, signals):
            t = np.linspace(0, len(sig) / sr, len(sig))
            self._style(ax)
            ax.plot(t, sig, color=color, lw=0.6, alpha=0.9)
            ax.fill_between(t, sig, 0, color=color, alpha=0.07)
            ax.set_title(title, color="white", fontsize=11)
            ax.set_ylabel("Genlik", color="#8B949E", fontsize=9)
            ax.grid(True, color="#21262D", linewidth=0.5)
        axes[-1].set_xlabel("Zaman (s)", color="#8B949E")
        plt.tight_layout()
        path = self.out / f"{filename_prefix}_7_filter_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[Visualizer] Filtre karşılaştırması → {path}")
        return str(path)

    def plot_full_dashboard(self, samples, sr, features, filtered_bp, filtered_sg,
                             frame_labels, label_times, filename_prefix="airport_noise"):
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(22, 18))
        fig.patch.set_facecolor("#0D1117")
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
        ax_wf  = fig.add_subplot(gs[0, 0]); ax_db  = fig.add_subplot(gs[0, 1])
        ax_cls = fig.add_subplot(gs[0, 2]); ax_spec = fig.add_subplot(gs[1, 0])
        ax_mel = fig.add_subplot(gs[1, 1]); ax_sc  = fig.add_subplot(gs[1, 2])
        ax_filt= fig.add_subplot(gs[2, 0]); ax_mfcc= fig.add_subplot(gs[2, 1])
        ax_zcr = fig.add_subplot(gs[2, 2])
        ACCENT = "#00D4FF"; ACCENT2 = "#FF6B35"; BG = "#161B22"; GRID = "#21262D"
        for ax in [ax_wf, ax_db, ax_cls, ax_spec, ax_mel, ax_sc, ax_filt, ax_mfcc, ax_zcr]:
            self._style(ax)
        analyzer = AudioAnalyzer(sr=sr)
        t, s = analyzer.waveform(samples)
        ax_wf.plot(t, s, color=ACCENT, lw=0.5, alpha=0.85)
        ax_wf.fill_between(t, s, 0, color=ACCENT, alpha=0.1)
        ax_wf.set_title("Waveform", color="white", fontsize=10, pad=6)
        ax_wf.set_xlabel("Zaman (s)", color="#8B949E", fontsize=8)
        ax_wf.set_ylabel("Genlik", color="#8B949E", fontsize=8)
        ax_wf.grid(True, color=GRID, linewidth=0.5)
        t_db, db = analyzer.db_over_time(samples)
        ax_db.plot(t_db, db, color=ACCENT2, lw=1.2)
        ax_db.fill_between(t_db, db, np.min(db) - 5, color=ACCENT2, alpha=0.15)
        ax_db.set_title("Ses Seviyesi (dBFS)", color="white", fontsize=10, pad=6)
        ax_db.set_xlabel("Zaman (s)", color="#8B949E", fontsize=8)
        ax_db.set_ylabel("dBFS", color="#8B949E", fontsize=8)
        ax_db.grid(True, color=GRID, linewidth=0.5)
        label_map  = {l: i for i, l in enumerate(NoiseClassifier.LABELS)}
        y_cls      = np.array([label_map.get(l, 4) for l in frame_labels])
        colors_cls = [self.LABEL_COLORS.get(l, "#6C757D") for l in frame_labels]
        bar_w = label_times[1] - label_times[0] if len(label_times) > 1 else 0.02
        ax_cls.bar(label_times, y_cls + 1, width=bar_w, color=colors_cls, alpha=0.85)
        ax_cls.set_yticks(range(1, len(NoiseClassifier.LABELS) + 1))
        ax_cls.set_yticklabels(NoiseClassifier.LABELS, fontsize=7, color="#8B949E")
        ax_cls.set_title("Sınıflandırma", color="white", fontsize=10, pad=6)
        ax_cls.set_xlabel("Zaman (s)", color="#8B949E", fontsize=8)
        ax_cls.grid(True, axis="x", color=GRID, linewidth=0.5)
        f_s, t_s, S_db = analyzer.spectrogram(samples)
        im = ax_spec.pcolormesh(t_s, f_s, S_db, shading="auto", cmap="inferno",
                                vmin=np.percentile(S_db, 10))
        ax_spec.set_ylim(0, min(8000, sr / 2))
        ax_spec.set_title("Spektrogram (STFT)", color="white", fontsize=10, pad=6)
        ax_spec.set_xlabel("Zaman (s)", color="#8B949E", fontsize=8)
        ax_spec.set_ylabel("Frekans (Hz)", color="#8B949E", fontsize=8)
        plt.colorbar(im, ax=ax_spec, label="dB").ax.yaxis.label.set_color("#8B949E")
        _, _, mel_db = analyzer.mel_spectrogram(samples, n_mels=64)
        im2 = ax_mel.pcolormesh(np.linspace(0, len(samples)/sr, mel_db.shape[1]),
                                np.arange(mel_db.shape[0]), mel_db, shading="auto", cmap="magma")
        ax_mel.set_title("Mel Spektrogram", color="white", fontsize=10, pad=6)
        ax_mel.set_xlabel("Zaman (s)", color="#8B949E", fontsize=8)
        ax_mel.set_ylabel("Mel Kanalı", color="#8B949E", fontsize=8)
        plt.colorbar(im2, ax=ax_mel, label="dB").ax.yaxis.label.set_color("#8B949E")
        t_sc, sc = features["spectral_centroid"]
        ax_sc.plot(t_sc, sc, color="#7EE8A2", lw=1.0)
        ax_sc.set_title("Spectral Centroid (Hz)", color="white", fontsize=10, pad=6)
        ax_sc.set_xlabel("Zaman (s)", color="#8B949E", fontsize=8)
        ax_sc.set_ylabel("Hz", color="#8B949E", fontsize=8)
        ax_sc.grid(True, color=GRID, linewidth=0.5)
        ax_filt.plot(t, samples, color=ACCENT, lw=0.5, alpha=0.5, label="Orijinal")
        t_bp = np.linspace(0, len(filtered_bp)/sr, len(filtered_bp))
        ax_filt.plot(t_bp, filtered_bp, color=ACCENT2, lw=0.8, alpha=0.75, label="Band-pass")
        t_sg = np.linspace(0, len(filtered_sg)/sr, len(filtered_sg))
        ax_filt.plot(t_sg, filtered_sg, color="#7EE8A2", lw=0.8, alpha=0.75, label="Sp.Gating")
        ax_filt.set_title("Filtre Karşılaştırması", color="white", fontsize=10, pad=6)
        ax_filt.set_xlabel("Zaman (s)", color="#8B949E", fontsize=8)
        ax_filt.set_ylabel("Genlik", color="#8B949E", fontsize=8)
        ax_filt.legend(fontsize=7, facecolor=BG, labelcolor="white")
        ax_filt.grid(True, color=GRID, linewidth=0.5)
        mfcc = features["mfcc"]
        im3  = ax_mfcc.imshow(mfcc, aspect="auto", origin="lower", cmap="coolwarm",
                               extent=[0, len(samples)/sr, 0, mfcc.shape[0]])
        ax_mfcc.set_title("MFCC (13 Katsayı)", color="white", fontsize=10, pad=6)
        ax_mfcc.set_xlabel("Zaman (s)", color="#8B949E", fontsize=8)
        ax_mfcc.set_ylabel("MFCC Katsayısı", color="#8B949E", fontsize=8)
        plt.colorbar(im3, ax=ax_mfcc).ax.yaxis.label.set_color("#8B949E")
        t_zcr, zcr_vals = features["zcr"]
        t_rms, rms_vals = features["rms"]
        ax_zcr2 = ax_zcr.twinx()
        ax_zcr.plot(t_zcr, zcr_vals, color="#FFE66D", lw=1.0, label="ZCR")
        ax_zcr2.plot(t_rms, rms_vals, color="#FF6B9D", lw=1.0, alpha=0.8, label="RMS")
        ax_zcr.set_title("ZCR & RMS Enerji", color="white", fontsize=10, pad=6)
        ax_zcr.set_xlabel("Zaman (s)", color="#8B949E", fontsize=8)
        ax_zcr.set_ylabel("ZCR", color="#FFE66D", fontsize=8)
        ax_zcr2.set_ylabel("RMS", color="#FF6B9D", fontsize=8)
        ax_zcr.grid(True, color=GRID, linewidth=0.5)
        lines1, labs1 = ax_zcr.get_legend_handles_labels()
        lines2, labs2 = ax_zcr2.get_legend_handles_labels()
        ax_zcr.legend(lines1 + lines2, labs1 + labs2, fontsize=7, facecolor=BG, labelcolor="white")
        ax_zcr2.tick_params(colors="#8B949E", labelsize=8)
        fig.suptitle("✈  HAVALIMANL ÇEVRESEL GÜRÜLTÜ ANALİZ PANOSU  ✈",
                     color="white", fontsize=15, fontweight="bold", y=0.98)
        path = self.out / f"{filename_prefix}_dashboard.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[Visualizer] Dashboard → {path}")
        return str(path)

    def plot_classification_pie(self, summary, filename_prefix="airport_noise"):
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.patch.set_facecolor("#0D1117")
        ax.set_facecolor("#0D1117")
        labels = list(summary.keys())
        sizes  = list(summary.values())
        colors = [self.LABEL_COLORS.get(l, "#6C757D") for l in labels]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140,
            textprops={"color": "white", "fontsize": 11},
            wedgeprops={"edgecolor": "#0D1117", "linewidth": 2})
        for at in autotexts:
            at.set_fontsize(10)
        ax.set_title("Gürültü Kaynağı Dağılımı", color="white", fontsize=14, pad=15)
        path = self.out / f"{filename_prefix}_classification_pie.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[Visualizer] Pasta grafiği → {path}")
        return str(path)


# ═══════════════════════════════════════════════════════
#  MODÜL 7 – AirportNoiseSystem (Orkestratör)
# ═══════════════════════════════════════════════════════

class AirportNoiseSystem:

    def __init__(self, target_sr=22050, output_dir="outputs"):
        self.loader     = AudioLoader(target_sr)
        self.analyzer   = AudioAnalyzer(sr=target_sr)
        self.extractor  = FeatureExtractor(sr=target_sr)
        self.filter     = NoiseFilter(sr=target_sr)
        self.classifier = NoiseClassifier()
        self.visualizer = Visualizer(output_dir)
        self.sr         = target_sr
        self.output_dir = output_dir

    def run(self, audio_path: str, prefix: str = "airport_noise"):
        print("\n" + "=" * 60)
        print("  HAVALIMANL GÜRÜLTÜ ANALİZ SİSTEMİ BAŞLADI")
        print("=" * 60)

        print("\n[ADIM 1] Ses dosyası yükleniyor...")
        samples, sr = self.loader.load(audio_path)

        print("\n[ADIM 2] Özellikler çıkarılıyor...")
        features = self.extractor.extract_all(samples)
        summary  = self.extractor.feature_summary(features)
        for k, v in summary.items():
            if k != "mfcc":
                print(f"    {k:22s}: ort={v['mean']:8.2f}  std={v['std']:.2f}")

        print("\n[ADIM 3] Sınıflandırılıyor...")
        frame_labels, label_times, cls_summary = self.classifier.classify(features)

        print("\n[ADIM 4] Filtreleme uygulanıyor...")
        filtered_bp = self.filter.bandpass_filter(samples, lowcut=100, highcut=2000)
        filtered_sg = self.filter.spectral_gating(samples, prop_decrease=0.85, n_std_thresh=1.5)

        print("\n[ADIM 5] Grafikler oluşturuluyor...")
        paths = []
        paths.append(self.visualizer.plot_waveform(samples, sr, prefix))
        paths.append(self.visualizer.plot_db_over_time(samples, sr, prefix))
        paths.append(self.visualizer.plot_spectrogram(samples, sr, prefix))
        paths.append(self.visualizer.plot_mel_spectrogram(samples, sr, prefix))
        paths.append(self.visualizer.plot_mfcc(features, samples, sr, prefix))
        paths.append(self.visualizer.plot_features(features, prefix))
        paths.append(self.visualizer.plot_filter_comparison(samples, sr, filtered_bp, filtered_sg, prefix))
        paths.append(self.visualizer.plot_full_dashboard(
            samples, sr, features, filtered_bp, filtered_sg, frame_labels, label_times, prefix))
        paths.append(self.visualizer.plot_classification_pie(cls_summary, prefix))

        report = {
            "audio_path":      audio_path,
            "sample_rate":     sr,
            "duration_s":      len(samples) / sr,
            "n_samples":       len(samples),
            "feature_summary": summary,
            "classification":  cls_summary,
            "output_files":    paths,
        }

        print("\n" + "=" * 60)
        print(f"  ANALİZ TAMAMLANDI | Süre: {report['duration_s']:.2f}s | SR: {sr} Hz")
        print(f"  Dağılım: {cls_summary}")
        print("=" * 60 + "\n")
        return report
```

---

### `main.py`

```python
"""
Havalimanı Gürültü Tespit Sistemi – Çalıştırıcı

Kullanım:
  python main.py --demo
  python main.py --audio "C:\\...\\ses.wav"
  python main.py --audio ses.wav --sr 44100 --out sonuclar/
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from noise_detector import AirportNoiseSystem, _synth_wav


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audio",  type=str, default=None)
    p.add_argument("--demo",   action="store_true")
    p.add_argument("--sr",     type=int, default=22050)
    p.add_argument("--out",    type=str, default="outputs")
    p.add_argument("--prefix", type=str, default="airport_noise")
    return p.parse_args()


def main():
    args = parse_args()

    if args.demo or args.audio is None:
        print("[DEMO MODU] Sentetik ses üretiliyor...")
        audio_path = "demo_airport_sound.wav"
        _synth_wav(audio_path, duration=10.0, sr=args.sr)
    else:
        audio_path = args.audio
        if not os.path.exists(audio_path):
            print(f"[HATA] Dosya bulunamadı: {audio_path}")
            sys.exit(1)

    system = AirportNoiseSystem(target_sr=args.sr, output_dir=args.out)
    report = system.run(audio_path, prefix=args.prefix)

    print("\n📊  RAPOR")
    print(f"  Dosya    : {report['audio_path']}")
    print(f"  Süre     : {report['duration_s']:.2f} s")
    print(f"  SR       : {report['sample_rate']} Hz")
    print("\n  🎯 Dağılım:")
    for label, pct in sorted(report["classification"].items(), key=lambda x: -x[1]):
        bar = "█" * int(pct / 2)
        print(f"    {label:10s} {bar:50s} {pct:5.1f}%")


if __name__ == "__main__":
    main()
```

---

### `ml_classifier.py`

```python
"""
ML Tabanlı Sınıflandırıcı – Uzantı Modülü

Şu an: GaussianNB ve KNN (kural tabanlı etiketlerden öğreniyor — geçici)
Sonraki: ESC-50 verileriyle eğitilmiş RandomForest / SVM

NOT: Bu modül şu an gerçek anlamda bağımsız bir model değil.
     ESC-50 eğitimi tamamlandıktan sonra bu dosya tamamen yeniden yazılacak.
"""

import numpy as np
from noise_detector import FeatureExtractor, NoiseClassifier, AudioLoader, _synth_wav


def build_feature_vector(features, frame_idx: int) -> np.ndarray:
    """[sc, sb, sr, zcr, rms] → 5 boyutlu vektör"""
    t_sc, sc   = features["spectral_centroid"]
    t_sb, sb   = features["spectral_bandwidth"]
    t_sr, sr_  = features["spectral_rolloff"]
    t_zcr, zcr = features["zcr"]
    t_rms, rms = features["rms"]
    n = min(len(sc), len(sb), len(sr_), len(zcr), len(rms))
    if frame_idx >= n:
        frame_idx = n - 1
    return np.array([sc[frame_idx], sb[frame_idx], sr_[frame_idx],
                     zcr[frame_idx], rms[frame_idx]], dtype=np.float32)


def build_dataset_from_features(features, rule_labels):
    label_to_idx = {l: i for i, l in enumerate(NoiseClassifier.LABELS)}
    n = len(rule_labels)
    X = np.zeros((n, 5), dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)
    for i in range(n):
        X[i] = build_feature_vector(features, i)
        y[i] = label_to_idx.get(rule_labels[i], 4)
    return X, y


class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes_  = np.unique(y)
        self.n_classes_= len(self.classes_)
        n_features     = X.shape[1]
        self.priors_   = np.zeros(self.n_classes_)
        self.means_    = np.zeros((self.n_classes_, n_features))
        self.vars_     = np.zeros((self.n_classes_, n_features))
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.priors_[i] = len(X_c) / len(X)
            self.means_[i]  = X_c.mean(axis=0)
            self.vars_[i]   = X_c.var(axis=0) + 1e-9
        return self

    def _log_likelihood(self, X):
        log_probs = np.zeros((len(X), self.n_classes_))
        for i in range(self.n_classes_):
            log_p = -0.5 * np.sum(
                np.log(2 * np.pi * self.vars_[i]) +
                ((X - self.means_[i]) ** 2) / self.vars_[i], axis=1)
            log_probs[:, i] = log_p + np.log(self.priors_[i])
        return log_probs

    def predict(self, X):
        return self.classes_[np.argmax(self._log_likelihood(X), axis=1)]


class KNNClassifier:
    def __init__(self, k=5): self.k = k

    def fit(self, X, y):
        self.X_mean_  = X.mean(axis=0)
        self.X_std_   = X.std(axis=0) + 1e-9
        self.X_train_ = (X - self.X_mean_) / self.X_std_
        self.y_train_ = y
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        X_norm = (X - self.X_mean_) / self.X_std_
        preds  = np.zeros(len(X), dtype=np.int32)
        for i, x in enumerate(X_norm):
            dists    = np.sqrt(np.sum((self.X_train_ - x) ** 2, axis=1))
            k_idx    = np.argsort(dists)[:self.k]
            k_labels = self.y_train_[k_idx]
            preds[i] = np.bincount(k_labels, minlength=len(self.classes_)).argmax()
        return preds


class MLNoiseClassifier:
    LABELS = NoiseClassifier.LABELS

    def __init__(self, model_type="gnb", k_neighbors=7):
        self.model_type = model_type
        self.model = GaussianNaiveBayes() if model_type == "gnb" else KNNClassifier(k=k_neighbors)
        self._trained = False

    def bootstrap_train(self, samples, sr=22050):
        extractor = FeatureExtractor(sr=sr)
        rule_clf  = NoiseClassifier()
        features  = extractor.extract_all(samples)
        rule_labels, _, _ = rule_clf.classify(features)
        X, y = build_dataset_from_features(features, rule_labels)
        n_train = int(0.8 * len(X))
        idx = np.random.permutation(len(X))
        self.model.fit(X[idx[:n_train]], y[idx[:n_train]])
        self._trained = True
        y_pred   = self.model.predict(X[idx[n_train:]])
        accuracy = np.mean(y_pred == y[idx[n_train:]]) * 100
        print(f"[MLClassifier] {self.model_type.upper()} | Doğruluk: {accuracy:.1f}%")
        return accuracy

    def predict_all(self, features):
        if not self._trained:
            raise RuntimeError("Önce bootstrap_train() çalıştırın.")
        n = len(features["spectral_centroid"][0])
        X = np.array([build_feature_vector(features, i) for i in range(n)])
        return [self.LABELS[p] for p in self.model.predict(X)]
```

---

### `train_model.py` — YAZILACAK (Şablon)

```python
"""
ESC-50 veri seti ile RandomForest / SVM eğitimi.
Bu dosya henüz yazılmadı — ESC-50 indirildikten sonra oluşturulacak.

Yapılacaklar:
  1. ESC-50 CSV'sini oku
  2. Kullanılacak kategorileri filtrele (airplane, helicopter, wind, vb.)
  3. Her ses dosyasından özellik vektörü çıkar (MFCC + spectral)
  4. Kendi uçak seslerini ekle (kendi_veri/airplane/)
  5. RF + SVM eğit, cross-validation uygula
  6. En iyi modeli models/ klasörüne kaydet (joblib)
  7. noise_detector.py'e entegre et
"""
```

---

## Çalıştırma Komutları

```bash
# Aktive et (her VS Code oturumunda)
venv\Scripts\activate

# Demo (sentetik ses)
python main.py --demo

# Gerçek ses dosyası
python main.py --audio "sounds/Flying Plane Sound Effect.wav"

# Farklı örnekleme hızı
python main.py --audio ses.wav --sr 44100

# ML sınıflandırıcı demo (bootstrap — geçici)
python ml_classifier.py
```

---

## Şu Anki Durum Özeti

```
✅ Ses yükleme (WAV + MP3, librosa ile)
✅ STFT / Mel spektrogram / dBFS analizi
✅ Özellik çıkarımı (MFCC, SC, SB, ZCR, RMS)
✅ Band-pass, Spectral Gating, Wiener filtresi
✅ Kural tabanlı sınıflandırma (zayıf — trafik→uçak hatası var)
✅ 9 ayrı grafik + dashboard + pasta grafiği
✅ Gerçek ses dosyasıyla test edildi

⏳ ESC-50 veri seti indirilecek
⏳ train_model.py yazılacak (RF + SVM)
⏳ Özellik mühendisliği geliştirilecek (spectral flatness, delta MFCC, band enerji oranları)
⏳ Eğitilmiş model noise_detector.py'e entegre edilecek
⏳ Gerçek zamanlı sistem (sounddevice)
```
