"""
=============================================================
 HAVALIMANL ÇEVRESEL GÜRÜLTÜ TESPİT SİSTEMİ
 Airport Environmental Noise Detection System
=============================================================
Modüler yapı:
  1. AudioLoader        – WAV/MP3 yükleme & normalize
  2. AudioAnalyzer      – Waveform, spektrogram, dB analizi
  3. FeatureExtractor   – MFCC, spectral centroid, ZCR, RMS
  4. NoiseFilter        – Band-pass, spectral gating
  5. NoiseClassifier    – Kural tabanlı sınıflandırıcı (yedek)
  6. Visualizer         – Tüm grafikleri çizer
  7. AirportCNN         – 3-katmanlı CNN (PyTorch)
  8. EfficientNetAirport – EfficientNet-B0 transfer learning
  9. BEATsClassifier    – Microsoft BEATs frozen encoder + MLP (v5.1)
 10. EnsembleClassifier – EfficientNet + BEATs softmax ensemble (v5.1)
 11. AirportNoiseSystem – Orkestratör (tüm modelleri yönetir)

v5.1 eklemeleri:
  - BEATsClassifier: frozen BEATs-Small encoder + 2-layer MLP
  - EnsembleClassifier: EfficientNet + BEATs softmax ortalaması (α=0.5)
  - AirportNoiseSystem: beats / ensemble model_pref desteği

v6 — Sınıf taksonomisi yenilendi:
  - Eski 6 düz sınıf (AIRCRAFT/AMBIENT/SPEECH/TRAFFIC/WIND/OTHER) yerine
    9 aktif + OTHER = 10 sınıf (bkz. class_config.py). Sınıf listesi,
    renkler ve ağırlıklar artık class_config.py'den import ediliyor —
    burada veya başka bir dosyada elle KOPYALANMAMALI.
  - Eğitim verisi kaynağı değişti: eski ESC-50/AeroSonicDB/GENERIC_AUDIO
    veri seti tamamen iptal edildi. Yeni kaynak: airport-audio-collector
    projesinin SQLite pipeline'ı + Svantek kayıtları + onaylı live klipler
    (bkz. dataset_builder.py).
"""

import struct
import wave
import os
import sys
import numpy as np
import scipy.signal as signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# Windows PowerShell bazı sistemlerde cp1254 ile açılır. Model yükleme
# günlüklerindeki Unicode karakterleri yazılırken hata oluşursa geniş try/except
# blokları modeli yanlışlıkla "yüklenemedi" durumuna geçiriyordu.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── ML entegrasyonu için gerekli kütüphaneler ──────────────────
try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False
    print("[UYARI] librosa bulunamadı — ML sınıflandırıcı devre dışı")

try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False
    print("[UYARI] joblib bulunamadı — ML sınıflandırıcı devre dışı")

try:
    import torch
    import torch.nn as nn
    import torchaudio.transforms as T
    TORCH_OK = True
    _TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    TORCH_OK = False
    print("[UYARI] torch/torchaudio bulunamadı — CNN devre dışı")

try:
    import torchvision.transforms as TV
    import torchvision.models as tvm
    TORCHVISION_OK = True
except ImportError:
    TORCHVISION_OK = False
    print("[UYARI] torchvision bulunamadı — EfficientNet devre dışı")

# ── BEATs entegrasyonu ────────────────────────────────────────
# Kurulum: microsoft/unilm reposunu klonla, BEATs klasörünü proje
# köküne kopyala. Alternatif: pip install unilm (varsa).
# Model dosyası: D:\models\BEATs_iter3_plus_AS2M.pt
try:
    from BEATs import BEATs, BEATsConfig  # type: ignore
    BEATS_OK = True
except ImportError:
    BEATS_OK = False
    # Sessizce geç — BEATs seçilirse o an uyarı basılır

# ── Sınıf tanımları — TEK kaynak class_config.py ────────────────
# Sınıf ismi/renk/ağırlık eklemek veya çıkarmak için SADECE class_config.py
# değişir; bu dosyada elle sınıf listesi kopyalanmaz.
from class_config import (
    CLASSES as _SHARED_CLASSES,
    CLASS_COLORS as _SHARED_CLASS_COLORS,
    INFERENCE_PRIOR_WEIGHTS as _SHARED_PRIOR_WEIGHTS,
)


# ══════════════════════════════════════════════════════════════
#  ML ENTEGRASYONU — extract_features_ml
#  dataset_builder_v2.py ile BİREBİR AYNI olmalı (264 boyut)
# ══════════════════════════════════════════════════════════════

# Parametreler — dataset_builder_v2.py ile senkronize
_ML_SR      = 22050
_ML_CLIP    = 5.0
_ML_HOP_SEC = 2.5
_ML_N_MFCC  = 40
_ML_N_FFT   = 2048
_ML_HOP_FFT = 512

# CNN parametreleri — train_cnn.py ile senkronize
_CNN_N_MELS = 128

# EfficientNet parametreleri — train_efficientnet.py ile senkronize
_EFF_N_MELS        = 128
_EFF_IMG_SIZE      = 224
_EFF_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_EFF_IMAGENET_STD  = [0.229, 0.224, 0.225]

# BEATs parametreleri (v5.1)
# Birleşik uygulamada modeller proje ile birlikte taşınır; kişisel D: yolu gerekmez.
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_BEATS_ENCODER_PATH = os.path.join(_MODEL_DIR, "BEATs_iter3_plus_AS2M.pt")
_BEATS_MLP_PATH     = os.path.join(_MODEL_DIR, "beats_mlp.pt")
_BEATS_EMBED_DIM    = 768
_BEATS_CLASSES      = _SHARED_CLASSES   # class_config.CLASSES — 9 aktif + OTHER
_ENSEMBLE_ALPHA     = 0.5   # EfficientNet ağırlığı; (1-α) BEATs ağırlığı


# ─── CNN Modeli (train_cnn.py ile aynı mimari) ────────────────────────────
# Torch yüklü değilse bu sınıf hiç kullanılmaz

if TORCH_OK:
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
                nn.ReLU(inplace=True), nn.Dropout(0.5),
                nn.Linear(256, n_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    # EfficientNet-B0 mimarisi — train_efficientnet.py ile aynı
    if TORCHVISION_OK:
        class EfficientNetAirport(nn.Module):
            def __init__(self, n_classes: int = 5):
                super().__init__()
                # Ağırlıkları boş başlat (yüklenmiş checkpoint'ten gelecek)
                backbone        = tvm.efficientnet_b0(weights=None)
                self.features   = backbone.features
                self.avgpool    = backbone.avgpool
                self.classifier = nn.Sequential(
                    nn.Dropout(p=0.4),
                    nn.Linear(1280, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.4),
                    nn.Linear(512, n_classes),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.features(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                return self.classifier(x)


# ══════════════════════════════════════════════════════════════
#  BEATsClassifier  (v5.1)
#  Microsoft BEATs-Small encoder (frozen) + 2-layer MLP
# ══════════════════════════════════════════════════════════════

if TORCH_OK:
    class BEATsClassifier(nn.Module):
        """
        BEATs-Small encoder (frozen) + lightweight MLP sınıflandırıcı.

        Encoder hiç eğitilmez — sadece MLP katmanları train_beats.py ile eğitilir.
        Girdi: ham waveform (float32, SR=22050, 5s → 110250 örnek)
        Çıktı: 10-sınıf logit (bkz. class_config.CLASSES)

        Kurulum:
          1. https://github.com/microsoft/unilm BEATs klasörünü proje köküne kopyala
          2. D:\\models\\BEATs_iter3_plus_AS2M.pt encoder checkpoint'i indir
          3. train_beats.py çalıştır → D:\\models\\beats_mlp.pt üretilir
        """
        CLASSES = _BEATS_CLASSES

        def __init__(self, n_classes: int = 6, classes=None):
            super().__init__()
            # Encoder placeholder — _load_encoder() ile doldurulur
            self.encoder     = None
            self.n_classes   = n_classes
            # Eski checkpointler 6 sınıflı olabilir; etiket sırası checkpointten gelir.
            self.classes     = list(classes) if classes else list(_BEATS_CLASSES[:n_classes])
            self.mlp = nn.Sequential(
                nn.Linear(_BEATS_EMBED_DIM, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, n_classes),
            )

        def _load_encoder(self, encoder_path: str):
            """BEATs encoder'ı yükle ve dondur."""
            if not BEATS_OK:
                raise ImportError(
                    "BEATs modülü bulunamadı. "
                    "microsoft/unilm reposundan BEATs/ klasörünü proje köküne kopyalayın."
                )
            
            # 1. Önce checkpoint'i hafızaya alıyoruz (Çünkü config bunun içinde saklı)
            ckpt = torch.load(encoder_path, map_location=_TORCH_DEVICE, weights_only=False)
            
            # 2. Boş BEATsConfig() yerine checkpoint içindeki orijinal konfigürasyonu veriyoruz
            # Böylece input_patch_size=-1 gelmiyor ve Conv2d katmanları düzgün boyutlanıyor
            cfg = BEATsConfig(ckpt["cfg"])
            enc = BEATs(cfg)
            
            # 3. Model ağırlıklarını güvenle yüklüyoruz
            state = ckpt.get("model", ckpt)
            enc.load_state_dict(state)
            
            # 4. Parametreleri dondurma ve değerlendirme modu
            enc.eval()
            for p in enc.parameters():
                p.requires_grad = False
                
            self.encoder = enc.to(_TORCH_DEVICE)

        def forward(self, waveform: "torch.Tensor") -> "torch.Tensor":
            """
            waveform : (B, T) float32, SR=22050
            Döndürür : (B, n_classes) logit
            """
            # padding_mask: tüm zaman adımları geçerli (False = geçerli)
            padding_mask = torch.zeros(
                waveform.shape[0], waveform.shape[1],
                dtype=torch.bool, device=waveform.device
            )
            with torch.no_grad():
                features, _ = self.encoder.extract_features(
                    waveform, padding_mask=padding_mask
                )                             # (B, T', 768)
            embedding = features.mean(dim=1)  # (B, 768) — temporal mean pooling
            return self.mlp(embedding)        # (B, n_classes)

        def infer(self, chunk: np.ndarray) -> tuple:
            """
            chunk: float32 numpy, SR=22050, 5s (110250 örnek)
            Döndürür: (label: str, probs_dict: dict[str, float])
            """
            with torch.no_grad():
                y_t    = torch.FloatTensor(chunk).unsqueeze(0).to(_TORCH_DEVICE)
                logits = self.forward(y_t)
                probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            label = self.classes[int(np.argmax(probs))]
            return label, {cls: float(p) for cls, p in zip(self.classes, probs)}


# ══════════════════════════════════════════════════════════════
#  EnsembleClassifier  (v5.1)
#  EfficientNet + BEATs softmax ortalaması
# ══════════════════════════════════════════════════════════════

if TORCH_OK:
    class EnsembleClassifier:
        """
        EfficientNet-B0 ve BEATs çıktılarını ağırlıklı olarak birleştirir.

        final_probs = α * eff_probs + (1 - α) * beats_probs

        Sınıf sırası her iki modelde de aynı olmalıdır.
        α = _ENSEMBLE_ALPHA (varsayılan 0.5); validation set üzerinde tune edilebilir.
        """
        def __init__(self, eff_system: "AirportNoiseSystem",
                     beats_classifier: "BEATsClassifier",
                     alpha: float = _ENSEMBLE_ALPHA):
            self.eff     = eff_system
            self.beats   = beats_classifier
            self.alpha   = alpha
            self.classes = list(beats_classifier.classes)

        def infer(self, chunk: np.ndarray,
                  apply_prior: bool = True) -> tuple:
            """
            chunk: float32 numpy, SR=22050, 5s
            Döndürür: (label: str, probs_dict: dict[str, float])

            EfficientNet: prior düzeltmeli olasılıklar (mevcut davranış korunur)
            BEATs: ham softmax (prior uygulanmaz — embedding zaten daha nötr)
            Ensemble: α * eff_adj + (1-α) * beats_raw → tekrar normalize
            """
            # ── EfficientNet tarafı ───────────────────────────────
            eff_label, eff_probs_dict = self.eff._infer_efficientnet_chunk(chunk)
            eff_probs = np.array(
                [eff_probs_dict.get(cls, 0.0) for cls in self.classes],
                dtype=np.float32
            )

            # ── BEATs tarafı ──────────────────────────────────────
            _, beats_probs_dict = self.beats.infer(chunk)
            beats_probs = np.array(
                [beats_probs_dict.get(cls, 0.0) for cls in self.classes],
                dtype=np.float32
            )

            # ── Ağırlıklı ortalama ────────────────────────────────
            combined = self.alpha * eff_probs + (1.0 - self.alpha) * beats_probs
            total = combined.sum()
            if total > 1e-9:
                combined /= total

            label = self.classes[int(np.argmax(combined))]
            probs_dict = {cls: float(p) for cls, p in zip(self.classes, combined)}
            return label, probs_dict


def extract_features_ml(y: np.ndarray, sr: int = _ML_SR) -> np.ndarray:
    """
    264 boyutlu ML özellik vektörü.
    dataset_builder_v2.py::extract_features() ile birebir aynı.
    DEĞİŞTİRME — modeli yeniden eğitmek gerekir.
    """
    features = []

    mfcc    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=_ML_N_MFCC,
                                    n_fft=_ML_N_FFT, hop_length=_ML_HOP_FFT)
    d_mfcc  = librosa.feature.delta(mfcc)
    d2_mfcc = librosa.feature.delta(mfcc, order=2)
    for m in [mfcc, d_mfcc, d2_mfcc]:
        features.extend([m.mean(axis=1), m.std(axis=1)])

    chroma = librosa.feature.chroma_stft(y=y, sr=sr,
                                          n_fft=_ML_N_FFT, hop_length=_ML_HOP_FFT)
    features.append(chroma.mean(axis=1))

    sc  = librosa.feature.spectral_centroid(y=y, sr=sr,
                                             n_fft=_ML_N_FFT, hop_length=_ML_HOP_FFT)
    sb  = librosa.feature.spectral_bandwidth(y=y, sr=sr,
                                              n_fft=_ML_N_FFT, hop_length=_ML_HOP_FFT)
    sr_ = librosa.feature.spectral_rolloff(y=y, sr=sr,
                                            n_fft=_ML_N_FFT, hop_length=_ML_HOP_FFT)
    sf  = librosa.feature.spectral_flatness(y=y,
                                             n_fft=_ML_N_FFT, hop_length=_ML_HOP_FFT)
    for feat in [sc, sb, sr_, sf]:
        features.extend([feat.mean(axis=1), feat.std(axis=1)])

    zcr = librosa.feature.zero_crossing_rate(y, hop_length=_ML_HOP_FFT)
    rms = librosa.feature.rms(y=y, hop_length=_ML_HOP_FFT)
    for feat in [zcr, rms]:
        features.extend([feat.mean(axis=1), feat.std(axis=1)])

    return np.concatenate(features, axis=0).astype(np.float32)


def _load_and_chunk_ml(wav_path: str):
    """
    Ses dosyasını 5 sn'lik pencerelere böl (%50 overlap).
    dataset_builder_v2.py::load_and_chunk() ile aynı mantık.
    """
    y, _ = librosa.load(wav_path, sr=_ML_SR, mono=True)
    clip_samples = int(_ML_CLIP * _ML_SR)
    hop_samples  = int(_ML_HOP_SEC * _ML_SR)

    if len(y) <= clip_samples:
        padded = np.zeros(clip_samples, dtype=np.float32)
        padded[:len(y)] = y
        return [padded], y

    chunks = []
    start = 0
    while start + clip_samples <= len(y):
        chunks.append(y[start:start + clip_samples])
        start += hop_samples
    return chunks, y


# ══════════════════════════════════════════════════════════════
#  YARDIMCI: saf-Python WAV okuyucu (librosa yok durumunda)
# ══════════════════════════════════════════════════════════════

def _read_wav(path: str):
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
    SUPPORTED = {".wav"}

    def __init__(self, target_sr: int = 22050):
        self.target_sr = target_sr

    def load(self, path: str):
        import librosa
        samples, sr = librosa.load(path, sr=self.target_sr, mono=True)
        duration = len(samples) / sr
        print(f"[AudioLoader] '{Path(path).name}' yüklendi | Süre: {duration:.2f}s | SR: {sr} Hz")
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
            nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length
        )
        S_db = 20 * np.log10(np.abs(Zxx) + 1e-10)
        return freqs, times, S_db

    def mel_spectrogram(self, samples, n_mels=128):
        freqs, times, Zxx = signal.stft(
            samples, fs=self.sr, window="hann",
            nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length
        )
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
    def __init__(self, sr=22050, n_fft=2048, hop_length=512, n_mfcc=13):
        self.sr         = sr
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.n_mfcc     = n_mfcc

    def extract_all(self, samples):
        features = {}
        freqs, times_stft, Zxx = signal.stft(
            samples, fs=self.sr, window="hann",
            nperseg=self.n_fft, noverlap=self.n_fft - self.hop_length
        )
        magnitude = np.abs(Zxx)
        power     = magnitude ** 2
        times     = times_stft

        sc = np.sum(freqs[:, None] * magnitude, axis=0) / (
            np.sum(magnitude, axis=0) + 1e-10)
        features["spectral_centroid"] = (times, sc)

        sb = np.sqrt(
            np.sum(((freqs[:, None] - sc[None, :]) ** 2) * magnitude, axis=0) /
            (np.sum(magnitude, axis=0) + 1e-10)
        )
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

        mfcc = self._mfcc(samples)
        features["mfcc"] = mfcc
        return features

    def _mfcc(self, samples):
        analyzer = AudioAnalyzer(self.n_fft, self.hop_length, self.sr)
        _, _, mel_db = analyzer.mel_spectrogram(samples, n_mels=40)
        n_mels, n_frames = mel_db.shape
        mfcc = np.zeros((self.n_mfcc, n_frames))
        for m in range(self.n_mfcc):
            mfcc[m] = np.sum(
                mel_db * np.cos(np.pi * m / n_mels *
                                (np.arange(n_mels)[:, None] + 0.5)), axis=0
            )
        return mfcc

    def feature_summary(self, features):
        summary = {}
        for name, val in features.items():
            arr = val if name == "mfcc" else val[1]
            summary[name] = {
                "mean": float(np.mean(arr)), "std": float(np.std(arr)),
                "min":  float(np.min(arr)),  "max": float(np.max(arr))
            }
        return summary


# ═══════════════════════════════════════════════════════
#  MODÜL 4 – NoiseFilter
# ═══════════════════════════════════════════════════════

class NoiseFilter:
    def __init__(self, sr=22050):
        self.sr = sr

    def bandpass_filter(self, samples, lowcut, highcut, order=5):
        nyq  = self.sr / 2
        low  = np.clip(lowcut  / nyq, 1e-6, 0.9999)
        high = np.clip(highcut / nyq, 1e-6, 0.9999)
        b, a = signal.butter(order, [low, high], btype="band")
        filtered = signal.filtfilt(b, a, samples)
        print(f"[NoiseFilter] Band-pass uygulandı: {lowcut}–{highcut} Hz")
        return filtered.astype(np.float32)

    def spectral_gating(self, samples, n_fft=2048, hop_length=512,
                        prop_decrease=0.9, n_std_thresh=1.5,
                        noise_clip_fraction=0.1):
        n_noise    = max(n_fft, int(len(samples) * noise_clip_fraction))
        noise_clip = samples[:n_noise]
        _, _, noise_stft = signal.stft(noise_clip, fs=self.sr, window="hann",
                                        nperseg=n_fft, noverlap=n_fft - hop_length)
        noise_power = np.mean(np.abs(noise_stft) ** 2, axis=1, keepdims=True)
        noise_std   = np.std(np.abs(noise_stft),       axis=1, keepdims=True)

        _, times_out, sig_stft = signal.stft(samples, fs=self.sr, window="hann",
                                              nperseg=n_fft, noverlap=n_fft - hop_length)
        sig_magnitude = np.abs(sig_stft)
        sig_phase     = np.angle(sig_stft)
        noise_thresh  = np.sqrt(noise_power) + n_std_thresh * noise_std
        mask          = sig_magnitude > noise_thresh
        mask_smoothed = np.clip(
            (sig_magnitude - noise_thresh) / (noise_thresh + 1e-10), 0, 1)
        mask_final    = mask * (1 - prop_decrease) + mask_smoothed * prop_decrease
        sig_filtered_stft = mask_final * sig_magnitude * np.exp(1j * sig_phase)
        _, filtered = signal.istft(sig_filtered_stft, fs=self.sr, window="hann",
                                    nperseg=n_fft, noverlap=n_fft - hop_length)
        filtered = filtered[:len(samples)].astype(np.float32)
        print(f"[NoiseFilter] Spectral gating tamamlandı")
        return filtered

    def wiener_filter(self, samples, n_fft=2048, hop_length=512,
                      noise_power_estimate_frames=10):
        _, _, stft = signal.stft(samples, fs=self.sr, window="hann",
                                  nperseg=n_fft, noverlap=n_fft - hop_length)
        S        = np.abs(stft) ** 2
        N        = np.mean(S[:, :noise_power_estimate_frames], axis=1, keepdims=True)
        N        = np.maximum(N, 1e-10)
        gain     = np.maximum((S - N) / S, 0.0)
        gain_db  = np.sqrt(gain)
        _, filtered = signal.istft(stft * gain_db, fs=self.sr, window="hann",
                                    nperseg=n_fft, noverlap=n_fft - hop_length)
        return filtered[:len(samples)].astype(np.float32)


# ═══════════════════════════════════════════════════════
#  MODÜL 5 – NoiseClassifier  (kural tabanlı — yedek)
# ═══════════════════════════════════════════════════════

class NoiseClassifier:
    """
    Kural tabanlı yedek sınıflandırıcı — hiçbir ML modeli yüklenemediğinde
    devreye girer (bkz. AirportNoiseSystem.run/analyze_for_gui).

    NOT: Bu basit spektral eşikler sadece 4 kaba kategoriyi ayırt edebilir.
    Yeni 10-sınıf taksonomisindeki ince ayrımları (ör. HELICOPTER vs
    APU_GSE, ya da PRECIPITATION/NATURE/SIREN_ALARM) gerçek veri olmadan
    güvenilir biçimde ayırt edecek yeni eşikler UYDURULMADI — bu yüzden
    yedek sınıflandırıcı bilinçli olarak dar kapsamlı bırakıldı:
    JET_AIRCRAFT/WIND/TRAFFIC/SPEECH/UNKNOWN dışındaki sınıfları hiç
    üretmez. Eski "AIRCRAFT" eşiği (geniş bant + orta-yüksek enerji)
    açıklamasına en yakın yeni sınıf olan JET_AIRCRAFT'a yeniden
    etiketlendi; eşik DEĞERLERİ değişmedi.
    """
    LABELS = ["JET_AIRCRAFT", "WIND", "TRAFFIC", "SPEECH", "UNKNOWN"]

    THRESHOLDS = {
        "jet_aircraft_sc_min":  400,
        "jet_aircraft_sc_max":  2000,
        "jet_aircraft_rms_min": 0.05,
        "wind_sc_max":      500,
        "wind_rms_max":     0.10,
        "traffic_sc_min":   200,
        "traffic_sc_max":   1500,
        "speech_sc_min":    500,
        "speech_sc_max":    3500,
        "speech_zcr_min":   0.05,
    }

    def classify_frame(self, sc, zcr, rms, sr=22050):
        th = self.THRESHOLDS
        if th["jet_aircraft_sc_min"] < sc < th["jet_aircraft_sc_max"] and rms > th["jet_aircraft_rms_min"]:
            return "JET_AIRCRAFT"
        if sc < th["wind_sc_max"] and rms < th["wind_rms_max"]:
            return "WIND"
        if th["speech_sc_min"] < sc < th["speech_sc_max"] and zcr > th["speech_zcr_min"]:
            return "SPEECH"
        if th["traffic_sc_min"] < sc < th["traffic_sc_max"]:
            return "TRAFFIC"
        return "UNKNOWN"

    def classify(self, features):
        times_sc, sc   = features["spectral_centroid"]
        times_zcr, zcr = features["zcr"]
        times_rms, rms = features["rms"]

        n        = min(len(sc), len(zcr), len(rms))
        sc       = sc[:n];  zcr = zcr[:n];  rms = rms[:n]
        t        = times_sc[:n]
        rms_norm = rms / (np.max(rms) + 1e-10)

        labels = [self.classify_frame(sc[i], zcr[i], rms_norm[i]) for i in range(n)]

        counts  = Counter(labels)
        total   = len(labels)
        summary = {k: round(100 * v / total, 1) for k, v in counts.items()}
        print(f"[NoiseClassifier] Kural tabanlı ({n} çerçeve): {summary}")
        return labels, t, summary


# ═══════════════════════════════════════════════════════
#  MODÜL 6 – Visualizer
# ═══════════════════════════════════════════════════════

class Visualizer:
    LABEL_COLORS = _SHARED_CLASS_COLORS   # class_config.CLASS_COLORS

    def __init__(self, output_dir="outputs"):
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)

    def plot_full_dashboard(self, samples, sr, features,
                            filtered_bp, filtered_sg,
                            frame_labels, label_times,
                            cls_summary=None,
                            filename_prefix="airport_noise"):
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(22, 18))
        fig.patch.set_facecolor("#0D1117")

        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
        ax_wf   = fig.add_subplot(gs[0, 0])
        ax_db   = fig.add_subplot(gs[0, 1])
        ax_cls  = fig.add_subplot(gs[0, 2])
        ax_spec = fig.add_subplot(gs[1, 0])
        ax_mel  = fig.add_subplot(gs[1, 1])
        ax_sc   = fig.add_subplot(gs[1, 2])
        ax_filt = fig.add_subplot(gs[2, 0])
        ax_mfcc = fig.add_subplot(gs[2, 1])
        ax_zcr  = fig.add_subplot(gs[2, 2])

        ACCENT = "#00D4FF"; ACCENT2 = "#FF6B35"; BG = "#161B22"; GRID = "#21262D"
        for ax in [ax_wf, ax_db, ax_cls, ax_spec, ax_mel, ax_sc, ax_filt, ax_mfcc, ax_zcr]:
            ax.set_facecolor(BG)
            ax.tick_params(colors="#8B949E", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#30363D")

        analyzer = AudioAnalyzer(sr=sr)
        t, s = analyzer.waveform(samples)

        # 1. Waveform
        ax_wf.plot(t, s, color=ACCENT, lw=0.5, alpha=0.85)
        ax_wf.fill_between(t, s, 0, color=ACCENT, alpha=0.1)
        ax_wf.set_title("Waveform", color="white", fontsize=10, pad=6)
        ax_wf.set_xlabel("Zaman (s)", color="#8B949E", fontsize=8)
        ax_wf.set_ylabel("Genlik", color="#8B949E", fontsize=8)
        ax_wf.grid(True, color=GRID, linewidth=0.5)

        # 2. dB
        t_db, db = analyzer.db_over_time(samples)
        ax_db.plot(t_db, db, color=ACCENT2, lw=1.2)
        ax_db.fill_between(t_db, db, np.min(db) - 5, color=ACCENT2, alpha=0.15)
        ax_db.set_title("Ses Seviyesi (dBFS)", color="white", fontsize=10, pad=6)
        ax_db.set_xlabel("Zaman (s)", color="#8B949E", fontsize=8)
        ax_db.set_ylabel("dBFS", color="#8B949E", fontsize=8)
        ax_db.grid(True, color=GRID, linewidth=0.5)

        # 3. Sınıflandırma zaman serisi
        # NOT: eskiden burası NoiseClassifier.LABELS (sadece 5 kaba kural-
        # tabanlı etiket) kullanıyordu; ML modellerinin ürettiği diğer
        # etiketler (ör. eskiden AMBIENT/OTHER, şimdi HELICOPTER, NATURE,
        # SIREN_ALARM vb.) fallback index'i UNKNOWN satırıyla çakıştığı
        # için sessizce yanlış satıra çiziliyordu. Artık tam sınıf
        # kümesi + UNKNOWN kullanılıyor.
        all_labels = _SHARED_CLASSES + ["UNKNOWN"]
        label_map  = {l: i for i, l in enumerate(all_labels)}
        y_cls      = np.array([label_map.get(l, len(all_labels) - 1) for l in frame_labels])
        colors_cls = [self.LABEL_COLORS.get(l, "#6C757D") for l in frame_labels]
        w = label_times[1] - label_times[0] if len(label_times) > 1 else 0.5
        ax_cls.bar(label_times, y_cls + 1, width=w, color=colors_cls, alpha=0.85)
        ax_cls.set_yticks(range(1, len(all_labels) + 1))
        ax_cls.set_yticklabels(all_labels, fontsize=7, color="#8B949E")
        ax_cls.set_title("Ses Sınıflandırması (ML)", color="white", fontsize=10, pad=6)
        ax_cls.set_xlabel("Zaman (s)", color="#8B949E", fontsize=8)
        ax_cls.grid(True, axis="x", color=GRID, linewidth=0.5)
        handles = [plt.Rectangle((0, 0), 1, 1, fc=self.LABEL_COLORS.get(lb, "#6C757D"))
                   for lb in all_labels]
        ax_cls.legend(handles, all_labels, loc="upper right",
                      fontsize=6, facecolor=BG, labelcolor="white")

        # 4. Spektrogram
        f_s, t_s, S_db = analyzer.spectrogram(samples)
        im = ax_spec.pcolormesh(t_s, f_s, S_db, shading="auto", cmap="inferno",
                                vmin=np.percentile(S_db, 10))
        ax_spec.set_ylim(0, min(8000, sr / 2))
        ax_spec.set_title("Spektrogram (STFT)", color="white", fontsize=10, pad=6)
        ax_spec.set_xlabel("Zaman (s)", color="#8B949E", fontsize=8)
        ax_spec.set_ylabel("Frekans (Hz)", color="#8B949E", fontsize=8)
        plt.colorbar(im, ax=ax_spec, label="dB").ax.yaxis.label.set_color("#8B949E")

        # 5. Mel Spektrogram
        _, _, mel_db = analyzer.mel_spectrogram(samples, n_mels=64)
        im2 = ax_mel.pcolormesh(
            np.linspace(0, len(samples) / sr, mel_db.shape[1]),
            np.arange(mel_db.shape[0]), mel_db, shading="auto", cmap="magma")
        ax_mel.set_title("Mel Spektrogram", color="white", fontsize=10, pad=6)
        ax_mel.set_xlabel("Zaman (s)", color="#8B949E", fontsize=8)
        ax_mel.set_ylabel("Mel Kanalı", color="#8B949E", fontsize=8)
        plt.colorbar(im2, ax=ax_mel, label="dB").ax.yaxis.label.set_color("#8B949E")

        # 6. Spectral Centroid
        t_sc, sc = features["spectral_centroid"]
        ax_sc.plot(t_sc, sc, color="#7EE8A2", lw=1.0)
        ax_sc.set_title("Spectral Centroid (Hz)", color="white", fontsize=10, pad=6)
        ax_sc.set_xlabel("Zaman (s)", color="#8B949E", fontsize=8)
        ax_sc.set_ylabel("Hz", color="#8B949E", fontsize=8)
        ax_sc.grid(True, color=GRID, linewidth=0.5)

        # 7. Filtre karşılaştırma
        ax_filt.plot(t, samples, color=ACCENT, lw=0.5, alpha=0.5, label="Orijinal")
        t_bp = np.linspace(0, len(filtered_bp) / sr, len(filtered_bp))
        ax_filt.plot(t_bp, filtered_bp, color=ACCENT2, lw=0.8, alpha=0.75, label="Band-pass")
        t_sg = np.linspace(0, len(filtered_sg) / sr, len(filtered_sg))
        ax_filt.plot(t_sg, filtered_sg, color="#7EE8A2", lw=0.8, alpha=0.75, label="Sp. Gating")
        ax_filt.set_title("Filtre Karşılaştırması", color="white", fontsize=10, pad=6)
        ax_filt.set_xlabel("Zaman (s)", color="#8B949E", fontsize=8)
        ax_filt.legend(fontsize=7, facecolor=BG, labelcolor="white")
        ax_filt.grid(True, color=GRID, linewidth=0.5)

        # 8. MFCC
        mfcc = features["mfcc"]
        im3  = ax_mfcc.imshow(mfcc, aspect="auto", origin="lower", cmap="coolwarm",
                               extent=[0, len(samples) / sr, 0, mfcc.shape[0]])
        ax_mfcc.set_title("MFCC (13 Katsayı)", color="white", fontsize=10, pad=6)
        ax_mfcc.set_xlabel("Zaman (s)", color="#8B949E", fontsize=8)
        ax_mfcc.set_ylabel("MFCC Katsayısı", color="#8B949E", fontsize=8)
        plt.colorbar(im3, ax=ax_mfcc).ax.yaxis.label.set_color("#8B949E")

        # 9. ZCR & RMS
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
        ax_zcr2.tick_params(colors="#8B949E", labelsize=8)
        lines1, labs1 = ax_zcr.get_legend_handles_labels()
        lines2, labs2 = ax_zcr2.get_legend_handles_labels()
        ax_zcr.legend(lines1 + lines2, labs1 + labs2,
                      fontsize=7, facecolor=BG, labelcolor="white")

        # ML özet — sağ üst köşe metni
        if cls_summary:
            summary_str = "  ".join(f"{k}: {v:.1f}%" for k, v in
                                    sorted(cls_summary.items(), key=lambda x: -x[1]))
            fig.text(0.5, 0.995, f"ML Dağılım → {summary_str}",
                     ha="center", va="top", color="#7EE8A2", fontsize=9)

        fig.suptitle("✈  HAVALIMANL ÇEVRESEL GÜRÜLTÜ ANALİZ PANOSU  ✈",
                     color="white", fontsize=15, fontweight="bold", y=0.98)

        path = self.out / f"{filename_prefix}_dashboard.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"[Visualizer] Dashboard kaydedildi → {path}")
        return str(path)

    def plot_waveform(self, samples, sr, filename_prefix="airport_noise"):
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor("#0D1117"); ax.set_facecolor("#161B22")
        t = np.linspace(0, len(samples) / sr, len(samples))
        ax.plot(t, samples, color="#00D4FF", lw=0.5, alpha=0.85)
        ax.fill_between(t, samples, 0, color="#00D4FF", alpha=0.08)
        ax.set_title("Waveform", color="white", fontsize=13)
        ax.set_xlabel("Zaman (s)", color="#8B949E"); ax.set_ylabel("Genlik", color="#8B949E")
        ax.tick_params(colors="#8B949E"); ax.grid(True, color="#21262D", linewidth=0.5)
        for spine in ax.spines.values(): spine.set_edgecolor("#30363D")
        path = self.out / f"{filename_prefix}_1_waveform.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig); print(f"[Visualizer] Waveform → {path}"); return str(path)

    def plot_db_over_time(self, samples, sr, filename_prefix="airport_noise"):
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor("#0D1117"); ax.set_facecolor("#161B22")
        analyzer = AudioAnalyzer(sr=sr); t_db, db = analyzer.db_over_time(samples)
        ax.plot(t_db, db, color="#FF6B35", lw=1.2)
        ax.fill_between(t_db, db, np.min(db) - 5, color="#FF6B35", alpha=0.15)
        ax.set_title("Ses Seviyesi (dBFS)", color="white", fontsize=13)
        ax.set_xlabel("Zaman (s)", color="#8B949E"); ax.set_ylabel("dBFS", color="#8B949E")
        ax.tick_params(colors="#8B949E"); ax.grid(True, color="#21262D", linewidth=0.5)
        for spine in ax.spines.values(): spine.set_edgecolor("#30363D")
        path = self.out / f"{filename_prefix}_2_db_over_time.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig); print(f"[Visualizer] dB/Zaman → {path}"); return str(path)

    def plot_spectrogram(self, samples, sr, filename_prefix="airport_noise"):
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#0D1117"); ax.set_facecolor("#161B22")
        analyzer = AudioAnalyzer(sr=sr); f, t, S_db = analyzer.spectrogram(samples)
        im = ax.pcolormesh(t, f, S_db, shading="auto", cmap="inferno",
                           vmin=np.percentile(S_db, 10))
        ax.set_ylim(0, min(8000, sr / 2))
        ax.set_title("Spektrogram (STFT)", color="white", fontsize=13)
        ax.set_xlabel("Zaman (s)", color="#8B949E"); ax.set_ylabel("Frekans (Hz)", color="#8B949E")
        ax.tick_params(colors="#8B949E")
        for spine in ax.spines.values(): spine.set_edgecolor("#30363D")
        cb = plt.colorbar(im, ax=ax); cb.set_label("dB", color="#8B949E")
        cb.ax.yaxis.set_tick_params(color="#8B949E")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8B949E")
        path = self.out / f"{filename_prefix}_3_spectrogram.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig); print(f"[Visualizer] Spektrogram → {path}"); return str(path)

    def plot_mel_spectrogram(self, samples, sr, filename_prefix="airport_noise"):
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#0D1117"); ax.set_facecolor("#161B22")
        analyzer = AudioAnalyzer(sr=sr); _, _, mel_db = analyzer.mel_spectrogram(samples, n_mels=64)
        im = ax.imshow(mel_db, aspect="auto", origin="lower", cmap="magma",
                       extent=[0, len(samples) / sr, 0, mel_db.shape[0]])
        ax.set_title("Mel Spektrogram", color="white", fontsize=13)
        ax.set_xlabel("Zaman (s)", color="#8B949E"); ax.set_ylabel("Mel Kanalı", color="#8B949E")
        ax.tick_params(colors="#8B949E")
        for spine in ax.spines.values(): spine.set_edgecolor("#30363D")
        cb = plt.colorbar(im, ax=ax); cb.set_label("dB", color="#8B949E")
        cb.ax.yaxis.set_tick_params(color="#8B949E")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8B949E")
        path = self.out / f"{filename_prefix}_4_mel_spectrogram.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig); print(f"[Visualizer] Mel Spektrogram → {path}"); return str(path)

    def plot_mfcc(self, features, samples, sr, filename_prefix="airport_noise"):
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor("#0D1117"); ax.set_facecolor("#161B22")
        mfcc = features["mfcc"]
        im = ax.imshow(mfcc, aspect="auto", origin="lower", cmap="coolwarm",
                       extent=[0, len(samples) / sr, 0, mfcc.shape[0]])
        ax.set_title("MFCC (13 Katsayı)", color="white", fontsize=13)
        ax.set_xlabel("Zaman (s)", color="#8B949E"); ax.set_ylabel("MFCC Katsayısı", color="#8B949E")
        ax.tick_params(colors="#8B949E")
        for spine in ax.spines.values(): spine.set_edgecolor("#30363D")
        cb = plt.colorbar(im, ax=ax)
        cb.ax.yaxis.set_tick_params(color="#8B949E")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8B949E")
        path = self.out / f"{filename_prefix}_5_mfcc.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig); print(f"[Visualizer] MFCC → {path}"); return str(path)

    def plot_features(self, features, filename_prefix="airport_noise"):
        plt.style.use("dark_background")
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)
        fig.patch.set_facecolor("#0D1117")
        fig.suptitle("Spektral Özellikler", color="white", fontsize=14)
        datasets = [
            ("spectral_centroid", "Spectral Centroid (Hz)", "#7EE8A2"),
            ("zcr",               "Zero Crossing Rate",     "#FFE66D"),
            ("rms",               "RMS Enerji",             "#FF6B9D"),
        ]
        for ax, (key, title, color) in zip(axes, datasets):
            t, vals = features[key]
            ax.set_facecolor("#161B22")
            ax.plot(t, vals, color=color, lw=1.0)
            ax.fill_between(t, vals, alpha=0.12, color=color)
            ax.set_title(title, color="white", fontsize=11)
            ax.set_xlabel("Zaman (s)", color="#8B949E", fontsize=9)
            ax.tick_params(colors="#8B949E")
            ax.grid(True, color="#21262D", linewidth=0.5)
            for spine in ax.spines.values(): spine.set_edgecolor("#30363D")
        plt.tight_layout()
        path = self.out / f"{filename_prefix}_6_features.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig); print(f"[Visualizer] Özellikler → {path}"); return str(path)

    def plot_filter_comparison(self, samples, sr, filtered_bp, filtered_sg,
                               filename_prefix="airport_noise"):
        plt.style.use("dark_background")
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.patch.set_facecolor("#0D1117")
        fig.suptitle("Filtre Karşılaştırması", color="white", fontsize=14)
        signals = [
            (samples,     "Orijinal Sinyal",         "#00D4FF"),
            (filtered_bp, "Band-pass Filtreli",       "#FF6B35"),
            (filtered_sg, "Spectral Gating Filtreli", "#7EE8A2"),
        ]
        for ax, (sig, title, color) in zip(axes, signals):
            t = np.linspace(0, len(sig) / sr, len(sig))
            ax.set_facecolor("#161B22")
            ax.plot(t, sig, color=color, lw=0.6, alpha=0.9)
            ax.fill_between(t, sig, 0, color=color, alpha=0.07)
            ax.set_title(title, color="white", fontsize=11)
            ax.set_ylabel("Genlik", color="#8B949E", fontsize=9)
            ax.tick_params(colors="#8B949E")
            ax.grid(True, color="#21262D", linewidth=0.5)
            for spine in ax.spines.values(): spine.set_edgecolor("#30363D")
        axes[-1].set_xlabel("Zaman (s)", color="#8B949E")
        plt.tight_layout()
        path = self.out / f"{filename_prefix}_7_filter_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig); print(f"[Visualizer] Filtre karşılaştırması → {path}"); return str(path)

    def plot_classification_pie(self, summary, filename_prefix="airport_noise"):
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.patch.set_facecolor("#0D1117"); ax.set_facecolor("#0D1117")
        labels  = list(summary.keys())
        sizes   = list(summary.values())
        colors  = [self.LABEL_COLORS.get(l, "#6C757D") for l in labels]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140,
            textprops={"color": "white", "fontsize": 11},
            wedgeprops={"edgecolor": "#0D1117", "linewidth": 2}
        )
        for at in autotexts: at.set_fontsize(10)
        ax.set_title("Gürültü Kaynağı Dağılımı (ML)", color="white", fontsize=14, pad=15)
        path = self.out / f"{filename_prefix}_classification_pie.png"
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig); print(f"[Visualizer] Pasta grafiği → {path}"); return str(path)


# ═══════════════════════════════════════════════════════
#  MODÜL 7 – AirportNoiseSystem  (ML entegreli orkestratör)
# ═══════════════════════════════════════════════════════

class AirportNoiseSystem:
    """
    Tüm modülleri bir araya getiren ana sınıf.
    ML modeli varsa kullanır, yoksa kural tabanlıya düşer.

    Kullanım:
        system = AirportNoiseSystem()
        results = system.run("ses.wav")
    """

    # models/ klasörü — main.py ile aynı dizinde aranır
    _MODELS_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models"
    )

    def __init__(self, target_sr=22050, output_dir="outputs"):
        self.loader     = AudioLoader(target_sr)
        self.analyzer   = AudioAnalyzer(sr=target_sr)
        self.extractor  = FeatureExtractor(sr=target_sr)
        self.filter     = NoiseFilter(sr=target_sr)
        self.classifier = NoiseClassifier()   # yedek
        self.visualizer = Visualizer(output_dir)
        self.sr         = target_sr
        self.output_dir = output_dir

        # ── SVM modeli yükle ─────────────────────────────────
        self.ml_model = None
        self.ml_le    = None
        self._load_ml_model()

        # ── CNN modeli yükle ─────────────────────────────────
        self.cnn_model = None
        self.cnn_le    = None
        self._load_cnn_model()

        # ── EfficientNet modeli yükle ─────────────────────────
        self.eff_model = None
        self.eff_le    = None
        self._load_efficientnet_model()

        # ── BEATs modeli yükle (v5.1) ─────────────────────────
        self.beats_model    = None   # BEATsClassifier örneği
        self.ensemble_model = None   # EnsembleClassifier örneği
        self._load_beats_model()

    def _load_ml_model(self):
        """models/best_model.pkl ve label_encoder.pkl yükle."""
        if not (LIBROSA_OK and JOBLIB_OK):
            print("[ML] librosa veya joblib eksik — kural tabanlı kullanılacak")
            return

        model_path = os.path.join(self._MODELS_DIR, "best_model.pkl")
        le_path    = os.path.join(self._MODELS_DIR, "label_encoder.pkl")

        if not os.path.exists(model_path):
            print(f"[ML] Model bulunamadı ({model_path}) — kural tabanlı kullanılacak")
            return
        if not os.path.exists(le_path):
            print(f"[ML] LabelEncoder bulunamadı ({le_path}) — kural tabanlı kullanılacak")
            return

        try:
            self.ml_model = joblib.load(model_path)
            self.ml_le    = joblib.load(le_path)
            classes       = list(self.ml_le.classes_)
            print(f"[ML] ✅ Model yüklendi  |  Sınıflar: {classes}")
        except Exception as e:
            print(f"[ML] Model yükleme hatası: {e} — kural tabanlı kullanılacak")
            self.ml_model = None
            self.ml_le    = None

    # ── Prior ağırlıkları ─────────────────────────────────────────
    # class_config.INFERENCE_PRIOR_WEIGHTS'tan geliyor — yeni taksonomi
    # ve yeni veri seti için henüz tune EDİLMEDİ, hepsi nötr (1.0).
    # Eski değerler (AIRCRAFT: 0.25 vb.) eski eğitim setindeki %78
    # AIRCRAFT dengesizliğine göre elle tune edilmişti; o dengesizlik
    # yeni veri setinde muhtemelen farklı olacak. Yeni model eğitildikten
    # sonra confusion matrix'e bakıp burayı (class_config.py içinde)
    # tekrar tune et:
    #   < 1.0  →  o sınıfı bastır
    #   > 1.0  →  o sınıfı yükselt
    PRIOR_WEIGHTS = _SHARED_PRIOR_WEIGHTS

    def _apply_prior(self, probs: np.ndarray,
                     classes: list | None = None) -> np.ndarray:
        if classes is None:
            classes = list(self.ml_le.classes_)
        adjusted = probs.copy()
        for i, cls in enumerate(classes):
            adjusted[i] *= self.PRIOR_WEIGHTS.get(cls, 1.0)
        total = adjusted.sum()
        if total > 1e-9:
            adjusted /= total
        return adjusted

    def _load_cnn_model(self):
        """models/best_cnn.pt ve cnn_label_encoder.pkl yükle."""
        if not (TORCH_OK and JOBLIB_OK):
            print("[CNN] torch/joblib eksik — CNN devre dışı")
            return

        cnn_path = os.path.join(self._MODELS_DIR, "best_cnn.pt")
        le_path  = os.path.join(self._MODELS_DIR, "cnn_label_encoder.pkl")

        if not os.path.exists(cnn_path):
            print(f"[CNN] Model bulunamadı ({cnn_path}) — CNN devre dışı")
            return
        if not os.path.exists(le_path):
            # SVM label encoder'ı dene
            le_path = os.path.join(self._MODELS_DIR, "label_encoder.pkl")
            if not os.path.exists(le_path):
                print("[CNN] LabelEncoder bulunamadı — CNN devre dışı")
                return

        try:
            ckpt = torch.load(cnn_path, map_location=_TORCH_DEVICE, weights_only=False)
            n_classes = ckpt.get("n_classes", 5)
            self.cnn_model = AirportCNN(n_classes=n_classes)
            self.cnn_model.load_state_dict(ckpt["model_state"])
            self.cnn_model.eval()
            self.cnn_model.to(_TORCH_DEVICE)
            self.cnn_le = joblib.load(le_path)
            classes     = list(self.cnn_le.classes_)
            print(f"[CNN] ✅ Model yüklendi  |  Epoch: {ckpt.get('epoch','?')}  "
                  f"|  Sınıflar: {classes}")
        except Exception as e:
            print(f"[CNN] Model yükleme hatası: {e} — CNN devre dışı")
            self.cnn_model = None
            self.cnn_le    = None

    def _classify_cnn(self, audio_path: str):
        """
        CNN ile sınıflandırma — _classify_ml ile aynı pencere mantığı.

        Her 5 sn'lik pencere için:
          1. Mel spektrogram hesapla (128 × ~216)
          2. AirportCNN'e ver → softmax olasılıkları
          3. Prior düzeltmesi uygula
          4. Pencereleri ortala → final etiket

        Sonuç: (frame_labels, label_times, summary_dict)
        """
        chunks, _ = _load_and_chunk_ml(audio_path)
        classes   = list(self.cnn_le.classes_)

        mel_tf = T.MelSpectrogram(
            sample_rate=_ML_SR, n_fft=_ML_N_FFT,
            hop_length=_ML_HOP_FFT, n_mels=_CNN_N_MELS
        )
        db_tf = T.AmplitudeToDB(top_db=80)

        frame_labels    = []
        frame_adj_probs = []

        with torch.no_grad():
            for chunk in chunks:
                try:
                    y_t   = torch.FloatTensor(chunk).unsqueeze(0)  # (1, samples)
                    spec  = mel_tf(y_t)                            # (1, N_MELS, T)
                    spec  = db_tf(spec)
                    spec  = spec.unsqueeze(0).to(_TORCH_DEVICE)    # (1, 1, N_MELS, T)

                    logits    = self.cnn_model(spec)
                    probs     = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                    adj_probs = self._apply_prior(probs, classes)

                    pred_idx = int(np.argmax(adj_probs))
                    label    = classes[pred_idx]

                    frame_labels.append(label)
                    frame_adj_probs.append(adj_probs)

                except Exception as e:
                    print(f"  [!] CNN pencere hatası: {e}")
                    frame_labels.append("UNKNOWN")
                    frame_adj_probs.append(np.ones(len(classes)) / len(classes))

        label_times = np.array([i * _ML_HOP_SEC for i in range(len(frame_labels))])

        counts  = Counter(frame_labels)
        total   = len(frame_labels)
        summary = {k: round(100 * v / total, 1) for k, v in
                   sorted(counts.items(), key=lambda x: -x[1])}

        if frame_adj_probs:
            avg_probs = np.mean(frame_adj_probs, axis=0)
            prob_str  = "  ".join(
                f"{cls}:{p:.2f}" for cls, p in
                sorted(zip(classes, avg_probs), key=lambda x: -x[1])
            )
            print(f"[CNN] {total} pencere | Ort. olasılık → {prob_str}")
            print(f"[CNN] Dağılım: {summary}")
        else:
            print("[CNN] Hiç pencere işlenemedi")

        return frame_labels, label_times, summary

    def _load_efficientnet_model(self):
        """models/best_efficientnet.pt ve efficientnet_label_encoder.pkl yükle."""
        if not (TORCH_OK and TORCHVISION_OK and JOBLIB_OK):
            print("[EfficientNet] torch/torchvision/joblib eksik — EfficientNet devre dışı")
            return

        eff_path = os.path.join(self._MODELS_DIR, "best_efficientnet.pt")
        le_path  = os.path.join(self._MODELS_DIR, "efficientnet_label_encoder.pkl")

        if not os.path.exists(eff_path):
            print(f"[EfficientNet] Model bulunamadı ({eff_path}) — devre dışı")
            return
        if not os.path.exists(le_path):
            le_path = os.path.join(self._MODELS_DIR, "label_encoder.pkl")
            if not os.path.exists(le_path):
                print("[EfficientNet] LabelEncoder bulunamadı — devre dışı")
                return

        try:
            ckpt = torch.load(eff_path, map_location=_TORCH_DEVICE, weights_only=False)
            self.eff_le  = joblib.load(le_path)
            n_classes    = len(self.eff_le.classes_)
            self.eff_model = EfficientNetAirport(n_classes=n_classes)
            self.eff_model.load_state_dict(ckpt["model_state"])
            self.eff_model.eval()
            self.eff_model.to(_TORCH_DEVICE)
            classes = list(self.eff_le.classes_)
            phase   = ckpt.get("phase", "?")
            epoch   = ckpt.get("epoch", "?")
            f1      = ckpt.get("val_f1", "?")
            print(f"[EfficientNet] ✅ Model yüklendi  |  {phase}  Epoch:{epoch}  "
                  f"ValF1:{f1}  |  Sınıflar: {classes}")
        except Exception as e:
            print(f"[EfficientNet] Model yükleme hatası: {e} — devre dışı")
            self.eff_model = None
            self.eff_le    = None

    def _load_beats_model(self):
        """
        BEATs encoder + MLP yükle ve EnsembleClassifier oluştur.

        Dosyalar:
          _BEATS_ENCODER_PATH  → D:\\models\\BEATs_iter3_plus_AS2M.pt  (encoder)
          _BEATS_MLP_PATH      → D:\\models\\beats_mlp.pt               (MLP, train_beats.py çıktısı)

        Yükleme koşulları:
          - TORCH_OK ve BEATS_OK olmalı
          - Encoder dosyası mevcut olmalı (zorunlu)
          - MLP dosyası mevcut değilse BEATsClassifier yüklenebilir ama
            inference rastgele olur — uyarı basılır
          - EfficientNet yüklüyse EnsembleClassifier da oluşturulur
        """
        if not TORCH_OK:
            print("[BEATs] torch eksik — BEATs devre dışı")
            return

        if not BEATS_OK:
            print("[BEATs] BEATs modülü bulunamadı. "
                  "microsoft/unilm BEATs/ klasörünü proje köküne kopyalayın.")
            return

        if not os.path.exists(_BEATS_ENCODER_PATH):
            print(f"[BEATs] Encoder bulunamadı ({_BEATS_ENCODER_PATH}) — devre dışı")
            return

        try:
            if os.path.exists(_BEATS_MLP_PATH):
                ckpt = torch.load(_BEATS_MLP_PATH, map_location=_TORCH_DEVICE, weights_only=False)
                checkpoint_classes = [str(label) for label in ckpt.get("label_names", _BEATS_CLASSES)]
            else:
                ckpt = None
                checkpoint_classes = list(_BEATS_CLASSES)

            beats = BEATsClassifier(n_classes=len(checkpoint_classes), classes=checkpoint_classes)
            beats._load_encoder(_BEATS_ENCODER_PATH)

            if ckpt is not None:
                beats.mlp.load_state_dict(ckpt["model_state"])
                beats.mlp.eval()
                beats.mlp.to(_TORCH_DEVICE)
                epoch = ckpt.get("epoch", "?")
                f1    = ckpt.get("val_f1", "?")
                print(f"[BEATs] ✅ MLP yüklendi  |  Epoch:{epoch}  ValF1:{f1}  "
                      f"|  Sınıflar: {checkpoint_classes}")
            else:
                beats.mlp.to(_TORCH_DEVICE)
                print(f"[BEATs] ⚠ Encoder yüklendi ama MLP bulunamadı "
                      f"({_BEATS_MLP_PATH}). train_beats.py çalıştırın.")

            self.beats_model = beats

            # EfficientNet varsa ensemble oluştur
            if self.eff_model is not None and list(self.eff_le.classes_) == checkpoint_classes:
                self.ensemble_model = EnsembleClassifier(eff_system=self, beats_classifier=self.beats_model, alpha=_ENSEMBLE_ALPHA)
                print(f"[Ensemble] ✅ EfficientNet + BEATs  |  α={_ENSEMBLE_ALPHA}")
            elif self.eff_model is not None:
                print("[Ensemble] Sınıf sıraları farklı — ensemble devre dışı")
            else:
                print("[Ensemble] EfficientNet yüklü değil — Ensemble devre dışı")

        except Exception as e:
            print(f"[BEATs] Model yükleme hatası: {e}")
            self.beats_model    = None
            self.ensemble_model = None

    def _classify_efficientnet(self, audio_path: str):
        """
        EfficientNet-B0 ile sınıflandırma.

        train_efficientnet.py'deki MelRGBDataset.__getitem__ ile aynı
        ön-işleme adımları uygulanır:
          1. Mel Spectrogram (128 mel, n_fft=2048, hop=512)
          2. AmplitudeToDB (top_db=80)
          3. Per-sample [0,1] normalize
          4. 3 kanala kopyala (R=G=B)
          5. 224×224 resize + ImageNet normalize
          6. EfficientNet-B0'a ver → softmax → prior düzeltme

        Sonuç: (frame_labels, label_times, summary_dict, frame_probs, class_names)
        """
        import torchaudio.transforms as TA

        chunks, _ = _load_and_chunk_ml(audio_path)
        classes   = list(self.eff_le.classes_)

        mel_tf = TA.MelSpectrogram(
            sample_rate=_ML_SR, n_fft=_ML_N_FFT,
            hop_length=_ML_HOP_FFT, n_mels=_EFF_N_MELS
        )
        db_tf = TA.AmplitudeToDB(top_db=80)
        to_rgb = TV.Compose([
            TV.Resize((_EFF_IMG_SIZE, _EFF_IMG_SIZE), antialias=True),
            TV.Normalize(mean=_EFF_IMAGENET_MEAN, std=_EFF_IMAGENET_STD),
        ])

        frame_labels = []
        frame_probs  = []

        with torch.no_grad():
            for chunk in chunks:
                try:
                    y_t  = torch.FloatTensor(chunk).unsqueeze(0)   # (1, samples)
                    spec = mel_tf(y_t)                              # (1, N_MELS, T)
                    spec = db_tf(spec)

                    # DÜZELTİLMİŞ — CMVN + min-max
                    mean = spec.mean(dim=-1, keepdim=True)
                    std  = spec.std(dim=-1, keepdim=True) + 1e-8
                    spec = (spec - mean) / std

                    s_min, s_max = spec.min(), spec.max()
                    if s_max > s_min:
                        spec = (spec - s_min) / (s_max - s_min)
                    else:
                        spec = torch.zeros_like(spec)

                    spec_rgb = spec.repeat(3, 1, 1)                 # (3, N_MELS, T)
                    spec_rgb = to_rgb(spec_rgb).unsqueeze(0).to(_TORCH_DEVICE)   # (1, 3, 224, 224)
                    logits   = self.eff_model(spec_rgb)
                    probs    = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                    adj      = self._apply_prior(probs, classes)

                    frame_labels.append(classes[int(np.argmax(adj))])
                    frame_probs.append(adj)

                except Exception as e:
                    print(f"  [!] EfficientNet pencere hatası: {e}")
                    frame_labels.append("UNKNOWN")
                    frame_probs.append(np.ones(len(classes)) / len(classes))

        label_times = np.array([i * _ML_HOP_SEC for i in range(len(frame_labels))])
        counts  = Counter(frame_labels)
        total   = len(frame_labels)
        summary = {k: round(100 * v / total, 1) for k, v in
                   sorted(counts.items(), key=lambda x: -x[1])}

        if frame_probs:
            avg   = np.mean(frame_probs, axis=0)
            pstr  = "  ".join(f"{c}:{p:.2f}" for c, p in
                               sorted(zip(classes, avg), key=lambda x: -x[1]))
            print(f"[EfficientNet] {total} pencere | Ort. olasılık → {pstr}")
            print(f"[EfficientNet] Dağılım: {summary}")

        return frame_labels, label_times, summary, frame_probs, classes
    
    def _classify_beats(self, audio_path: str):
        """
        BEATs ile dosya analizi — _classify_efficientnet ile aynı arayüz.

        Her 5s pencere için BEATsClassifier.infer() çağrılır.
        Prior düzeltmesi uygulanmaz (embedding nötr davranır).

        Sonuç: (frame_labels, label_times, summary_dict, frame_probs, class_names)
        """
        chunks, _ = _load_and_chunk_ml(audio_path)
        classes   = self.beats_model.classes

        frame_labels = []
        frame_probs  = []

        for chunk in chunks:
            try:
                label, probs_dict = self.beats_model.infer(chunk)
                probs_arr = np.array(
                    [probs_dict.get(cls, 0.0) for cls in classes],
                    dtype=np.float32
                )
                frame_labels.append(label)
                frame_probs.append(probs_arr)
            except Exception as e:
                print(f"  [!] BEATs pencere hatası: {e}")
                frame_labels.append("UNKNOWN")
                frame_probs.append(np.ones(len(classes), dtype=np.float32) / len(classes))

        label_times = np.array([i * _ML_HOP_SEC for i in range(len(frame_labels))])
        counts  = Counter(frame_labels)
        total   = len(frame_labels)
        summary = {k: round(100 * v / total, 1) for k, v in
                   sorted(counts.items(), key=lambda x: -x[1])}

        if frame_probs:
            avg  = np.mean(frame_probs, axis=0)
            pstr = "  ".join(f"{c}:{p:.2f}" for c, p in
                              sorted(zip(classes, avg), key=lambda x: -x[1]))
            print(f"[BEATs] {total} pencere | Ort. olasılık → {pstr}")
            print(f"[BEATs] Dağılım: {summary}")
        else:
            print("[BEATs] Hiç pencere işlenemedi")

        return frame_labels, label_times, summary, frame_probs, list(classes)

    def _classify_ensemble(self, audio_path: str):
        """
        Ensemble (EfficientNet + BEATs) ile dosya analizi.

        Her 5s pencere için EnsembleClassifier.infer() çağrılır.
        α ağırlığı _ENSEMBLE_ALPHA sabitine göre belirlenir (varsayılan 0.5).

        Sonuç: (frame_labels, label_times, summary_dict, frame_probs, class_names)
        """
        chunks, _ = _load_and_chunk_ml(audio_path)
        classes   = self.ensemble_model.classes

        frame_labels = []
        frame_probs  = []

        for chunk in chunks:
            try:
                label, probs_dict = self.ensemble_model.infer(chunk)
                probs_arr = np.array(
                    [probs_dict.get(cls, 0.0) for cls in classes],
                    dtype=np.float32
                )
                frame_labels.append(label)
                frame_probs.append(probs_arr)
            except Exception as e:
                print(f"  [!] Ensemble pencere hatası: {e}")
                frame_labels.append("UNKNOWN")
                frame_probs.append(np.ones(len(classes), dtype=np.float32) / len(classes))

        label_times = np.array([i * _ML_HOP_SEC for i in range(len(frame_labels))])
        counts  = Counter(frame_labels)
        total   = len(frame_labels)
        summary = {k: round(100 * v / total, 1) for k, v in
                   sorted(counts.items(), key=lambda x: -x[1])}

        if frame_probs:
            avg  = np.mean(frame_probs, axis=0)
            pstr = "  ".join(f"{c}:{p:.2f}" for c, p in
                              sorted(zip(classes, avg), key=lambda x: -x[1]))
            print(f"[Ensemble] {total} pencere | Ort. olasılık → {pstr}")
            print(f"[Ensemble] Dağılım: {summary}")
        else:
            print("[Ensemble] Hiç pencere işlenemedi")

        return frame_labels, label_times, summary, frame_probs, list(classes)

    def analyze_for_gui(self, audio_path: str, model_pref: str = "auto") -> dict:
        """
        GUI için tasarlanmış analiz metodu — PNG kaydetmez, ham veri döner.

        model_pref: "auto" | "efficientnet" | "cnn" | "svm" | "rule"
          "auto" → EfficientNet > CNN > SVM > kural sıralamasıyla kullanır.

        Döndürür:
          samples, sr, duration, frame_labels, frame_times,
          frame_probs, class_names, db_times, db_values,
          mel_db, zcr, rms, sc, summary, model_used
        """
        # ── Ses yükle ─────────────────────────────────────────────
        samples, sr = self.loader.load(audio_path)
        duration    = len(samples) / sr

        # ── Özellikler ────────────────────────────────────────────
        analyzer = AudioAnalyzer(sr=sr)
        db_times, db_values = analyzer.db_over_time(samples)
        _, _, mel_db        = analyzer.mel_spectrogram(samples, n_mels=_EFF_N_MELS)

        extractor = FeatureExtractor(sr=sr)
        feats     = extractor.extract_all(samples)
        zcr_t, zcr_v = feats["zcr"]
        rms_t, rms_v = feats["rms"]
        sc_t,  sc_v  = feats["spectral_centroid"]

        # ── Model seçimi ──────────────────────────────────────────
        if model_pref == "efficientnet" and self.eff_model is not None:
            fl, lt, sm, fp, cn = self._classify_efficientnet(audio_path)
            used = "EfficientNet-B0"
        elif model_pref == "cnn" and self.cnn_model is not None:
            fl, lt, sm = self._classify_cnn(audio_path)
            fp = []; cn = list(self.cnn_le.classes_)
            used = "CNN"
        elif model_pref == "svm" and self.ml_model is not None:
            fl, lt, sm = self._classify_ml(audio_path)
            fp = []; cn = list(self.ml_le.classes_)
            used = "SVM"
        elif model_pref == "beats" and self.beats_model is not None:
            fl, lt, sm, fp, cn = self._classify_beats(audio_path)
            used = "BEATs (Modern)"
        elif model_pref == "ensemble" and self.ensemble_model is not None:
            fl, lt, sm, fp, cn = self._classify_ensemble(audio_path)
            used = "Ensemble (EfficientNet + BEATs)"
        elif model_pref == "auto":
            if self.beats_model is not None:
                fl, lt, sm, fp, cn = self._classify_beats(audio_path)
                used = "BEATs (iter3)"
            elif self.eff_model is not None:
                fl, lt, sm, fp, cn = self._classify_efficientnet(audio_path)
                used = "EfficientNet-B0"
            elif self.cnn_model is not None:
                fl, lt, sm = self._classify_cnn(audio_path)
                fp = []; cn = list(self.cnn_le.classes_)
                used = "CNN"
            elif self.ml_model is not None:
                fl, lt, sm = self._classify_ml(audio_path)
                fp = []; cn = list(self.ml_le.classes_)
                used = "SVM"
            else:
                fl, lt, sm = self.classifier.classify(extractor.extract_all(samples))
                fp = []; cn = list(NoiseClassifier.LABELS)
                used = "Kural Tabanlı"
        else:
            # Fallback: auto
            return self.analyze_for_gui(audio_path, model_pref="auto")

        return {
            "audio_path":   audio_path,
            "samples":      samples,
            "sr":           sr,
            "duration":     duration,
            "frame_labels": fl,
            "frame_times":  lt,
            "frame_probs":  fp,   # list of np.ndarray (n_windows, n_classes) — boş olabilir
            "class_names":  cn,
            "db_times":     db_times,
            "db_values":    db_values,
            "mel_db":       mel_db,
            "zcr":          (zcr_t, zcr_v),
            "rms":          (rms_t, rms_v),
            "sc":           (sc_t, sc_v),
            "summary":      sm,
            "model_used":   used,
        }

    # ─────────────────────────────────────────────────────────────────────
    #  CANLI MİKROFON — Chunk Inference (dosya I/O yok)
    # ─────────────────────────────────────────────────────────────────────

    def _infer_efficientnet_chunk(self, chunk: np.ndarray) -> tuple:
        """EfficientNet ile tek chunk (5s numpy) → (label, probs_dict)."""
        import torchaudio.transforms as TA
        classes = list(self.eff_le.classes_)

        mel_tf = TA.MelSpectrogram(
            sample_rate=_ML_SR, n_fft=_ML_N_FFT,
            hop_length=_ML_HOP_FFT, n_mels=_EFF_N_MELS
        )
        db_tf  = TA.AmplitudeToDB(top_db=80)
        to_rgb = TV.Compose([
            TV.Resize((_EFF_IMG_SIZE, _EFF_IMG_SIZE), antialias=True),
            TV.Normalize(mean=_EFF_IMAGENET_MEAN, std=_EFF_IMAGENET_STD),
        ])

        with torch.no_grad():
            y_t  = torch.FloatTensor(chunk).unsqueeze(0)
            spec = mel_tf(y_t)
            spec = db_tf(spec)
            
            mean = spec.mean(dim=-1, keepdim=True)
            std  = spec.std(dim=-1, keepdim=True) + 1e-8
            spec = (spec - mean) / std
            s_min, s_max = spec.min(), spec.max()
            if s_max > s_min:
                spec = (spec - s_min) / (s_max - s_min)
            else:
                spec = torch.zeros_like(spec)

            spec_rgb = spec.repeat(3, 1, 1)
            spec_rgb = to_rgb(spec_rgb).unsqueeze(0).to(_TORCH_DEVICE)
            logits   = self.eff_model(spec_rgb)
            probs    = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            adj      = self._apply_prior(probs, classes)
            label    = classes[int(np.argmax(adj))]
        return label, {cls: float(p) for cls, p in zip(classes, adj)}

    def _infer_cnn_chunk(self, chunk: np.ndarray) -> tuple:
        """CNN ile tek chunk (5s numpy) → (label, probs_dict)."""
        import torchaudio.transforms as TA
        classes = list(self.cnn_le.classes_)

        mel_tf = TA.MelSpectrogram(
            sample_rate=_ML_SR, n_fft=_ML_N_FFT,
            hop_length=_ML_HOP_FFT, n_mels=_CNN_N_MELS
        )
        db_tf = TA.AmplitudeToDB(top_db=80)

        with torch.no_grad():
            y_t   = torch.FloatTensor(chunk).unsqueeze(0)
            spec  = db_tf(mel_tf(y_t)).unsqueeze(0).to(_TORCH_DEVICE)
            logits = self.cnn_model(spec)
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            adj    = self._apply_prior(probs, classes)
            label  = classes[int(np.argmax(adj))]
        return label, {cls: float(p) for cls, p in zip(classes, adj)}

    def _infer_svm_chunk(self, chunk: np.ndarray) -> tuple:
        """SVM ile tek chunk (5s numpy) → (label, probs_dict)."""
        classes = list(self.ml_le.classes_)
        feat    = extract_features_ml(chunk).reshape(1, -1)

        if hasattr(self.ml_model, "predict_proba"):
            raw = self.ml_model.predict_proba(feat)[0]
            adj = self._apply_prior(raw)
        else:
            idx = int(self.ml_model.predict(feat)[0])
            adj = np.zeros(len(classes), dtype=np.float32)
            adj[idx] = 1.0

        label = classes[int(np.argmax(adj))]
        return label, {cls: float(p) for cls, p in zip(classes, adj)}

    def _infer_beats_chunk(self, chunk: np.ndarray) -> tuple:
        """
        BEATs ile tek chunk (5s numpy) → (label, probs_dict).

        Prior düzeltmesi uygulanmaz — BEATs embedding daha nötr,
        prior'ı kaldırmak ensemble dengesini bozabilir.
        Confidence threshold classify_chunk_live'da uygulanır.
        """
        if self.beats_model is None:
            raise RuntimeError("BEATs modeli yüklü değil")
        return self.beats_model.infer(chunk)

    def _infer_ensemble_chunk(self, chunk: np.ndarray) -> tuple:
        """
        Ensemble ile tek chunk (5s numpy) → (label, probs_dict).

        EfficientNet prior-düzeltmeli + BEATs ham softmax → ağırlıklı ortalama.
        """
        if self.ensemble_model is None:
            raise RuntimeError("Ensemble modeli yüklü değil (EfficientNet veya BEATs eksik)")
        return self.ensemble_model.infer(chunk)

    def classify_chunk_live(self, chunk: np.ndarray,
                            model_pref: str = "auto") -> dict:
        """
        Canlı mikrofon modu için ana inference metodu.

        chunk    : np.ndarray, shape=(n_samples,), SR=22050, float32 [-1,1]
                   Tam olarak 5s * 22050 = 110250 örnek beklenir;
                   kısa gelirse sıfır-dolgu uygulanır.
        model_pref: "auto" | "efficientnet" | "cnn" | "svm" | "beats" | "ensemble"

        Döndürür:
            label   : str  — kazanan sınıf
            probs   : dict[str, float]  — olasılıklar
            db_rms  : float  — anlık dBFS
        """
        # Uzunluk standardizasyonu
        target = int(_ML_CLIP * _ML_SR)
        if len(chunk) < target:
            padded = np.zeros(target, dtype=np.float32)
            padded[:len(chunk)] = chunk
            chunk = padded
        else:
            chunk = chunk[:target].astype(np.float32)

        # dBFS
        rms    = float(np.sqrt(np.mean(chunk ** 2)))
        db_rms = float(20 * np.log10(rms + 1e-10))

        # Model seçimi
        try:
            if model_pref == "efficientnet" and self.eff_model is not None:
                label, probs = self._infer_efficientnet_chunk(chunk)
            elif model_pref == "cnn" and self.cnn_model is not None:
                label, probs = self._infer_cnn_chunk(chunk)
            elif model_pref == "svm" and self.ml_model is not None:
                label, probs = self._infer_svm_chunk(chunk)
            elif model_pref == "beats" and self.beats_model is not None:
                label, probs = self._infer_beats_chunk(chunk)
            elif model_pref == "ensemble" and self.ensemble_model is not None:
                label, probs = self._infer_ensemble_chunk(chunk)
            elif model_pref == "auto":
                if self.beats_model is not None:
                    label, probs = self._infer_beats_chunk(chunk)
                elif self.eff_model is not None:
                    label, probs = self._infer_efficientnet_chunk(chunk)
                elif self.cnn_model is not None:
                    label, probs = self._infer_cnn_chunk(chunk)
                elif self.ml_model is not None:
                    label, probs = self._infer_svm_chunk(chunk)
                else:
                    label, probs = "UNKNOWN", {}
            else:
                # İstenen model yüklü değil — auto'ya düş, uyar
                print(f"[classify_chunk_live] '{model_pref}' modeli yüklü değil, auto kullanılıyor")
                return self.classify_chunk_live(chunk, "auto")
        except Exception as e:
            print(f"[classify_chunk_live] {e}")
            label, probs = "UNKNOWN", {}

        # ── Confidence threshold ──────────────────────────────────────
        if probs:
            max_prob = max(probs.values())
            if max_prob < 0.45:
                label = "UNKNOWN"
        # ─────────────────────────────────────────────────────────────

        return {"label": label, "probs": probs, "db_rms": db_rms}

    def _classify_ml(self, audio_path: str):
        """
        Ses dosyasını 5 sn'lik pencerelere böl, her pencereyi sınıflandır.

        Önceki yaklaşım: her pencerede argmax → label → majority vote
        Yeni yaklaşım:
          1. Her pencereden olasılık vektörü al  (predict_proba)
          2. Prior düzeltmesi uygula             (_apply_prior)
          3. Tüm pencerelerin vektörlerini ORTALA (label'dan önce)
          4. Ortalamanın argmax'ı → pencere etiketi

        Neden ortalama sonra argmax?
          Bireysel pencere hatalarını yumuşatır.
          "90% AIRCRAFT, 10% WIND" çıktısı verenler birleşince
          gerçek ortalaması daha anlamlı bir dağılım verir.

        Sonuç: (frame_labels, label_times, summary_dict)
        """
        chunks, y_full = _load_and_chunk_ml(audio_path)
        classes        = list(self.ml_le.classes_)
        has_proba      = hasattr(self.ml_model, "predict_proba")

        frame_labels    = []
        frame_adj_probs = []   # prior-düzeltmeli olasılık vektörleri

        for chunk in chunks:
            try:
                feat = extract_features_ml(chunk).reshape(1, -1)

                if has_proba:
                    raw_probs = self.ml_model.predict_proba(feat)[0]
                    adj_probs = self._apply_prior(raw_probs)
                else:
                    # predict_proba yoksa one-hot
                    pred_idx  = self.ml_model.predict(feat)[0]
                    adj_probs = np.zeros(len(classes))
                    adj_probs[pred_idx] = 1.0

                pred_idx = int(np.argmax(adj_probs))
                label    = classes[pred_idx]

                frame_labels.append(label)
                frame_adj_probs.append(adj_probs)

            except Exception as e:
                print(f"  [!] Pencere sınıflandırma hatası: {e}")
                frame_labels.append("UNKNOWN")
                frame_adj_probs.append(np.ones(len(classes)) / len(classes))

        # Zaman ekseni
        label_times = np.array([i * _ML_HOP_SEC for i in range(len(frame_labels))])

        # Özet: ham etiket dağılımı
        counts  = Counter(frame_labels)
        total   = len(frame_labels)
        summary = {k: round(100 * v / total, 1) for k, v in
                   sorted(counts.items(), key=lambda x: -x[1])}

        # Ortalama olasılık (debug için yararlı)
        if frame_adj_probs:
            avg_probs = np.mean(frame_adj_probs, axis=0)
            prob_str  = "  ".join(
                f"{cls}:{p:.2f}" for cls, p in
                sorted(zip(classes, avg_probs), key=lambda x: -x[1])
            )
            print(f"[ML] {total} pencere | Ort. olasılık → {prob_str}")
            print(f"[ML] Dağılım: {summary}")
        else:
            print("[ML] Hiç pencere işlenemedi")

        return frame_labels, label_times, summary

    def run(self, audio_path: str, prefix: str = "airport_noise"):
        """
        Tam analiz hattını çalıştır.
        ML modeli yüklüyse ML, değilse kural tabanlı sınıflandırma kullanılır.

        Returns: dict — tüm sonuçları içeren rapor
        """
        print("\n" + "=" * 60)
        print("  HAVALIMANI GÜRÜLTÜ ANALİZ SİSTEMİ BAŞLADI")
        if self.eff_model is not None:
            print("  Sınıflandırma modu: ML (EfficientNet-B0)")
        elif self.cnn_model is not None:
            print("  Sınıflandırma modu: ML (CNN)")
        elif self.ml_model is not None:
            print("  Sınıflandırma modu: ML (SVM)")
        else:
            print("  Sınıflandırma modu: Kural tabanlı (yedek)")
        print("=" * 60)

        # Adım 1: Yükleme
        print("\n[ADIM 1] Ses dosyası yükleniyor...")
        samples, sr = self.loader.load(audio_path)

        # Adım 2: Özellik çıkarımı (görselleştirme için)
        print("\n[ADIM 2] Özellikler çıkarılıyor...")
        features = self.extractor.extract_all(samples)
        summary  = self.extractor.feature_summary(features)
        print("  Özellik özeti:")
        for k, v in summary.items():
            if k != "mfcc":
                print(f"    {k:22s}: ort={v['mean']:8.2f}  std={v['std']:.2f}")

        # Adım 3: Sınıflandırma
        print("\n[ADIM 3] Gürültü kaynakları sınıflandırılıyor...")
        if self.beats_model is not None:
            frame_labels, label_times, cls_summary, _, _ = self._classify_beats(audio_path)
            clf_mode = "ML (BEATs)"
        elif self.eff_model is not None:
            frame_labels, label_times, cls_summary, _, _ = self._classify_efficientnet(audio_path)
            clf_mode = "ML (EfficientNet-B0)"
        elif self.cnn_model is not None:
            frame_labels, label_times, cls_summary = self._classify_cnn(audio_path)
            clf_mode = "ML (CNN)"
        elif self.ml_model is not None:
            frame_labels, label_times, cls_summary = self._classify_ml(audio_path)
            clf_mode = "ML (SVM)"
        else:
            frame_labels, label_times, cls_summary = self.classifier.classify(features)
            clf_mode = "rule-based"
        print("  Kaynak dağılımı:", cls_summary)

        # Adım 4: Filtreleme
        print("\n[ADIM 4] Gürültü filtreleme uygulanıyor...")
        filtered_bp = self.filter.bandpass_filter(samples, lowcut=100, highcut=2000)
        filtered_sg = self.filter.spectral_gating(samples, prop_decrease=0.85, n_std_thresh=1.5)

        # Adım 5: Görselleştirme
        print("\n[ADIM 5] Grafikler oluşturuluyor...")
        paths = []
        paths.append(self.visualizer.plot_waveform(samples, sr, prefix))
        paths.append(self.visualizer.plot_db_over_time(samples, sr, prefix))
        paths.append(self.visualizer.plot_spectrogram(samples, sr, prefix))
        paths.append(self.visualizer.plot_mel_spectrogram(samples, sr, prefix))
        paths.append(self.visualizer.plot_mfcc(features, samples, sr, prefix))
        paths.append(self.visualizer.plot_features(features, prefix))
        paths.append(self.visualizer.plot_filter_comparison(
            samples, sr, filtered_bp, filtered_sg, prefix))
        paths.append(self.visualizer.plot_full_dashboard(
            samples, sr, features, filtered_bp, filtered_sg,
            frame_labels, label_times,
            cls_summary=cls_summary,
            filename_prefix=prefix))
        paths.append(self.visualizer.plot_classification_pie(cls_summary, prefix))

        # Sonuç raporu
        report = {
            "audio_path":      audio_path,
            "sample_rate":     sr,
            "duration_s":      len(samples) / sr,
            "n_samples":       len(samples),
            "classifier_mode": clf_mode,
            "feature_summary": summary,
            "classification":  cls_summary,
            "output_files":    paths,
        }

        print("\n" + "=" * 60)
        print("  ANALİZ TAMAMLANDI")
        print(f"  Süre     : {report['duration_s']:.2f} saniye")
        print(f"  Mod      : {clf_mode}")
        print(f"  Dağılım  : {cls_summary}")
        print(f"  Çıktılar : {[Path(p).name for p in paths]}")
        print("=" * 60 + "\n")

        return report
