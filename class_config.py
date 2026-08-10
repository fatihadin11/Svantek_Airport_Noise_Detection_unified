"""
class_config.py — Tüm sınıf tanımlarının TEK kaynağı.

noise_detector.py, gui_main.py, mic_map.py, train_beats.py ve
train_efficientnet.py buradan import eder. Yeni bir sınıf eklenip
çıkarılacağı zaman SADECE bu dosya değişir — başka hiçbir yerde
sınıf ismi/rengi/ağırlığı hardcode EDİLMEMELİ.

Neden bu dosya eklendi:
  Eskiden her dosya kendi sınıf listesini/renk sözlüğünü elle
  kopyalıyordu (noise_detector._BEATS_CLASSES, train_beats.BEATS_CLASSES,
  gui_main.CLASS_COLORS, mic_map.CLASS_COLORS...). Bu üçü bile birbirinden
  küçük farklarla sapmıştı (biri AMBIENT'i unutmuş, biri "OTHER" yerine
  "---" kullanmış). Tek kaynak bu sınıf driftini yapısal olarak engeller.

Taksonomi — 2 seviyeli, ama sınıflandırma/eğitim hep ALT sınıf (leaf)
düzeyinde çalışır. Ana sınıf sadece CLASS_GROUPS üzerinden dokümantasyon
amaçlıdır ve airport-audio-collector projesindeki (veri toplama
pipeline'ı) config/settings.py::CLASS_GROUPS ile birebir aynıdır —
iki proje aynı taksonomiyi paylaşır:

  AIRCRAFT     → JET_AIRCRAFT, HELICOPTER, APU_GSE
  ENVIRONMENT  → WIND, PRECIPITATION, NATURE
  CITY_LIFE    → TRAFFIC, SIREN_ALARM, SPEECH
  OTHER        → OTHER
"""

# ---------------------------------------------------------------------------
# Model çıktı sınıfları — 9 aktif + OTHER = 10 sınıf.
# BEATs MLP / EfficientNet-B0 çıkış katmanı boyutu buradan türetilir
# (bkz. train_beats.N_CLASSES = len(CLASSES)).
#
# ⚠ SIRA ALFABETİK OLMAK ZORUNDA — bunu okunabilirlik için değiştirme!
# Sebep: train_beats.py `LabelEncoder().fit(BEATS_CLASSES)` kullanıyor ve
# sklearn'ün LabelEncoder'ı fit()'e verilen sırayı YOK SAYIP HER ZAMAN
# alfabetik sıralar (`le.classes_` her koşulda `sorted(...)` gibi davranır).
# train_beats.py bunu `assert labels == BEATS_CLASSES` ile doğruluyor —
# liste alfabetik değilse bu assert PATLAR (yaşadığımız hata buydu).
# Ayrıca noise_detector.py, eğitilmiş modelin çıktı nöronlarını bu listeyle
# POZİSYONEL olarak eşliyor (checkpoint'ten okumuyor, hardcoded sıraya
# güveniyor) — yani sıra, eğitim ve inference arasındaki gerçek bağlantı.
# Grup bazlı okunabilirlik istiyorsan CLASS_GROUPS'a bak, CLASSES'a değil.
# ---------------------------------------------------------------------------
CLASSES = [
    "APU_GSE",
    "HELICOPTER",
    "JET_AIRCRAFT",
    "NATURE",
    "OTHER",
    "PRECIPITATION",
    "SIREN_ALARM",
    "SPEECH",
    "TRAFFIC",
    "WIND",
]

N_CLASSES = len(CLASSES)

# ---------------------------------------------------------------------------
# Ana sınıf gruplaması — dokümantasyon/raporlama amaçlı.
# Pipeline mantığını etkilemez; sınıflandırma hep alt sınıf seviyesinde kalır.
# ---------------------------------------------------------------------------
CLASS_GROUPS = {
    "AIRCRAFT":    ["JET_AIRCRAFT", "HELICOPTER", "APU_GSE"],
    "ENVIRONMENT": ["WIND", "PRECIPITATION", "NATURE"],
    "CITY_LIFE":   ["TRAFFIC", "SIREN_ALARM", "SPEECH"],
    "OTHER":       ["OTHER"],
}

SUBCLASS_TO_MAIN_CLASS = {
    sub: main for main, subs in CLASS_GROUPS.items() for sub in subs
}

# ---------------------------------------------------------------------------
# GUI / plot renkleri — her sınıf için sabit, ayırt edici renk.
# "UNKNOWN" gerçek bir sınıf değil — confidence eşiğinin altında kalan ya
# da hataya düşen pencereler için sentinel etiket; yine de UI'da renk
# gerektiği için burada tutuluyor.
# ---------------------------------------------------------------------------
CLASS_COLORS = {
    "JET_AIRCRAFT":  "#FF6B35",
    "HELICOPTER":    "#FF9F1C",
    "APU_GSE":       "#C77DFF",
    "WIND":          "#4ECDC4",
    "PRECIPITATION": "#4A90D9",
    "NATURE":        "#7EE8A2",
    "TRAFFIC":       "#FFE66D",
    "SIREN_ALARM":   "#FF4444",
    "SPEECH":        "#A8DADC",
    "OTHER":         "#9E9E9E",
    "UNKNOWN":       "#6C757D",
}

# ---------------------------------------------------------------------------
# Eğitim-zamanı loss ağırlıkları (CrossEntropyLoss weight=...).
# train_beats.py ve train_efficientnet.py'de kullanılır.
#
# Eski değerler eski taksonomideki gerçek dağılıma göre elle tune
# edilmişti (AIRCRAFT eğitim setinin %78'iydi vb.). Yeni taksonomi/veri
# setiyle henüz hiç eğitim yapılmadığı için NÖTR (1.0) başlatıldı —
# gerçek sayı fabrike edilmedi. İlk eğitimden sonra per-class recall'a
# bakıp (özellikle az örnekli sınıflar: APU_GSE, PRECIPITATION,
# SIREN_ALARM gibi nadir sesler) burayı elle tune etmen gerekecek.
# ---------------------------------------------------------------------------
TRAINING_CLASS_WEIGHTS = {cls: 1.0 for cls in CLASSES}

# ---------------------------------------------------------------------------
# Inference-zamanı prior düzeltmesi (noise_detector.py::_apply_prior).
# Eğitim ağırlığından KAVRAMSAL OLARAK FARKLI: bu, canlı/dosya inference
# sırasında modelin softmax çıktısını sınıf dengesizliğine göre düzeltmek
# için kullanılır — eğitim setindeki class-weight ile aynı sayı olmak
# zorunda değildir (eskiden de değildi). Aynı sebeple nötr (1.0) başlatıldı.
# ---------------------------------------------------------------------------
INFERENCE_PRIOR_WEIGHTS = {cls: 1.0 for cls in CLASSES}
