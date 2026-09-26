"""
paths.py — Airport Noise Detection projesi için TEK yol kaynağı.

Bu proje birden fazla bilgisayarda / farklı klasör yollarında çalışacağı
için (D:\\ sürücüsü, farklı kullanıcı adı, farklı klonlama yolu vb.)
HİÇBİR dosyada mutlak yol elle yazılmaz. Bunun yerine:

  1. PROJECT_ROOT, bu dosyanın (paths.py) bulunduğu klasörden otomatik
     hesaplanır — yani repo nereye klonlanırsa klonlansın (D:\\, C:\\,
     başka bir kullanıcı adı vb.) doğru çalışır.
  2. Modeller ve canlı klipler için varsayılan konum PROJECT_ROOT
     altında bir alt klasördür — başka bir sürücüye (D:\\) bağımlılık
     yoktur.
  3. Gerekirse (örn. modelleri başka bir diskte tutmak istersen) her
     yol, aynı isimde bir ortam değişkeniyle override edilebilir.

noise_detector.py, gui_main.py, dataset_builder.py ve train_beats.py bu
yolları elle KOPYALAMAZ, buradan import eder — bir yolu değiştirmek
gerekirse SADECE bu dosya değişir (class_config.py'nin sınıflar için
tek kaynak olması gibi).
"""

import os


def _resolve(env_var: str, *default_parts: str) -> str:
    """Ortam değişkeni set edilmişse onu, yoksa PROJECT_ROOT'a göre
    varsayılan alt yolu döndürür."""
    override = os.environ.get(env_var)
    if override:
        return override
    return os.path.join(PROJECT_ROOT, *default_parts)


# Bu dosyanın bulunduğu klasör = repo kökü (Airport_Noise).
# Elle yazılmıyor — otomatik hesaplanıyor, bu yüzden hangi bilgisayarda
# hangi klasöre klonlanırsa klonlansın doğru sonucu verir.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Ana sistemin model ağırlıkları (best_model.pkl, best_cnn.pt, ...).
# Varsayılan: <repo>/models  |  Override: AIRPORT_NOISE_MODELS_DIR
MODELS_DIR = _resolve("AIRPORT_NOISE_MODELS_DIR", "models")

# Canlı mikrofon klipleri (GUI Faz 2 — pending/approved/rejected).
# Varsayılan: <repo>/Airport_Live_Clips  |  Override: AIRPORT_NOISE_LIVE_CLIPS_DIR
LIVE_CLIPS_DIR = _resolve("AIRPORT_NOISE_LIVE_CLIPS_DIR", "Airport_Live_Clips")
APPROVED_MANIFEST = os.path.join(LIVE_CLIPS_DIR, "approved_manifest.csv")

# Eğitim önbelleği / manifest.
# Varsayılan: <repo>/cache  |  Override: AIRPORT_NOISE_CACHE_DIR
CACHE_DIR = _resolve("AIRPORT_NOISE_CACHE_DIR", "cache")
MANIFEST_CSV = os.path.join(CACHE_DIR, "manifest_v6.csv")

# BEATs eğitim grafikleri/çıktıları — her zaman repo altında.
PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "training_beats")

# BEATs model dosyaları — hepsi MODELS_DIR altında (D:\models yerine).
BEATS_ENCODER_PATH = os.path.join(MODELS_DIR, "BEATs_iter3_plus_AS2M.pt")
BEATS_MLP_PATH = os.path.join(MODELS_DIR, "beats_mlp.pt")
BEATS_EMBED_CACHE = os.path.join(MODELS_DIR, "beats_embed_cache.pkl")
BEATS_AUG_CACHE = os.path.join(MODELS_DIR, "beats_aug_cache.pkl")

# İsteğe bağlı ek eğitim verisi (gerçek SVANTEK mikrofon kayıtları).
# NOT: Bu klasör büyük ham ses kayıtları içerebilir; varsayılan olarak
# repo kökü altına koyduk ama muhtemelen ayrı bir diskte/klasörde
# tutmak isteyeceksin — o durumda AIRPORT_NOISE_SVANTEK_DIR ortam
# değişkenini kendi yoluna ayarla (bkz. README).
SVANTEK_RECORDINGS_DIR = _resolve("AIRPORT_NOISE_SVANTEK_DIR", "Svantek_Recordings")
