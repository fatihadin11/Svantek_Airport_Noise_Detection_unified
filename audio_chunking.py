"""
audio_chunking.py — Uzun ses dosyalarını sabit uzunluklu (CLIP_DUR)
pencerelere bölmek için TEK kaynak.

NEDEN EKLENDİ:
  load_audio_fixed()/load_audio_clip() her zaman bir dosyanın sadece
  İLK 5.5 saniyesini okuyordu (librosa.load(..., duration=5.5)) — saatlerce
  süren collector kayıtlarında geri kalan her şey çöpe gidiyordu. Bu modül
  bir dosyayı süresine göre N adet ayrık 5 saniyelik pencereye bölüp her
  birini AYRI bir eğitim örneği yapıyor.

dataset_builder.py (collector DB + live clips) VE train_beats.py
(Svantek tarayıcı) burada import eder — ikisi de aynı chunklama mantığını
ve aynı path-kodlama biçimini kullanmalı, yoksa extract_source_id() /
load_audio_clip() senkron kalamaz. Yeni bir yerde chunklama gerekirse
BURAYA eklenmeli, başka dosyada elle kopyalanmamalı.

KODLAMA BİÇİMİ: "{gerçek_yol}::{start_sec}"
  Her chunk'ı manifest'te ayrı bir "path" değeri gibi taşır (CSV şeması
  değişmez — hâlâ sadece "path","label" iki sütun). load_audio_clip()
  bunu çözüp librosa.load(..., offset=start_sec) ile doğru pencereyi okur.
  extract_source_id() gerçek dosya adını (:: öncesini) kullanarak aynı
  kaynağın TÜM chunk'larını aynı gruba toplar — GroupShuffleSplit
  leakage-safe kalır (aynı videonun 40 chunk'ı asla train/val/test'e
  BÖLÜNMEZ, hepsi aynı split'e düşer).
"""

import os
import re
import librosa

CLIP_DUR             = 5.0   # train_beats.py / train_efficientnet.py DURATION ile AYNI olmalı
MAX_CHUNKS_PER_FILE   = 40   # ~3.3 dk/dosya tavanı — tek uzun video bir sınıfı domine etmesin diye
CHUNK_SEP             = "::"


def encode_chunk_path(real_path: str, start_sec: float) -> str:
    return f"{real_path}{CHUNK_SEP}{start_sec:.2f}"


def decode_chunk_path(encoded_path: str) -> tuple[str, float]:
    """':: 'yoksa (eski manifest satırı / tek-chunk kayıt) start_sec=0.0 varsayar —
    geriye dönük uyumlu, eski manifest_v6.csv satırları da çalışmaya devam eder."""
    if CHUNK_SEP in encoded_path:
        real_path, start_str = encoded_path.rsplit(CHUNK_SEP, 1)
        try:
            return real_path, float(start_str)
        except ValueError:
            return encoded_path, 0.0   # ':: ' tesadüfen başka amaçla geçiyorsa güvenli çık
    return encoded_path, 0.0


def chunk_path_exists(encoded_path: str) -> bool:
    """os.path.exists() encoded path ile ÇALIŞMAZ (böyle bir dosya yok) —
    gerçek yolu çözüp onu kontrol eder. df['path'].apply(...) yerine kullan."""
    real_path, _ = decode_chunk_path(encoded_path)
    return os.path.exists(real_path)


def chunk_source_file(real_path: str, clip_dur: float = CLIP_DUR,
                       max_chunks: int = MAX_CHUNKS_PER_FILE) -> list[str]:
    """
    Bir ses dosyasını clip_dur'luk ayrık pencerelere böler.
    Kısa dosyalar (<=clip_dur) TEK elemanlı liste döner (start_sec=0.0) —
    eskisiyle birebir aynı davranış, hiçbir kısa dosya için değişiklik yok.
    Dosya yoksa/süresi okunamazsa boş liste döner (çağıran 'missing' sayar).
    """
    if not os.path.exists(real_path):
        return []
    try:
        total_dur = librosa.get_duration(path=real_path)
    except Exception:
        return [encode_chunk_path(real_path, 0.0)]   # süre okunamadıysa eski (tek-chunk) davranışa düş

    if total_dur <= clip_dur:
        return [encode_chunk_path(real_path, 0.0)]

    n_chunks = min(int(total_dur // clip_dur), max_chunks)
    return [encode_chunk_path(real_path, i * clip_dur) for i in range(n_chunks)]


def extract_source_id(path: str) -> str:
    """
    Aynı ses kaynağından gelen dosyaları/chunk'ları gruplar — GroupShuffleSplit
    ile kullanılır, aynı gruptaki örnekler AYNI split'e düşer (leakage önlenir).
    train_beats.py VE train_efficientnet.py burada import eder — ikisi de
    aynı gruplamayı kullanmalı.

    ⚠ path 'gerçek_yol::start_sec' kodlu olabilir — ÖNCE gerçek yol çözülür,
    yoksa aynı videonun N chunk'ı N ayrı grup sayılır ve leakage-safe split
    anlamsızlaşır.

    ESC-50  :  1-101296-A-19.wav  →  esc50_101296
               1-101296-B-19.wav  →  esc50_101296   (aynı grup!)
    Diğerleri: dosya adının kendisi (her KAYNAK DOSYA benzersiz kaynak;
               o dosyanın tüm chunk'ları bu ismi paylaşır)
    """
    real_path, _ = decode_chunk_path(path)
    fname = os.path.basename(real_path)

    m = re.match(r'^\d+-(\d+)-[A-Z]-\d+\.wav$', fname)
    if m:
        return f"esc50_{m.group(1)}"

    return os.path.splitext(fname)[0]
