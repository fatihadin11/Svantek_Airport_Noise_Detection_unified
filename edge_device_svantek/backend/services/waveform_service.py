"""
Ses dosyasından waveform (dalga formu) görselleştirme verisi üretir.

Yöntem: Ses dosyasını okuyup, istenen nokta sayısına (örn. 600) göre
bloklara ayırır ve her blok için maksimum genliği (amplitude) alır.
Bu, tarayıcıda hızlıca çizilebilecek küçük boyutlu bir veri seti üretir.
"""
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_POINTS = 600


def generate_waveform(file_path: Path, num_points: int = DEFAULT_POINTS) -> list[float]:
    """
    Verilen ses dosyası için 0-1 arasında normalize edilmiş genlik değerlerinden
    oluşan bir liste döndürür. Liste uzunluğu num_points'tir.
    """
    data, _sample_rate = sf.read(str(file_path), always_2d=False)

    # Stereo ise kanalları ortalayarak mono'ya indir.
    if data.ndim > 1:
        data = data.mean(axis=1)

    if len(data) == 0:
        return [0.0] * num_points

    # Veriyi num_points bloğa böl, her blokta maksimum mutlak genliği al.
    block_size = max(1, len(data) // num_points)
    points: list[float] = []
    for i in range(0, len(data), block_size):
        block = data[i : i + block_size]
        if len(block) == 0:
            continue
        points.append(float(np.abs(block).max()))

    # Tam olarak num_points uzunluğa getir (son blok kısa kalabilir).
    if len(points) > num_points:
        points = points[:num_points]
    elif len(points) < num_points:
        points.extend([0.0] * (num_points - len(points)))

    # 0-1 aralığına normalize et (sessiz dosyalarda sıfıra bölünmeyi önle).
    max_val = max(points) if points else 0.0
    if max_val > 0:
        points = [round(p / max_val, 4) for p in points]

    return points


def get_duration(file_path: Path) -> float:
    """Ses dosyasının süresini saniye cinsinden döndürür."""
    info = sf.info(str(file_path))
    return info.frames / info.samplerate
