"""Ses kayıtlarını Airport AI modelinin beklediği biçime dönüştürür."""

from __future__ import annotations

import math
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


ANALYSIS_SAMPLE_RATE = 22_050


@contextmanager
def convert_for_analysis(source_path: Path) -> Iterator[Path]:
    """Kaydı mono, 22.05 kHz WAV olarak geçici dosyaya dönüştürür.

    SVANTEK WAV'leri çoğunlukla 8 kHz olduğundan, Airport AI'nin eğitim
    örnekleme hızıyla eşleştirilmesi gerekir. Çıktı kalıcı olarak tutulmaz.
    """
    samples, source_rate = sf.read(source_path, dtype="float32", always_2d=True)
    if samples.size == 0:
        raise ValueError("Ses kaydı boş")

    mono = samples.mean(axis=1)
    if source_rate != ANALYSIS_SAMPLE_RATE:
        divisor = math.gcd(int(source_rate), ANALYSIS_SAMPLE_RATE)
        mono = resample_poly(mono, ANALYSIS_SAMPLE_RATE // divisor, int(source_rate) // divisor)

    with tempfile.NamedTemporaryFile(delete=False, suffix="_analysis.wav") as temp_file:
        target_path = Path(temp_file.name)

    try:
        sf.write(target_path, mono, ANALYSIS_SAMPLE_RATE, subtype="PCM_16")
        yield target_path
    finally:
        target_path.unlink(missing_ok=True)
