"""
Mikrofon ile ses kaydı alma servisi.

Tasarım:
- Tek seferde sadece bir kayıt aktif olabilir (Pi'de tek mikrofon, tek panel kullanımı varsayımıyla).
- Kayıt, sounddevice.InputStream üzerinden gelen ses bloklarını bir bellek tamponunda
  (numpy dizisi listesi) biriktirir; "stop" çağrıldığında bu tampon WAV dosyasına yazılır.
- Bu sayede kayıt süresi kullanıcı "durdur" diyene kadar esnek olabilir.
"""
import threading
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 44100
CHANNELS = 1
DTYPE = "int16"

RECORDINGS_DIR = Path(__file__).parent.parent / "recordings"
RECORDINGS_DIR.mkdir(exist_ok=True)


class RecordingAlreadyActiveError(Exception):
    """Zaten aktif bir kayıt varken yeni kayıt başlatılmaya çalışıldığında fırlatılır."""


class NoActiveRecordingError(Exception):
    """Aktif kayıt yokken durdurma çağrıldığında fırlatılır."""


class _RecorderState:
    """Aktif kayıt durumunu tutan iç sınıf (tekil/singleton gibi kullanılır)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.is_recording: bool = False
        self.stream: Optional[sd.InputStream] = None
        self._frames: list[np.ndarray] = []
        self.start_time: Optional[datetime] = None

    def _callback(self, indata, frames, time_info, status):
        # PortAudio'dan gelen her ses bloğunu tamponlara ekler.
        # Not: callback ayrı bir thread'den çağrılır, bu yüzden basit append güvenlidir
        # (GIL altında liste append'i atomik kabul edilebilir).
        self._frames.append(indata.copy())

    def start(self) -> None:
        with self.lock:
            if self.is_recording:
                raise RecordingAlreadyActiveError("Zaten aktif bir kayıt var.")
            self._frames = []
            self.start_time = datetime.now()
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                callback=self._callback,
            )
            self.stream.start()
            self.is_recording = True

    def stop(self) -> tuple[Path, float, int]:
        """
        Kaydı durdurur, WAV dosyasına yazar.
        Dönüş: (dosya_yolu, süre_saniye, dosya_boyutu_bayt)
        """
        with self.lock:
            if not self.is_recording or self.stream is None:
                raise NoActiveRecordingError("Aktif kayıt bulunamadı.")

            self.stream.stop()
            self.stream.close()
            self.is_recording = False

            if self._frames:
                audio_data = np.concatenate(self._frames, axis=0)
            else:
                audio_data = np.zeros((0, CHANNELS), dtype=DTYPE)

            timestamp = self.start_time.strftime("%Y-%m-%d_%H%M%S")
            filename = f"mic_{timestamp}.wav"
            file_path = RECORDINGS_DIR / filename

            with wave.open(str(file_path), "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # int16 -> 2 byte
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_data.tobytes())

            duration_sec = len(audio_data) / SAMPLE_RATE
            file_size = file_path.stat().st_size

            self._frames = []
            self.stream = None

            return file_path, duration_sec, file_size


# Uygulama genelinde tek bir kayıt durumu örneği kullanılır.
recorder_state = _RecorderState()


def list_input_devices() -> list[dict]:
    """Pi'ye bağlı ses giriş cihazlarını listeler (debug/kurulum amaçlı kullanışlı)."""
    devices = sd.query_devices()
    return [
        {"index": i, "name": d["name"], "max_input_channels": d["max_input_channels"]}
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]
