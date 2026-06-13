"""
svantek_hid.py  —  Svantek SV 971 USB-HID İletişim Katmanı
═══════════════════════════════════════════════════════════════

Cihaz:  Svantek SV 971 (Pocket-Size Sound Level Meter & Analyser)
USB:    VID=0x0017  PID=0x0001  (driver: "USB driver for Svantek devices")
Protokol: Svantek Remote Control (Appendix A, SVAN 971 User Manual v3.1)

Desteklenen Fonksiyonlar:
  Function #2  →  SLM anlık ölçüm sonuçları (Lpeak, Lmax, Lmin, L, Leq…)
  Function #3  →  1/3 oktav spektrum sonuçları
  Function #7  →  Start/Stop/Reset kontrol

Bağımlılık:
  pip install hidapi         (Windows: hidapi.dll otomatik yüklenir)
  pip install numpy

Kurulum Notu:
  Windows'ta SvanPC+ kapalıyken çalıştırın.
  Gerekirse "Aygıt Yöneticisi → Svantek Devices" üzerinde sürücü yoksa
  zadig.exe ile WinUSB/HidUsb atayın.
"""

import struct
import time
import threading
import queue
from typing import Optional

import numpy as np

# ── Bağımlılık kontrolü ─────────────────────────────────────────────────────
try:
    import hid as _hid
    HID_OK = True
    HID_ERROR = ""
except ImportError as _e:
    HID_OK = False
    HID_ERROR = str(_e)

# ── Cihaz Sabitleri ─────────────────────────────────────────────────────────
SVANTEK_VID = 0x0017
SVANTEK_PID = 0x0001

# Protokol Sabitleri (Appendix A)
STX = 0x02          # Start of text
ETX = 0x03          # End of text
ACK = 0x06          # Acknowledgment
NAK = 0x15          # Negative acknowledgment

# Fonksiyon kodları
FUNC_CONTROL   = b'\x01'   # Function #1: kontrol ayarları
FUNC_SLM       = b'\x02'   # Function #2: SLM ölçüm sonuçları
FUNC_SPECTRUM  = b'\x03'   # Function #3: 1/1 ve 1/3 oktav
FUNC_SPECIAL   = b'\x07'   # Function #7: özel kontrol (Start/Stop)

# Function #7 alt komutları
CMD_START      = b'\x01'   # Ölçümü başlat
CMD_STOP       = b'\x00'   # Ölçümü durdur
CMD_RESET      = b'\x02'   # Sıfırla

# HID paket boyutu (Svantek USB: 64 byte rapor)
HID_REPORT_SIZE = 64

# 1/3 oktav merkez frekansları (Hz) — 20 Hz - 20 kHz, 30 bant
THIRD_OCT_FREQS = [
    20, 25, 31.5, 40, 50, 63, 80,
    100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
    1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
    10000, 12500, 16000, 20000
]


# ════════════════════════════════════════════════════════════════════════════
#  DÜŞÜK SEVİYE İLETİŞİM
# ════════════════════════════════════════════════════════════════════════════

def list_svantek_devices() -> list[dict]:
    """
    Bağlı tüm Svantek cihazlarını listeler.
    Dönüş: [{"path": ..., "serial": ..., "product": ...}, ...]
    """
    if not HID_OK:
        return []
    found = []
    try:
        for dev in _hid.enumerate(SVANTEK_VID, SVANTEK_PID):
            found.append({
                "path":    dev.get("path", b""),
                "serial":  dev.get("serial_number", ""),
                "product": dev.get("product_string", "SV 971"),
                "vid":     SVANTEK_VID,
                "pid":     SVANTEK_PID,
            })
    except Exception:
        pass
    return found


def _compute_checksum(data: bytes) -> int:
    """XOR checksum — Svantek protokolü."""
    cs = 0
    for b in data:
        cs ^= b
    return cs & 0xFF


def _build_request(func: bytes, sub: bytes = b"") -> bytes:
    """
    Svantek isteği oluşturur:
    STX | FUNC | [SUB] | ETX | CHECKSUM
    Toplam HID_REPORT_SIZE byte'a 0x00 ile doldurulur.
    """
    payload = func + sub
    body    = bytes([STX]) + payload + bytes([ETX])
    cs      = _compute_checksum(payload + bytes([ETX]))
    frame   = body + bytes([cs])
    # HID raporu: ilk byte report-id (0x00), sonra frame, geri kalanı 0
    packet  = bytes([0x00]) + frame
    packet  = packet.ljust(HID_REPORT_SIZE + 1, b'\x00')
    return packet


def _parse_slm_response(raw: bytes) -> Optional[dict]:
    """
    Function #2 SLM yanıtını çözümler.

    Svantek SLM yanıt formatı (Appendix A, Function #2):
    Byte 0:    STX
    Byte 1:    Function ID (0x02)
    Byte 2-3:  L (anlık SPL) — little-endian int16, 0.1 dB çözünürlük
    Byte 4-5:  Leq           — little-endian int16, 0.1 dB
    Byte 6-7:  Lmax          — little-endian int16, 0.1 dB
    Byte 8-9:  Lmin          — little-endian int16, 0.1 dB
    Byte 10-11:Lpeak         — little-endian int16, 0.1 dB
    Byte 12:   Filter (0=A, 1=B, 2=C, 3=Z, 4=LF)
    Byte 13:   Detector (0=Slow, 1=Fast, 2=Impulse)
    Byte 14:   Status bits
    Byte 15:   ETX
    Byte 16:   Checksum
    """
    try:
        # STX ara
        stx_pos = raw.find(STX)
        if stx_pos < 0:
            return None
        data = raw[stx_pos:]
        if len(data) < 10:
            return None

        func_id = data[1] if len(data) > 1 else 0

        # L (anlık SPL)
        L_raw    = struct.unpack_from("<h", data, 2)[0]  # signed int16
        Leq_raw  = struct.unpack_from("<h", data, 4)[0]
        Lmax_raw = struct.unpack_from("<h", data, 6)[0]
        Lmin_raw = struct.unpack_from("<h", data, 8)[0]

        # 0.1 dB adımlı → gerçek dB
        L    = L_raw    / 10.0
        Leq  = Leq_raw  / 10.0
        Lmax = Lmax_raw / 10.0
        Lmin = Lmin_raw / 10.0

        Lpeak_raw = struct.unpack_from("<h", data, 10)[0] if len(data) > 11 else 0
        Lpeak = Lpeak_raw / 10.0

        filt_byte = data[12] if len(data) > 12 else 0
        filt_map  = {0: "A", 1: "B", 2: "C", 3: "Z", 4: "LF"}
        filt      = filt_map.get(filt_byte, "A")

        det_byte = data[13] if len(data) > 13 else 1
        det_map  = {0: "Slow", 1: "Fast", 2: "Impulse"}
        detector = det_map.get(det_byte, "Fast")

        return {
            "L":        L,        # Anlık SPL (dB)
            "Leq":      Leq,      # Eşdeğer sürekli ses seviyesi (dB)
            "Lmax":     Lmax,     # Maksimum (dB)
            "Lmin":     Lmin,     # Minimum (dB)
            "Lpeak":    Lpeak,    # Peak (dB)
            "filter":   filt,     # Frekans ağırlıklandırması (A/C/Z…)
            "detector": detector, # Detector tipi
            "raw":      raw[:32].hex(),
        }
    except Exception:
        return None


def _parse_spectrum_response(raw: bytes, n_bands: int = 31) -> Optional[dict]:
    """
    Function #3 1/3 oktav yanıtını çözümler.

    Her bant: little-endian int16, 0.1 dB çözünürlük.
    Toplam n_bands × 2 byte veri, STX'ten sonra function id ve band sayısı gelir.
    """
    try:
        stx_pos = raw.find(STX)
        if stx_pos < 0:
            return None
        data = raw[stx_pos:]
        if len(data) < 4 + n_bands * 2:
            return None

        # data[1]=func, data[2]=bant sayısı (cihaza göre 0x1F=31 veya 0x0A=10)
        actual_bands = data[2] if len(data) > 2 else n_bands
        actual_bands = min(actual_bands, n_bands, len(THIRD_OCT_FREQS))

        levels = []
        offset = 3  # STX + func + band_count
        for i in range(actual_bands):
            if offset + 2 > len(data):
                break
            val = struct.unpack_from("<h", data, offset)[0]
            levels.append(val / 10.0)
            offset += 2

        freqs = THIRD_OCT_FREQS[:len(levels)]
        return {
            "freqs":  freqs,
            "levels": levels,  # dB
            "n":      len(levels),
        }
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════════════════
#  YÜKSEK SEVİYE BAĞLANTI SINIFI
# ════════════════════════════════════════════════════════════════════════════

class SvantekDevice:
    """
    SV 971 ile tek oturum bağlantısı.

    Kullanım:
        dev = SvantekDevice()
        if dev.connect():
            result = dev.read_slm()       # {"L": 72.3, "Leq": 70.1, ...}
            spec   = dev.read_spectrum()  # {"freqs": [...], "levels": [...]}
            dev.disconnect()
    """

    TIMEOUT_MS = 500    # HID read timeout

    def __init__(self, path: Optional[bytes] = None):
        self._path   = path     # None → ilk bulunan Svantek cihazı
        self._dev    = None
        self._lock   = threading.Lock()

    @property
    def connected(self) -> bool:
        return self._dev is not None

    def connect(self) -> bool:
        """Cihaza bağlan. Başarılıysa True döner."""
        if not HID_OK:
            raise RuntimeError(
                f"hidapi yüklü değil: {HID_ERROR}\n"
                "pip install hidapi"
            )
        try:
            dev = _hid.device()
            if self._path:
                dev.open_path(self._path)
            else:
                dev.open(SVANTEK_VID, SVANTEK_PID)
            dev.set_nonblocking(False)
            self._dev = dev
            return True
        except Exception as e:
            self._dev = None
            raise ConnectionError(f"Svantek SV 971 bağlanamadı: {e}\n"
                                  "Cihazın açık ve USB'ye takılı olduğundan emin olun.\n"
                                  "SvanPC+ programı açıksa kapatın.") from e

    def disconnect(self):
        """Bağlantıyı kapat."""
        with self._lock:
            if self._dev:
                try:
                    self._dev.close()
                except Exception:
                    pass
                self._dev = None

    def _write(self, packet: bytes):
        if not self._dev:
            raise ConnectionError("Cihaz bağlı değil")
        self._dev.write(packet)

    def _read(self) -> bytes:
        if not self._dev:
            raise ConnectionError("Cihaz bağlı değil")
        data = self._dev.read(HID_REPORT_SIZE, self.TIMEOUT_MS)
        return bytes(data) if data else b""

    def _transact(self, request: bytes) -> bytes:
        """İstek gönder, yanıt oku."""
        with self._lock:
            self._write(request)
            return self._read()

    # ── Ölçüm Komutları ─────────────────────────────────────────────────────

    def read_slm(self) -> Optional[dict]:
        """
        Anlık SLM sonuçlarını oku (Function #2).
        Dönüş: {"L": float, "Leq": float, "Lmax": float, ...} veya None
        """
        pkt = _build_request(FUNC_SLM)
        raw = self._transact(pkt)
        return _parse_slm_response(raw)

    def read_spectrum(self) -> Optional[dict]:
        """
        1/3 oktav spektrum oku (Function #3).
        Dönüş: {"freqs": [Hz], "levels": [dB], "n": int} veya None
        """
        pkt = _build_request(FUNC_SPECTRUM, b'\x03')  # 0x03 = 1/3 oktav
        raw = self._transact(pkt)
        return _parse_spectrum_response(raw)

    def start_measurement(self) -> bool:
        """Cihazda ölçümü başlat (Function #7, cmd=0x01)."""
        pkt = _build_request(FUNC_SPECIAL, CMD_START)
        raw = self._transact(pkt)
        return raw[0:1] == bytes([ACK]) if raw else False

    def stop_measurement(self) -> bool:
        """Cihazda ölçümü durdur (Function #7, cmd=0x00)."""
        pkt = _build_request(FUNC_SPECIAL, CMD_STOP)
        raw = self._transact(pkt)
        return raw[0:1] == bytes([ACK]) if raw else False


# ════════════════════════════════════════════════════════════════════════════
#  PROTOKOL KEŞİF ARACI  (bağımsız çalıştırılabilir)
# ════════════════════════════════════════════════════════════════════════════

class SvantekSniffer:
    """
    HID paketlerini ham olarak döker — protokol analizi için.
    python svantek_hid.py --sniff komutunu çalıştır.
    """

    def __init__(self, duration: float = 10.0):
        self._duration = duration

    def run(self):
        if not HID_OK:
            print(f"[Sniffer] hidapi yüklü değil: {HID_ERROR}")
            return

        devices = list_svantek_devices()
        if not devices:
            print("[Sniffer] Svantek cihazı bulunamadı.")
            print(f"  Beklenen: VID=0x{SVANTEK_VID:04X}  PID=0x{SVANTEK_PID:04X}")
            print("  Aygıt Yöneticisi → Svantek Devices → USB driver'ı kontrol et.")
            return

        print(f"[Sniffer] {len(devices)} Svantek cihazı bulundu:")
        for d in devices:
            print(f"  {d['product']}  serial={d['serial']}  path={d['path']}")

        dev = SvantekDevice()
        try:
            dev.connect()
        except ConnectionError as e:
            print(f"[Sniffer] Bağlantı hatası: {e}")
            return

        print(f"\n[Sniffer] {self._duration}s boyunca SLM + Spektrum okuyorum...")
        print("─" * 60)

        t0 = time.time()
        n  = 0
        while time.time() - t0 < self._duration:
            slm = dev.read_slm()
            if slm:
                print(f"[SLM]  L={slm['L']:.1f} dB{slm['filter']}  "
                      f"Leq={slm['Leq']:.1f}  Lmax={slm['Lmax']:.1f}  "
                      f"Lpeak={slm['Lpeak']:.1f}  det={slm['detector']}")
                print(f"       raw: {slm['raw']}")
            else:
                print("[SLM]  Yanıt yok / çözümlenemedi")

            spec = dev.read_spectrum()
            if spec and spec["levels"]:
                top5 = sorted(zip(spec["freqs"], spec["levels"]),
                              key=lambda x: -x[1])[:5]
                top5_str = "  ".join(f"{int(f)}Hz:{v:.0f}dB" for f, v in top5)
                print(f"[SPEC] {spec['n']} bant — en yüksek 5: {top5_str}")

            n += 1
            time.sleep(0.5)

        print("─" * 60)
        print(f"[Sniffer] {n} okuma tamamlandı.")
        dev.disconnect()


# ════════════════════════════════════════════════════════════════════════════
#  PYQT WORKER  (gui_main.py tarafından kullanılır)
# ════════════════════════════════════════════════════════════════════════════

try:
    from PyQt6.QtCore import QThread, pyqtSignal
    _QT_OK = True
except ImportError:
    _QT_OK = False


if _QT_OK:

    class SvantekWorker(QThread):
        """
        MicrophoneWorker ile aynı sinyal arayüzü.
        GUI'nin mevcut slot'larına bağlanabilir.

        Yayımlanan Sinyaller:
            result_signal(dict)   — inference sonucu (label, probs, db_rms, elapsed)
            vu_signal(float)      — anlık dBSPL (A-ağırlıklı)
            chunk_signal(object)  — sentetik waveform chunk (spektrum görselleştirme için)
            status_signal(str)    — durum mesajı
            error_signal(str)     — hata mesajı
            svantek_signal(dict)  — Svantek'e özgü ham ölçüm (L, Leq, Lmax, spectrum)
        """

        result_signal  = pyqtSignal(dict)
        vu_signal      = pyqtSignal(float)
        chunk_signal   = pyqtSignal(object)
        status_signal  = pyqtSignal(str)
        error_signal   = pyqtSignal(str)
        svantek_signal = pyqtSignal(dict)   # Svantek'e özgü ek veri

        # Polling aralığı (ms)
        POLL_INTERVAL_MS = 250   # 4 Hz SLM okuma
        SPEC_EVERY_N     = 4     # Her 4 SLM okumada bir spektrum

        SR             = 22050
        WINDOW_SAMPLES = int(5.0 * 22050)

        def __init__(self, system, device_path: bytes, model_pref: str):
            super().__init__()
            self._system       = system
            self._device_path  = device_path
            self._model_pref   = model_pref
            self._stop_flag    = False
            self._rec_active   = False
            self._rec_chunks: list = []
            self._rec_path     = ""
            self._last_slm     = None
            self._last_spec    = None

        # ── Kayıt Kontrolü (MicrophoneWorker uyumlu) ────────────────────────

        def start_recording(self, path: str):
            self._rec_chunks = []
            self._rec_path   = path
            self._rec_active = True

        def stop_recording(self) -> str:
            self._rec_active = False
            if not self._rec_chunks or not self._rec_path:
                return ""
            audio = np.concatenate(self._rec_chunks)
            try:
                import soundfile as sf
                sf.write(self._rec_path, audio, self.SR)
            except Exception:
                import wave as wv, struct as st
                with wv.open(self._rec_path, "wb") as wf:
                    wf.setnchannels(1); wf.setsampwidth(2)
                    wf.setframerate(self.SR)
                    int16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
                    wf.writeframes(st.pack(f"<{len(int16)}h", *int16))
            self._rec_chunks = []
            return self._rec_path

        def stop(self):
            self._stop_flag = True

        # ── Ana Thread ───────────────────────────────────────────────────────

        def run(self):
            if not HID_OK:
                self.error_signal.emit(
                    f"hidapi kütüphanesi yüklü değil.\n"
                    f"  pip install hidapi\n\nDetay: {HID_ERROR}"
                )
                return

            dev = SvantekDevice(path=self._device_path if self._device_path else None)
            try:
                dev.connect()
            except ConnectionError as e:
                self.error_signal.emit(str(e))
                return

            self.status_signal.emit("🔬  Svantek SV 971 bağlandı — ölçüm okunuyor…")
            start_time = time.time()
            n = 0
            rolling_buffer = np.zeros(self.WINDOW_SAMPLES, dtype=np.float32)

            while not self._stop_flag:
                try:
                    slm = dev.read_slm()
                except Exception as e:
                    self.error_signal.emit(f"Svantek okuma hatası: {e}")
                    break

                if slm is None:
                    time.sleep(self.POLL_INTERVAL_MS / 1000.0)
                    continue

                self._last_slm = slm

                # dB(A) → VU metre sinyali (kalibreli SPL, dBFS değil)
                db_spl = slm["L"]
                self.vu_signal.emit(db_spl)

                # Spektrum (her SPEC_EVERY_N okumada bir)
                spec = None
                if n % self.SPEC_EVERY_N == 0:
                    try:
                        spec = dev.read_spectrum()
                        self._last_spec = spec
                    except Exception:
                        pass

                # Svantek'e özgü sinyal — UI'daki özel panel için
                sv_data = {**slm}
                if spec:
                    sv_data["spectrum"] = spec
                self.svantek_signal.emit(sv_data)

                # Sentetik waveform üret (spektrum görselleştirme için)
                # Svantek waveform vermediği için 1/3 oktav → sentetik sinyal
                chunk = self._synthesize_chunk(slm, spec)
                rolling_buffer = np.roll(rolling_buffer, -len(chunk))
                rolling_buffer[-len(chunk):] = chunk

                if self._rec_active:
                    self._rec_chunks.append(chunk.copy())

                self.chunk_signal.emit(chunk.copy())

                # Inference (5s buffer dolduğunda)
                if n > 0 and n % 20 == 0:   # ~5s @ 250ms
                    elapsed = time.time() - start_time
                    try:
                        if self._system is not None:
                            res = self._system.classify_chunk_live(
                                rolling_buffer.copy(), self._model_pref
                            )
                            res["elapsed"]  = elapsed
                            res["samples"]  = rolling_buffer.copy()
                            res["db_rms"]   = db_spl          # kalibreli dB
                            res["db_spl"]   = db_spl
                            self.result_signal.emit(res)
                    except Exception as inf_e:
                        print(f"[SvantekWorker] Inference hatası: {inf_e}")

                n += 1
                time.sleep(self.POLL_INTERVAL_MS / 1000.0)

            dev.disconnect()
            self.status_signal.emit("⏹  Svantek bağlantısı kesildi.")

        # ── Yardımcı ─────────────────────────────────────────────────────────

        def _synthesize_chunk(self,
                              slm: dict,
                              spec: Optional[dict],
                              dur: float = 0.25) -> np.ndarray:
            """
            Svantek ölçüm verisinden sentetik ses chunk'ı üretir.

            1/3 oktav spektrum varsa her bant için sinüs dalgası ekler,
            yoksa pink noise ile L değerine göre ölçekler.
            Amaç: LiveSpectrumWidget'in görsel güncellenmesi.
            """
            n_smp = int(self.SR * dur)
            t     = np.linspace(0, dur, n_smp, endpoint=False)

            if spec and spec.get("levels"):
                # 1/3 oktav → sinüs bileşenlerinin toplamı
                sig = np.zeros(n_smp, dtype=np.float32)
                for freq, lev in zip(spec["freqs"], spec["levels"]):
                    if freq <= 0 or lev < -80:
                        continue
                    amp = 10 ** ((lev - 94.0) / 20.0) * 0.1   # dBSPL → lineer
                    sig += amp * np.sin(2 * np.pi * freq * t).astype(np.float32)
            else:
                # Pink noise, L değerine göre ölçeklendirilmiş
                white = np.random.randn(n_smp).astype(np.float32)
                # Basit pink noise yaklaşımı (1/f)
                fft   = np.fft.rfft(white)
                freqs = np.fft.rfftfreq(n_smp, 1 / self.SR)
                freqs[0] = 1   # DC'den kaçın
                fft  /= np.sqrt(freqs)
                sig   = np.fft.irfft(fft, n_smp).astype(np.float32)

                # dBSPL → normalize
                db_spl = slm.get("L", 60.0)
                target_rms = 10 ** ((db_spl - 94.0) / 20.0) * 0.1
                current_rms = float(np.sqrt(np.mean(sig ** 2))) + 1e-12
                sig *= target_rms / current_rms

            # Klip
            peak = float(np.max(np.abs(sig)))
            if peak > 0.99:
                sig /= peak * 1.01
            return sig


# ════════════════════════════════════════════════════════════════════════════
#  YARDIMCI FONKSİYONLAR  (gui_main.py tarafından çağrılır)
# ════════════════════════════════════════════════════════════════════════════

def find_svantek_gui_entry() -> list[dict]:
    """
    GUI dropdown'u için Svantek cihaz girişleri döner.
    Dönüş: [{"label": str, "path": bytes, "is_svantek": True}, ...]
    """
    entries = []
    if not HID_OK:
        return entries
    for dev in list_svantek_devices():
        serial = dev["serial"] or "?"
        label  = f"🔬  Svantek {dev['product']} (USB-HID, s/n:{serial})"
        entries.append({
            "label":      label,
            "path":       dev["path"],
            "is_svantek": True,
            "product":    dev["product"],
            "serial":     serial,
        })
    return entries


def hid_available() -> tuple[bool, str]:
    """(ok: bool, mesaj: str) döner."""
    return HID_OK, HID_ERROR


# ════════════════════════════════════════════════════════════════════════════
#  KOMİT SATIRI ÇALIŞTIRMA
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    if "--sniff" in sys.argv:
        dur = float(sys.argv[sys.argv.index("--sniff") + 1]) \
              if len(sys.argv) > sys.argv.index("--sniff") + 1 else 10.0
        SvantekSniffer(duration=dur).run()
    elif "--list" in sys.argv:
        devs = list_svantek_devices()
        if devs:
            print(f"{len(devs)} Svantek cihazı:")
            for d in devs:
                print(f"  {d['product']}  VID={SVANTEK_VID:#06x}  "
                      f"PID={SVANTEK_PID:#06x}  serial={d['serial']}")
        else:
            print("Svantek cihazı bulunamadı.")
            if not HID_OK:
                print(f"  hidapi hatası: {HID_ERROR}")
    else:
        print("Kullanım:")
        print("  python svantek_hid.py --list          # Bağlı cihazları listele")
        print("  python svantek_hid.py --sniff [süre]  # Ham veri dök (varsayılan 10s)")
