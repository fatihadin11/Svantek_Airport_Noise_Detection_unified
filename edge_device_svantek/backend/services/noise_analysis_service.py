"""Kayıt tamamlandıktan sonra Airport AI ile olay analizi yapar."""

from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from database import get_connection
from services import encryption_service
from services.audio_conversion_service import ANALYSIS_SAMPLE_RATE, convert_for_analysis


PROJECT_ROOT = Path(__file__).resolve().parents[2]   # edge_device_svantek
CODE_ROOT = PROJECT_ROOT.parent                       # Airport_Noise — kaynak kod burada
MODELS_DIR = PROJECT_ROOT / "models"                  # edge'e özel modeller (yeni klasör)
WINDOW_SEC = 5.0
HOP_SEC = 2.5
MIN_CONFIDENCE = 0.60
IGNORED_LABELS = {"UNKNOWN", "SILENCE", "OTHER"}

_system = None
_system_lock = threading.Lock()


def _get_system():
    """Airport AI'yi yalnızca ilk analiz gerektiğinde belleğe alır."""
    global _system
    with _system_lock:
        if _system is not None:
            return _system
        if not CODE_ROOT.exists():
            raise RuntimeError(f"Airport AI kaynak kodu bulunamadı: {CODE_ROOT}")
        if not MODELS_DIR.exists():
            raise RuntimeError(f"edge_device_svantek/models klasörü bulunamadı: {MODELS_DIR}")
        sys.path.insert(0, str(CODE_ROOT))
        from noise_detector import AirportNoiseSystem  # pylint: disable=import-outside-toplevel

        system = AirportNoiseSystem(
            target_sr=ANALYSIS_SAMPLE_RATE,
            output_dir=str(PROJECT_ROOT / "analysis_outputs"),
            models_dir=str(MODELS_DIR),
            beats_encoder_path=str(MODELS_DIR / "BEATs_iter3_plus_AS2M.pt"),
            beats_mlp_path=str(MODELS_DIR / "beats_mlp.pt"), 
        )
        if not any(model is not None for model in (system.eff_model, system.cnn_model, system.ml_model, system.beats_model)):
            raise RuntimeError(
                "Eğitilmiş Airport AI modeli yüklenemedi. "
                f"Bağımlılıkları ve {MODELS_DIR} altındaki model dosyalarını kontrol edin."
            )
        _system = system
        return _system


@contextmanager
def _plain_audio_for_recording(row):
    plain_path = Path(row["file_path"])
    if plain_path.exists():
        yield plain_path
        return

    encrypted_path_value = row["audio_encrypted_path"]
    if not encrypted_path_value:
        raise FileNotFoundError("Kayıt ses dosyası bulunamadı")
    encrypted_path = Path(encrypted_path_value)
    if not encrypted_path.exists():
        raise FileNotFoundError("Şifreli kayıt ses dosyası bulunamadı")
    with encryption_service.decrypt_file_to_temp(encrypted_path, suffix=plain_path.suffix or ".wav") as temp_path:
        yield temp_path


def _event_confidence(label: str, probabilities, class_names, index: int) -> float:
    if not probabilities or index >= len(probabilities) or label not in class_names:
        return 0.0
    vector = probabilities[index]
    return float(vector[class_names.index(label)])


def _merge_windows(labels, frame_times, probabilities, class_names, duration: float) -> list[dict]:
    """Ardışık pencereleri tek, dinlenebilir olaylara dönüştürür."""
    events: list[dict] = []
    for index, label in enumerate(labels):
        confidence = _event_confidence(label, probabilities, class_names, index)
        if label in IGNORED_LABELS or confidence < MIN_CONFIDENCE:
            continue
        start_sec = float(frame_times[index])
        end_sec = min(duration, start_sec + WINDOW_SEC)
        if events and events[-1]["label"] == label and start_sec <= events[-1]["end_sec"] + HOP_SEC:
            event = events[-1]
            event["end_sec"] = max(event["end_sec"], end_sec)
            event["confidence_values"].append(confidence)
        else:
            events.append({
                "label": label,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "confidence_values": [confidence],
            })

    for event in events:
        confidence_values = event.pop("confidence_values")
        event["confidence"] = round(sum(confidence_values) / len(confidence_values), 4)
    return events


def analyze_recording(recording_id: int) -> dict:
    """Tek kaydı analiz eder ve olaylarını SQLite'a yazar."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, file_path, audio_encrypted_path FROM recordings WHERE id = ?", (recording_id,)
        ).fetchone()
        if row is None:
            raise LookupError("Kayıt bulunamadı")
        conn.execute(
            """INSERT INTO recording_analyses (recording_id, status, source_sample_rate, analysis_sample_rate, window_sec, hop_sec)
               VALUES (?, 'running', NULL, ?, ?, ?)
               ON CONFLICT(recording_id) DO UPDATE SET status='running', error_message=NULL, completed_at=NULL""",
            (recording_id, ANALYSIS_SAMPLE_RATE, WINDOW_SEC, HOP_SEC),
        )
        conn.execute("DELETE FROM recording_events WHERE recording_id = ?", (recording_id,))
        conn.commit()

    try:
        with _plain_audio_for_recording(row) as source_path:
            import soundfile as sf  # pylint: disable=import-outside-toplevel
            source_info = sf.info(source_path)
            with convert_for_analysis(source_path) as analysis_path:
                # BEATs hazırsa, mevcut EfficientNet ile beraber ensemble kullanılır;
                # aksi halde Airport projesinin güvenli otomatik seçimine düşer.
                system = _get_system()
                preferred_model = "ensemble" if system.ensemble_model is not None else "auto"
                result = system.analyze_for_gui(str(analysis_path), model_pref=preferred_model)

        events = _merge_windows(
            result["frame_labels"], result["frame_times"], result.get("frame_probs", []),
            result.get("class_names", []), float(result["duration"]),
        )
        model_name = result.get("model_used", "Airport AI")
        with get_connection() as conn:
            conn.execute(
                """UPDATE recording_analyses
                   SET status='completed', model_name=?, source_sample_rate=?, completed_at=?, error_message=NULL
                   WHERE recording_id=?""",
                (model_name, source_info.samplerate, datetime.now().isoformat(), recording_id),
            )
            conn.executemany(
                """INSERT INTO recording_events (recording_id, start_sec, end_sec, label, confidence)
                   VALUES (?, ?, ?, ?, ?)""",
                [(recording_id, item["start_sec"], item["end_sec"], item["label"], item["confidence"]) for item in events],
            )
            conn.commit()
        return {"status": "completed", "model_name": model_name, "events": events}
    except Exception as exc:
        with get_connection() as conn:
            conn.execute(
                "UPDATE recording_analyses SET status='error', error_message=? WHERE recording_id=?",
                (str(exc), recording_id),
            )
            conn.commit()
        raise


def get_analysis(recording_id: int) -> dict:
    with get_connection() as conn:
        analysis = conn.execute("SELECT * FROM recording_analyses WHERE recording_id = ?", (recording_id,)).fetchone()
        events = conn.execute(
            "SELECT id, start_sec, end_sec, label, confidence FROM recording_events WHERE recording_id = ? ORDER BY start_sec",
            (recording_id,),
        ).fetchall()
    return {
        "status": analysis["status"] if analysis else "not_started",
        "model_name": analysis["model_name"] if analysis else None,
        "error_message": analysis["error_message"] if analysis else None,
        "events": [dict(event) for event in events],
    }
