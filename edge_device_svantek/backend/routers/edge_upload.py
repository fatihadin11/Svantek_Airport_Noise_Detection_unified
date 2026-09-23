import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile

from database import get_connection, row_to_dict
from models import RecordingOut
from services.audio_service import RECORDINGS_DIR
from services import encryption_service, noise_analysis_service, waveform_service

router = APIRouter(prefix="/api/edge-upload", tags=["edge-upload"])


def serialize_recording(row) -> dict:
    data = row_to_dict(row)
    data["has_csv"] = bool(data.get("csv_path") or data.get("csv_encrypted_path"))
    data["has_encrypted_audio"] = bool(data.get("audio_encrypted_path"))
    data["has_encrypted_csv"] = bool(data.get("csv_encrypted_path"))
    data["has_encrypted_raw"] = bool(data.get("raw_encrypted_path"))
    data["plain_deleted"] = bool(data.get("plain_deleted"))
    data["encryption_status"] = data.get("encryption_status") or "plain"
    for key in (
        "file_path", "csv_path", "raw_path",
        "audio_encrypted_path", "csv_encrypted_path", "raw_encrypted_path",
        "waveform_cache",
    ):
        data.pop(key, None)
    return data


def _safe_name(value: str, fallback: str = "recording") -> str:
    value = (value or fallback).strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("._") or fallback


async def _save_and_verify(upload: UploadFile, destination: Path) -> bytes:
    contents = await upload.read()
    if not encryption_service.is_encrypted_blob(contents):
        raise HTTPException(
            status_code=400,
            detail=f"{upload.filename} beklenen şifreli formatta değil",
        )
    try:
        plain = encryption_service.decrypt_bytes(contents)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Şifreli upload doğrulanamadı: {exc}",
        ) from exc
    destination.write_bytes(contents)
    return plain


def _duration_from_wav_bytes(wav_bytes: bytes) -> Optional[float]:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = Path(tmp.name)
    try:
        with tmp:
            tmp.write(wav_bytes)
        return waveform_service.get_duration(tmp_path)
    except Exception:
        return None
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/recording", response_model=RecordingOut)
async def receive_edge_recording(
    background_tasks: BackgroundTasks,
    wav_file_enc: UploadFile = File(...),
    csv_file_enc: UploadFile = File(...),
    svl_file_enc: UploadFile = File(...),
    encrypted_upload: bool = Form(...),
    encryption_algorithm: str = Form(""),
    title: str = Form(""),
    folder: str = Form("Genel"),
    tag: str = Form(""),
    color: str = Form("#f2a65a"),
    device_file_name: str = Form(""),
    edge_id: str = Form("default-edge"),
    sample_rate: int = Form(8000),
    channels: int = Form(1),
):
    if not encrypted_upload:
        raise HTTPException(
            status_code=400,
            detail="Edge upload yalnızca şifreli kayıt kabul eder",
        )

    created_at = datetime.now().isoformat()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_title = title.strip() or f"svan_{timestamp}"
    target_dir = RECORDINGS_DIR / _safe_name(f"{timestamp}_{base_title}")
    target_dir.mkdir(parents=True, exist_ok=True)

    wav_path = target_dir / "audio.wav"
    csv_path = target_dir / "data_all.csv"
    raw_path = target_dir / "raw.SVL"

    audio_enc = encryption_service.encrypted_path_for(wav_path)
    csv_enc = encryption_service.encrypted_path_for(csv_path)
    raw_enc = encryption_service.encrypted_path_for(raw_path)

    try:
        wav_plain = await _save_and_verify(wav_file_enc, audio_enc)
        await _save_and_verify(csv_file_enc, csv_enc)
        await _save_and_verify(svl_file_enc, raw_enc)

        duration_sec = _duration_from_wav_bytes(wav_plain)
        file_size = len(wav_plain)
        encrypted_at = datetime.now().isoformat()
        algorithm = encryption_algorithm or encryption_service.ALGORITHM_NAME

        with get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO folders (name) VALUES (?)",
                (folder or "Genel",),
            )
            cursor = conn.execute(
                """
                INSERT INTO recordings (
                    title, file_path, csv_path, raw_path,
                    device_file_name, edge_id,
                    file_format, duration_sec, file_size_bytes,
                    source, created_at,
                    sample_rate, channels, folder, tag, color,
                    encryption_status, encryption_algorithm, encrypted_at,
                    audio_encrypted_path, csv_encrypted_path, raw_encrypted_path,
                    plain_deleted, encryption_error
                ) VALUES (
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    base_title,
                    str(wav_path),
                    str(csv_path),
                    str(raw_path),
                    device_file_name or None,
                    edge_id,
                    "wav",
                    duration_sec,
                    file_size,
                    "svantek",
                    created_at,
                    sample_rate,
                    channels,
                    folder or "Genel",
                    tag or None,
                    color or "#f2a65a",
                    "encrypted",
                    algorithm,
                    encrypted_at,
                    str(audio_enc),
                    str(csv_enc),
                    str(raw_enc),
                    1,
                    None,
                ),
            )
            conn.commit()
            row = conn.execute(
                "SELECT * FROM recordings WHERE id = ?",
                (cursor.lastrowid,),
            ).fetchone()

        # SVANTEK kaydı merkeze ulaştığında AI analizi arka planda başlar.
        # Panel kayıt listesini hemen gösterebilir; analiz sonucu ayrı endpointten gelir.
        background_tasks.add_task(noise_analysis_service.analyze_recording, cursor.lastrowid)
        return serialize_recording(row)
    except HTTPException:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=f"Şifreli edge upload kaydedilemedi: {exc}",
        ) from exc
