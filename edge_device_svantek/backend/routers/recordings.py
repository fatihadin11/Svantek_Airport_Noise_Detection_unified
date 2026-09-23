import csv
import io
import sqlite3
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Form, Response
from fastapi.responses import FileResponse

from database import get_connection, row_to_dict
from models import (
    RecordingOut,
    RecordingUpdate,
    RecordingStartResponse,
    RecordingStopResponse,
    WaveformOut,
    FolderOut,
    FolderCreate,
    CsvGraphOut,
    RecordingAnalysisOut,
)
from services import audio_service, waveform_service, encryption_service, noise_analysis_service
from services.audio_service import RecordingAlreadyActiveError, NoActiveRecordingError

router = APIRouter(prefix="/api/recordings", tags=["recordings"])

ALLOWED_UPLOAD_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
ALLOWED_CSV_EXTENSION = ".csv"
MAX_CSV_GRAPH_POINTS = 2000
DELIMITER_CANDIDATES = (";", ",", "\t")


async def _save_encrypted_upload(upload: UploadFile, destination: Path) -> bytes:
    """Şifreli dosyayı anahtarla doğrular, diskte yalnızca şifreli kopyayı tutar."""
    contents = await upload.read()
    if not encryption_service.is_encrypted_blob(contents):
        raise HTTPException(status_code=400, detail=f"{upload.filename} şifreli kayıt formatında değil")
    try:
        plain = encryption_service.decrypt_bytes(contents)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Şifreli kayıt açılamadı: {exc}") from exc
    destination.write_bytes(contents)
    return plain


def _duration_from_wav_bytes(wav_bytes: bytes) -> Optional[float]:
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = Path(temp.name)
    try:
        with temp:
            temp.write(wav_bytes)
        return waveform_service.get_duration(temp_path)
    except Exception:
        return None
    finally:
        temp_path.unlink(missing_ok=True)


def serialize_recording(row) -> dict:
    """DB satırını API cevabına hazırlar; sunucu dosya yollarını dışarı vermez."""
    data = row_to_dict(row)
    data["has_csv"] = bool(data.get("csv_path") or data.get("csv_encrypted_path"))
    data["has_encrypted_audio"] = bool(data.get("audio_encrypted_path"))
    data["has_encrypted_csv"] = bool(data.get("csv_encrypted_path"))
    data["has_encrypted_raw"] = bool(data.get("raw_encrypted_path"))
    data["plain_deleted"] = bool(data.get("plain_deleted"))
    data["encryption_status"] = data.get("encryption_status") or "plain"
    data.pop("file_path", None)
    data.pop("csv_path", None)
    data.pop("raw_path", None)
    data.pop("audio_encrypted_path", None)
    data.pop("csv_encrypted_path", None)
    data.pop("raw_encrypted_path", None)
    data.pop("waveform_cache", None)
    data.pop("encryption_error", None)
    return data


def _read_text_flexible(csv_path: Path) -> str:
    """Logger CSV'leri farklı encoding ile gelebilir; yaygın encodingleri dener."""
    for encoding in ("utf-8-sig", "utf-8", "cp1254", "latin-1"):
        try:
            return csv_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return csv_path.read_text(encoding="utf-8", errors="replace")


def _detect_delimiter(text: str) -> str:
    """
    csv.Sniffer bazı logger çıktılarında ilk metadata satırındaki virgülden dolayı
    yanlış delimiter seçebiliyor. Bu yüzden en çok kullanılan adayı manuel seçiyoruz.
    """
    non_empty_lines = [line for line in text.splitlines() if line.strip()]
    sample_lines = non_empty_lines[:50]
    scores = {
        delimiter: sum(line.count(delimiter) for line in sample_lines)
        for delimiter in DELIMITER_CANDIDATES
    }

    best_delimiter, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score > 0:
        return best_delimiter

    try:
        dialect = csv.Sniffer().sniff(text[:2048], delimiters=",;\t")
        return dialect.delimiter
    except csv.Error:
        return ","


def _to_float(value: str) -> Optional[float]:
    cleaned = str(value).strip()
    if not cleaned:
        return None

    # Binlik/ondalık ayrımlarını temel düzeyde destekle.
    # "1,25" -> 1.25, "1.234,56" -> 1234.56, "1,234.56" -> 1234.56
    cleaned = cleaned.replace(" ", "")
    if cleaned.count(",") == 1 and cleaned.count(".") == 0:
        cleaned = cleaned.replace(",", ".")
    elif cleaned.count(",") == 1 and cleaned.count(".") >= 1 and cleaned.rfind(",") > cleaned.rfind("."):
        cleaned = cleaned.replace(".", "").replace(",", ".")
    elif cleaned.count(".") == 1 and cleaned.count(",") >= 1 and cleaned.rfind(".") > cleaned.rfind(","):
        cleaned = cleaned.replace(",", "")

    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_datetime(value: str) -> Optional[datetime]:
    cleaned = str(value).strip()
    if not cleaned:
        return None

    formats = (
        "%d.%m.%Y %H:%M:%S",
        "%d.%m.%Y %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    )
    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            pass
    return None


def _is_numbering_header(label: str) -> bool:
    normalized = label.strip().lower().replace(" ", "")
    return normalized in {"no.", "no", "number", "index", "sample", "sıra", "sira"}


def _is_event_header(label: str) -> bool:
    normalized = label.strip().lower()
    return normalized in {"event", "events", "olay", "olaylar"} or "event" in normalized


def _is_time_header(label: str) -> bool:
    normalized = label.strip().lower()
    return (
        "date" in normalized
        or "time" in normalized
        or "tarih" in normalized
        or "zaman" in normalized
        or normalized in {"t", "time(s)", "time [s]", "time_sec", "seconds", "saniye"}
    )


def _numeric_count(row: list[str]) -> int:
    return sum(1 for cell in row if _to_float(cell) is not None)


def _find_table_start(rows: list[list[str]]) -> tuple[Optional[list[str]], int]:
    """
    Metadata satırlarını atlayıp asıl tablo başlığını bulur.
    Örnek format:
      82 : Logger results...
      ;;P1;P1...
      No.;Date & time;LCpeak...
      1;01.07.2026...
    """
    for index, row in enumerate(rows[:-1]):
        next_row = rows[index + 1]
        row_has_text = any(cell and _to_float(cell) is None for cell in row)
        next_has_numbers = _numeric_count(next_row) >= 2
        same_or_similar_width = len(row) > 1 and len(next_row) >= max(2, len(row) - 1)

        if row_has_text and next_has_numbers and same_or_similar_width:
            return row, index + 1

    # Header yoksa ilk sayısal satırı data başlangıcı kabul et.
    for index, row in enumerate(rows):
        if _numeric_count(row) > 0:
            return None, index

    return None, len(rows)


def _downsample_graph(x_values: list[float], series: list[dict], max_points: int = MAX_CSV_GRAPH_POINTS) -> tuple[list[float], list[dict]]:
    if len(x_values) <= max_points:
        return x_values, series

    step = len(x_values) / max_points
    indices = [min(len(x_values) - 1, int(i * step)) for i in range(max_points)]
    sampled_x = [x_values[index] for index in indices]
    sampled_series = [
        {"name": item["name"], "values": [item["values"][index] for index in indices]}
        for item in series
    ]
    return sampled_x, sampled_series


def parse_csv_graph(csv_path: Path) -> CsvGraphOut:
    text = _read_text_flexible(csv_path)
    if not text.strip():
        return CsvGraphOut(x_values=[], series=[], rows=0)

    delimiter = _detect_delimiter(text)
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = [[cell.strip() for cell in row] for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        return CsvGraphOut(x_values=[], series=[], rows=0)

    header, data_start = _find_table_start(rows)
    data_rows = rows[data_start:]

    if not data_rows:
        return CsvGraphOut(x_values=[], series=[], rows=0)

    max_columns = max(len(row) for row in data_rows)
    labels = header or [f"Kolon {index + 1}" for index in range(max_columns)]
    labels = labels + [f"Kolon {index + 1}" for index in range(len(labels), max_columns)]

    # X ekseni: önce tarih/saat kolonu, sonra No./index/time benzeri kolon, yoksa örnek numarası.
    datetime_col = next((index for index, label in enumerate(labels) if _is_time_header(label)), None)
    x_col = None
    x_label = "Örnek"

    if datetime_col is not None:
        x_col = datetime_col
        x_label = labels[datetime_col] or "Zaman"
    else:
        for index, label in enumerate(labels):
            if _is_numbering_header(label) or _is_time_header(label):
                x_col = index
                x_label = label or "Örnek"
                break

    # Sayısal seri kolonları: No./Date/Events kolonlarını çizimden çıkar.
    candidate_series_cols: list[int] = []
    for col_index in range(max_columns):
        label = labels[col_index] if col_index < len(labels) else f"Kolon {col_index + 1}"
        if col_index == x_col or _is_numbering_header(label) or _is_time_header(label) or _is_event_header(label):
            continue

        numeric_values_in_col = 0
        for row in data_rows:
            if col_index < len(row) and _to_float(row[col_index]) is not None:
                numeric_values_in_col += 1
        if numeric_values_in_col > 0:
            candidate_series_cols.append(col_index)

    # Header yoksa veya filtreler bütün kolonları elediyse, ilk sayısal kolonu seri olarak kullan.
    if not candidate_series_cols:
        for col_index in range(max_columns):
            if col_index == x_col:
                continue
            if any(col_index < len(row) and _to_float(row[col_index]) is not None for row in data_rows):
                candidate_series_cols.append(col_index)

    x_values: list[float] = []
    series_values: dict[int, list[float]] = {col_index: [] for col_index in candidate_series_cols}
    first_datetime: Optional[datetime] = None

    for row_index, row in enumerate(data_rows):
        # X değeri
        x_value: Optional[float] = None
        if x_col is not None and x_col < len(row):
            if x_col == datetime_col:
                dt_value = _parse_datetime(row[x_col])
                if dt_value is not None:
                    if first_datetime is None:
                        first_datetime = dt_value
                    x_value = (dt_value - first_datetime).total_seconds()
            else:
                x_value = _to_float(row[x_col])

        if x_value is None:
            x_value = float(len(x_values))

        parsed_series: dict[int, float] = {}
        for col_index in candidate_series_cols:
            if col_index < len(row):
                y_value = _to_float(row[col_index])
                if y_value is not None:
                    parsed_series[col_index] = y_value

        # Aynı satırda çizilecek tüm seçili seriler yoksa satırı atla.
        if len(parsed_series) != len(candidate_series_cols):
            continue

        x_values.append(float(x_value))
        for col_index in candidate_series_cols:
            series_values[col_index].append(float(parsed_series[col_index]))

    series = [
        {
            "name": labels[col_index] if col_index < len(labels) and labels[col_index] else f"Kolon {col_index + 1}",
            "values": values,
        }
        for col_index, values in series_values.items()
        if values
    ]

    x_values, series = _downsample_graph(x_values, series)

    return CsvGraphOut(
        x_values=x_values,
        x_label=x_label,
        series=series,
        rows=len(data_rows),
    )


# --- KLASÖR İŞLEMLERİ ---
# (Routelarda ID çakışmasını önlemek için en üstte tanımlanmalıdır)

@router.get("/folders/list", response_model=list[FolderOut])
def list_folders():
    with get_connection() as conn:
        rows = conn.execute("SELECT name FROM folders ORDER BY name ASC").fetchall()
        return [{"name": r["name"]} for r in rows]

@router.post("/folders/create", response_model=FolderOut)
def create_folder(folder: FolderCreate):
    with get_connection() as conn:
        try:
            conn.execute("INSERT INTO folders (name) VALUES (?)", (folder.name,))
            conn.commit()
        except sqlite3.IntegrityError:
            pass # Zaten varsa görmezden gel
    return {"name": folder.name}

@router.delete("/folders/{folder_name}")
def delete_folder(folder_name: str):
    if folder_name == "Genel":
        raise HTTPException(status_code=400, detail="Genel klasörü silinemez")
    with get_connection() as conn:
        # Klasör silinince içindeki sesleri Genel klasörüne taşı
        conn.execute("UPDATE recordings SET folder = 'Genel' WHERE folder = ?", (folder_name,))
        conn.execute("DELETE FROM folders WHERE name = ?", (folder_name,))
        conn.commit()
    return {"status": "deleted"}

# --- KAYIT İŞLEMLERİ ---

@router.get("/{recording_id}/analysis", response_model=RecordingAnalysisOut)
def get_recording_analysis(recording_id: int):
    with get_connection() as conn:
        exists = conn.execute("SELECT 1 FROM recordings WHERE id = ?", (recording_id,)).fetchone()
    if not exists:
        raise HTTPException(status_code=404, detail="Kayıt bulunamadı")
    return noise_analysis_service.get_analysis(recording_id)


@router.post("/{recording_id}/analysis", response_model=RecordingAnalysisOut, status_code=202)
def start_recording_analysis(recording_id: int, background_tasks: BackgroundTasks):
    with get_connection() as conn:
        exists = conn.execute("SELECT 1 FROM recordings WHERE id = ?", (recording_id,)).fetchone()
    if not exists:
        raise HTTPException(status_code=404, detail="Kayıt bulunamadı")

    current = noise_analysis_service.get_analysis(recording_id)
    if current["status"] == "running":
        return current
    background_tasks.add_task(noise_analysis_service.analyze_recording, recording_id)
    return {"status": "running", "model_name": None, "error_message": None, "events": []}

@router.get("", response_model=list[RecordingOut])
def list_recordings():
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM recordings ORDER BY created_at DESC").fetchall()
        return [serialize_recording(r) for r in rows]

@router.get("/{recording_id}", response_model=RecordingOut)
def get_recording(recording_id: int):
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM recordings WHERE id = ?", (recording_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Kayıt bulunamadı")
        return serialize_recording(row)

@router.get("/{recording_id}/audio")
def stream_audio(recording_id: int):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT file_path, file_format, audio_encrypted_path FROM recordings WHERE id = ?",
            (recording_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Kayıt bulunamadı")

    file_path = Path(row["file_path"])
    media_type = f"audio/{row['file_format']}"

    if file_path.exists():
        return FileResponse(path=file_path, media_type=media_type, filename=file_path.name, content_disposition_type="inline")

    encrypted_value = row["audio_encrypted_path"]
    if encrypted_value:
        encrypted_path = Path(encrypted_value)
        if encrypted_path.exists():
            try:
                plain_audio = encryption_service.decrypt_file_to_bytes(encrypted_path)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"Şifreli ses dosyası açılamadı: {exc}")
            headers = {"Content-Disposition": f'inline; filename="{file_path.name}"'}
            return Response(content=plain_audio, media_type=media_type, headers=headers)

    raise HTTPException(status_code=404, detail="Ses dosyası bulunamadı")

@router.get("/{recording_id}/waveform", response_model=WaveformOut)
def get_waveform(recording_id: int):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT file_path, duration_sec, waveform_cache, audio_encrypted_path FROM recordings WHERE id = ?",
            (recording_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Kayıt bulunamadı")

        if row["waveform_cache"]:
            import json
            points = json.loads(row["waveform_cache"])
        else:
            file_path = Path(row["file_path"])
            if file_path.exists():
                points = waveform_service.generate_waveform(file_path)
            elif row["audio_encrypted_path"] and Path(row["audio_encrypted_path"]).exists():
                try:
                    with encryption_service.decrypt_file_to_temp(Path(row["audio_encrypted_path"]), suffix=file_path.suffix) as temp_audio:
                        points = waveform_service.generate_waveform(temp_audio)
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=f"Şifreli ses dosyası açılamadı: {exc}")
            else:
                raise HTTPException(status_code=404, detail="Ses dosyası bulunamadı")

            import json
            conn.execute("UPDATE recordings SET waveform_cache = ? WHERE id = ?", (json.dumps(points), recording_id))
            conn.commit()

        return WaveformOut(points=points, duration_sec=row["duration_sec"] or 0.0)

@router.get("/{recording_id}/csv-graph", response_model=CsvGraphOut)
def get_csv_graph(recording_id: int):
    with get_connection() as conn:
        row = conn.execute("SELECT csv_path, csv_encrypted_path FROM recordings WHERE id = ?", (recording_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Kayıt bulunamadı")

    if not row["csv_path"] and not row["csv_encrypted_path"]:
        raise HTTPException(status_code=404, detail="Bu kayıt için CSV dosyası yok")

    csv_path = Path(row["csv_path"]) if row["csv_path"] else None
    try:
        if csv_path and csv_path.exists():
            return parse_csv_graph(csv_path)

        encrypted_value = row["csv_encrypted_path"]
        if encrypted_value and Path(encrypted_value).exists():
            with encryption_service.decrypt_file_to_temp(Path(encrypted_value), suffix=".csv") as temp_csv:
                return parse_csv_graph(temp_csv)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CSV okunamadı: {exc}")

    raise HTTPException(status_code=404, detail="CSV dosyası bulunamadı")

@router.post("/start", response_model=RecordingStartResponse)
def start_recording():
    try:
        audio_service.recorder_state.start()
    except RecordingAlreadyActiveError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return RecordingStartResponse(status="recording", message="Kayıt başladı")

@router.post("/stop", response_model=RecordingStopResponse)
def stop_recording():
    try:
        file_path, duration_sec, file_size = audio_service.recorder_state.stop()
    except NoActiveRecordingError as e:
        raise HTTPException(status_code=409, detail=str(e))

    title = file_path.stem
    created_at = datetime.now().isoformat()
    csv_path = file_path.with_suffix(".csv")
    csv_path_value = str(csv_path) if csv_path.exists() else None

    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO recordings (title, file_path, csv_path, file_format, duration_sec, file_size_bytes, source, created_at, sample_rate, channels)
            VALUES (?, ?, ?, ?, ?, ?, 'microphone', ?, ?, ?)""",
            (
                title,
                str(file_path),
                csv_path_value,
                "wav",
                duration_sec,
                file_size,
                created_at,
                audio_service.SAMPLE_RATE,
                audio_service.CHANNELS,
            )
        )
        conn.commit()
        new_row = conn.execute("SELECT * FROM recordings WHERE id = ?", (cursor.lastrowid,)).fetchone()

    return RecordingStopResponse(status="stopped", recording=serialize_recording(new_row))

@router.post("/upload", response_model=RecordingOut)
async def upload_recording(
    file: UploadFile = File(...),
    csv_file: Optional[UploadFile] = File(None),
):
    original_name = Path(file.filename)
    extension = original_name.suffix.lower()

    if extension not in ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Desteklenmeyen dosya formatı")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    safe_filename = f"upload_{timestamp}{extension}"
    destination = audio_service.RECORDINGS_DIR / safe_filename

    csv_destination = None
    if csv_file and csv_file.filename:
        csv_name = Path(csv_file.filename)
        if csv_name.suffix.lower() != ALLOWED_CSV_EXTENSION:
            raise HTTPException(status_code=400, detail="CSV dosyası .csv uzantılı olmalıdır")
        csv_destination = audio_service.RECORDINGS_DIR / f"upload_{timestamp}.csv"

    contents = await file.read()
    destination.write_bytes(contents)

    if csv_destination and csv_file:
        csv_contents = await csv_file.read()
        csv_destination.write_bytes(csv_contents)

    try:
        duration_sec = waveform_service.get_duration(destination)
    except Exception:
        duration_sec = None

    file_size = destination.stat().st_size
    created_at = datetime.now().isoformat()

    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO recordings (title, file_path, csv_path, file_format, duration_sec, file_size_bytes, source, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'upload', ?)""",
            (
                original_name.stem,
                str(destination),
                str(csv_destination) if csv_destination else None,
                extension.lstrip("."),
                duration_sec,
                file_size,
                created_at,
            )
        )
        conn.commit()
        new_row = conn.execute("SELECT * FROM recordings WHERE id = ?", (cursor.lastrowid,)).fetchone()

    return serialize_recording(new_row)


@router.post("/upload-encrypted", response_model=RecordingOut)
async def upload_encrypted_recording(
    background_tasks: BackgroundTasks,
    audio_file_enc: UploadFile = File(...),
    csv_file_enc: Optional[UploadFile] = File(None),
    svl_file_enc: Optional[UploadFile] = File(None),
    title: str = Form(""),
    folder: str = Form("Genel"),
):
    """Pi/edge tarafında üretilmiş .enc kayıtları panele doğrudan aktarır."""
    if not (audio_file_enc.filename or "").lower().endswith(".enc"):
        raise HTTPException(status_code=400, detail="Ses kaydı .enc uzantılı olmalıdır")
    for companion in (csv_file_enc, svl_file_enc):
        if companion and companion.filename and not companion.filename.lower().endswith(".enc"):
            raise HTTPException(status_code=400, detail="Yan dosyalar .enc uzantılı olmalıdır")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_title = title.strip() or Path(audio_file_enc.filename or "kayit.wav.enc").stem.replace(".wav", "")
    target_dir = audio_service.RECORDINGS_DIR / f"imported_{timestamp}"
    target_dir.mkdir(parents=True, exist_ok=True)
    wav_path = target_dir / "audio.wav"
    csv_path = target_dir / "data_all.csv"
    raw_path = target_dir / "raw.SVL"
    audio_enc = encryption_service.encrypted_path_for(wav_path)
    csv_enc = encryption_service.encrypted_path_for(csv_path)
    raw_enc = encryption_service.encrypted_path_for(raw_path)

    try:
        wav_plain = await _save_encrypted_upload(audio_file_enc, audio_enc)
        if csv_file_enc and csv_file_enc.filename:
            await _save_encrypted_upload(csv_file_enc, csv_enc)
        else:
            csv_enc = None
        if svl_file_enc and svl_file_enc.filename:
            await _save_encrypted_upload(svl_file_enc, raw_enc)
        else:
            raw_enc = None

        with get_connection() as conn:
            selected_folder = folder or "Genel"
            conn.execute("INSERT OR IGNORE INTO folders (name) VALUES (?)", (selected_folder,))
            cursor = conn.execute(
                """
                INSERT INTO recordings (
                    title, file_path, csv_path, raw_path, file_format, duration_sec,
                    file_size_bytes, source, created_at, folder,
                    encryption_status, encryption_algorithm, encrypted_at,
                    audio_encrypted_path, csv_encrypted_path, raw_encrypted_path, plain_deleted
                ) VALUES (?, ?, ?, ?, 'wav', ?, ?, 'svantek', ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    base_title, str(wav_path), str(csv_path) if csv_enc else None,
                    str(raw_path) if raw_enc else None, _duration_from_wav_bytes(wav_plain),
                    len(wav_plain), datetime.now().isoformat(), selected_folder,
                    "encrypted", encryption_service.ALGORITHM_NAME, datetime.now().isoformat(),
                    str(audio_enc), str(csv_enc) if csv_enc else None, str(raw_enc) if raw_enc else None,
                ),
            )
            conn.commit()
            new_row = conn.execute("SELECT * FROM recordings WHERE id = ?", (cursor.lastrowid,)).fetchone()

        background_tasks.add_task(noise_analysis_service.analyze_recording, cursor.lastrowid)
        return serialize_recording(new_row)
    except HTTPException:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(target_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Şifreli kayıt kaydedilemedi: {exc}") from exc

@router.patch("/{recording_id}", response_model=RecordingOut)
def update_recording(recording_id: int, update: RecordingUpdate):
    with get_connection() as conn:
        existing = conn.execute("SELECT * FROM recordings WHERE id = ?", (recording_id,)).fetchone()
        if existing is None:
            raise HTTPException(status_code=404, detail="Kayıt bulunamadı")

        new_title = update.title if update.title is not None else existing["title"]
        new_folder = update.folder if update.folder is not None else existing["folder"]
        new_tag = update.tag if update.tag is not None else existing["tag"]
        new_color = update.color if update.color is not None else existing["color"]

        conn.execute(
            "UPDATE recordings SET title = ?, folder = ?, tag = ?, color = ? WHERE id = ?",
            (new_title, new_folder, new_tag, new_color, recording_id),
        )
        conn.commit()
        updated_row = conn.execute("SELECT * FROM recordings WHERE id = ?", (recording_id,)).fetchone()

    return serialize_recording(updated_row)

@router.delete("/{recording_id}")
def delete_recording(recording_id: int):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT file_path, csv_path, raw_path, audio_encrypted_path, csv_encrypted_path, raw_encrypted_path FROM recordings WHERE id = ?",
            (recording_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Kayıt bulunamadı")

        paths_to_delete = [row["file_path"], row["csv_path"], row["raw_path"], row["audio_encrypted_path"], row["csv_encrypted_path"], row["raw_encrypted_path"]]
        for value in paths_to_delete:
            if not value:
                continue
            path = Path(value)
            if path.exists():
                path.unlink()

        conn.execute("DELETE FROM recordings WHERE id = ?", (recording_id,))
        conn.commit()

    return {"status": "deleted", "id": recording_id}
