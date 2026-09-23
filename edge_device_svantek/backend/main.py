import os
import time
import logging
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from database import init_db, get_connection
from routers import recordings
from routers import edge_upload, recording_session, csv_views, live
from services.audio_service import RECORDINGS_DIR
from services import waveform_service

logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.include_router(recordings.router)
app.include_router(edge_upload.router)
app.include_router(recording_session.router)
app.include_router(csv_views.router)
app.include_router(live.router)


def find_matching_csv(audio_path: Path) -> Path | None:
    candidates = [audio_path.with_suffix(".csv"), audio_path.with_suffix(".CSV")]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def attach_csv_to_existing_recording(csv_path: Path) -> bool:
    with get_connection() as conn:
        rows = conn.execute("SELECT id, file_path FROM recordings").fetchall()
        for row in rows:
            audio_path = Path(row["file_path"])
            if audio_path.stem == csv_path.stem and audio_path.parent == csv_path.parent:
                conn.execute("UPDATE recordings SET csv_path = ? WHERE id = ?", (str(csv_path), row["id"]))
                conn.commit()
                logging.info(f"CSV mevcut kayda bağlandı: {csv_path.name} -> {audio_path.name}")
                return True
    return False


class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        extension = file_path.suffix.lower()
        time.sleep(0.5)

        if extension == ".csv":
            attach_csv_to_existing_recording(file_path)
            return

        if extension != ".wav":
            return

        # Edge upload endpoint zaten DB'ye eklediği için aynı dosyayı tekrar eklemeyelim.
        with get_connection() as conn:
            existing = conn.execute("SELECT id FROM recordings WHERE file_path = ?", (str(file_path),)).fetchone()
            if existing:
                return

        logging.info(f"Yeni dosya algılandı: {event.src_path}")
        title = file_path.stem
        created_at = datetime.now().isoformat()
        csv_path = find_matching_csv(file_path)

        try:
            duration_sec = waveform_service.get_duration(file_path)
        except Exception as e:
            logging.warning(f"Süre hesaplanamadı: {e}")
            duration_sec = None

        try:
            file_size = file_path.stat().st_size
        except Exception:
            file_size = None

        try:
            with get_connection() as conn:
                conn.execute(
                    """INSERT INTO recordings
                       (title, file_path, csv_path, file_format, duration_sec, file_size_bytes, source, created_at, folder)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (title, str(file_path), str(csv_path) if csv_path else None, "wav", duration_sec, file_size, "upload", created_at, "Genel"),
                )
                conn.commit()
            logging.info(f"Sisteme otomatik eklendi: {file_path.name}")
        except Exception as e:
            logging.error(f"Veritabanı kayıt hatası: {e}")


@app.on_event("startup")
def startup_event():
    init_db()
    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    observer = Observer()
    event_handler = NewFileHandler()
    observer.schedule(event_handler, str(RECORDINGS_DIR), recursive=True)
    observer.start()
    logging.info(f"İzleyici başlatıldı, klasör dinleniyor: {RECORDINGS_DIR}")


@app.on_event("shutdown")
def shutdown_event():
    pass
