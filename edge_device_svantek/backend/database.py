"""
SQLite veritabanı bağlantı katmanı.
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "recordings.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS folders (
    name TEXT PRIMARY KEY
);

INSERT OR IGNORE INTO folders (name) VALUES ('Genel');

CREATE TABLE IF NOT EXISTS recordings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT NOT NULL,
    file_path       TEXT NOT NULL UNIQUE,
    csv_path        TEXT,
    raw_path        TEXT,
    device_file_name TEXT,
    edge_id         TEXT,
    file_format     TEXT NOT NULL,
    duration_sec    REAL,
    file_size_bytes INTEGER,
    source          TEXT NOT NULL CHECK (source IN ('microphone', 'upload', 'svantek')),
    created_at      TEXT NOT NULL,
    sample_rate     INTEGER,
    channels        INTEGER,
    waveform_cache  TEXT,
    folder          TEXT DEFAULT 'Genel',
    tag             TEXT,
    color           TEXT DEFAULT '#f2a65a',
    encryption_status TEXT DEFAULT 'plain',
    encryption_algorithm TEXT,
    encrypted_at     TEXT,
    audio_encrypted_path TEXT,
    csv_encrypted_path TEXT,
    raw_encrypted_path TEXT,
    plain_deleted   INTEGER DEFAULT 0,
    encryption_error TEXT
);

CREATE INDEX IF NOT EXISTS idx_created_at ON recordings(created_at);

CREATE TABLE IF NOT EXISTS recording_analyses (
    recording_id          INTEGER PRIMARY KEY REFERENCES recordings(id) ON DELETE CASCADE,
    status                TEXT NOT NULL DEFAULT 'not_started',
    model_name            TEXT,
    source_sample_rate    INTEGER,
    analysis_sample_rate  INTEGER,
    window_sec            REAL,
    hop_sec               REAL,
    completed_at          TEXT,
    error_message         TEXT
);

CREATE TABLE IF NOT EXISTS recording_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id  INTEGER NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    start_sec     REAL NOT NULL,
    end_sec       REAL NOT NULL,
    label         TEXT NOT NULL,
    confidence    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_recording_events_recording ON recording_events(recording_id, start_sec);
"""


def _ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, column_definition: str) -> None:
    """Mevcut veritabanlarını yeni kolonlarla uyumlu hale getirir."""
    columns = [row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()]
    if column_name not in columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")


def _recordings_table_sql(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='recordings'").fetchone()
    return row["sql"] if row and row["sql"] else ""


def _migrate_recordings_source_check(conn: sqlite3.Connection) -> None:
    """Eski DB'deki source CHECK ('microphone','upload') ise tabloyu svantek destekli yeniden kurar."""
    sql = _recordings_table_sql(conn)
    if not sql or "svantek" in sql:
        return

    conn.execute("ALTER TABLE recordings RENAME TO recordings_old")
    conn.executescript(
        """
        CREATE TABLE recordings (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            title           TEXT NOT NULL,
            file_path       TEXT NOT NULL UNIQUE,
            csv_path        TEXT,
            raw_path        TEXT,
            device_file_name TEXT,
            edge_id         TEXT,
            file_format     TEXT NOT NULL,
            duration_sec    REAL,
            file_size_bytes INTEGER,
            source          TEXT NOT NULL CHECK (source IN ('microphone', 'upload', 'svantek')),
            created_at      TEXT NOT NULL,
            sample_rate     INTEGER,
            channels        INTEGER,
            waveform_cache  TEXT,
            folder          TEXT DEFAULT 'Genel',
            tag             TEXT,
            color           TEXT DEFAULT '#f2a65a',
            encryption_status TEXT DEFAULT 'plain',
            encryption_algorithm TEXT,
            encrypted_at     TEXT,
            audio_encrypted_path TEXT,
            csv_encrypted_path TEXT,
            raw_encrypted_path TEXT,
            plain_deleted   INTEGER DEFAULT 0,
            encryption_error TEXT
        );
        """
    )

    old_cols = [row["name"] for row in conn.execute("PRAGMA table_info(recordings_old)").fetchall()]
    new_cols = [row["name"] for row in conn.execute("PRAGMA table_info(recordings)").fetchall()]
    common = [col for col in new_cols if col in old_cols]
    if common:
        columns_sql = ", ".join(common)
        conn.execute(f"INSERT INTO recordings ({columns_sql}) SELECT {columns_sql} FROM recordings_old")

    conn.execute("DROP TABLE recordings_old")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON recordings(created_at)")


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)
        _migrate_recordings_source_check(conn)
        _ensure_column(conn, "recordings", "csv_path", "TEXT")
        _ensure_column(conn, "recordings", "raw_path", "TEXT")
        _ensure_column(conn, "recordings", "device_file_name", "TEXT")
        _ensure_column(conn, "recordings", "edge_id", "TEXT")
        _ensure_column(conn, "recordings", "encryption_status", "TEXT DEFAULT 'plain'")
        _ensure_column(conn, "recordings", "encryption_algorithm", "TEXT")
        _ensure_column(conn, "recordings", "encrypted_at", "TEXT")
        _ensure_column(conn, "recordings", "audio_encrypted_path", "TEXT")
        _ensure_column(conn, "recordings", "csv_encrypted_path", "TEXT")
        _ensure_column(conn, "recordings", "raw_encrypted_path", "TEXT")
        _ensure_column(conn, "recordings", "plain_deleted", "INTEGER DEFAULT 0")
        _ensure_column(conn, "recordings", "encryption_error", "TEXT")
        conn.commit()


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def row_to_dict(row: sqlite3.Row) -> dict:
    return dict(row)
