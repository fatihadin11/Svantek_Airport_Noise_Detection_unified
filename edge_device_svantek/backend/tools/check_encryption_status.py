import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from database import get_connection


def exists_text(value):
    if not value:
        return "—"
    path = Path(value)
    return "var" if path.exists() else "yok"


with get_connection() as conn:
    rows = conn.execute("""
        SELECT id, title,
               encryption_status,
               encryption_error,
               plain_deleted,
               file_path,
               csv_path,
               raw_path,
               audio_encrypted_path,
               csv_encrypted_path,
               raw_encrypted_path
        FROM recordings
        ORDER BY id
    """).fetchall()

for r in rows:
    print("-" * 70)
    print("ID:", r["id"])
    print("Title:", r["title"])
    print("Status:", r["encryption_status"])
    print("Plain deleted:", bool(r["plain_deleted"]))
    print("Error:", r["encryption_error"])
    print()
    print("Şifresiz audio:", exists_text(r["file_path"]), "|", r["file_path"])
    print("Şifresiz csv  :", exists_text(r["csv_path"]), "|", r["csv_path"])
    print("Şifresiz raw  :", exists_text(r["raw_path"]), "|", r["raw_path"])
    print()
    print("Şifreli audio :", exists_text(r["audio_encrypted_path"]), "|", r["audio_encrypted_path"])
    print("Şifreli csv   :", exists_text(r["csv_encrypted_path"]), "|", r["csv_encrypted_path"])
    print("Şifreli raw   :", exists_text(r["raw_encrypted_path"]), "|", r["raw_encrypted_path"])
