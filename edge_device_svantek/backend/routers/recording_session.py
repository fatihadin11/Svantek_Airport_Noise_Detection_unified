import os
import requests
from datetime import datetime
from threading import Lock
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from models import RecordingStartResponse, RecordingStopResponse

router = APIRouter(prefix="/api/recording-session", tags=["recording-session"])

EDGE_BASE_URL = os.getenv("EDGE_BASE_URL", "http://127.0.0.1:8010").rstrip("/")
EDGE_TIMEOUT_START = float(os.getenv("EDGE_TIMEOUT_START", "10"))
EDGE_TIMEOUT_STOP = float(os.getenv("EDGE_TIMEOUT_STOP", "900"))

_session_lock = Lock()
_active_session: dict | None = None


class SessionStartRequest(BaseModel):
    folder: Optional[str] = "Genel"
    tag: Optional[str] = ""
    color: Optional[str] = "#f2a65a"
    title: Optional[str] = None


def _edge_post(path: str, payload: dict, timeout: float) -> dict:
    url = f"{EDGE_BASE_URL}{path}"
    try:
        res = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Edge cihaza ulaşılamadı: {exc}") from exc

    if not res.ok:
        try:
            detail = res.json().get("detail", res.text)
        except Exception:
            detail = res.text
        raise HTTPException(status_code=res.status_code, detail=f"Edge hata: {detail}")

    return res.json()


@router.get("/status")
def session_status():
    try:
        res = requests.get(f"{EDGE_BASE_URL}/api/edge/status", timeout=EDGE_TIMEOUT_START)
        edge = res.json() if res.ok else {"status": "error", "detail": res.text}
    except requests.RequestException as exc:
        edge = {"status": "offline", "detail": str(exc)}

    with _session_lock:
        session = dict(_active_session) if _active_session else None

    return {"edge_base_url": EDGE_BASE_URL, "active_session": session, "edge": edge}


@router.post("/start", response_model=RecordingStartResponse)
def start_session(payload: SessionStartRequest):
    global _active_session

    with _session_lock:
        if _active_session is not None:
            raise HTTPException(status_code=409, detail="Zaten aktif bir SVAN kaydı var")

        session = {
            "folder": payload.folder or "Genel",
            "tag": payload.tag or "",
            "color": payload.color or "#f2a65a",
            "title": payload.title,
            "started_at": datetime.now().isoformat(),
        }
        _active_session = session

    try:
        _edge_post("/api/edge/start", session, timeout=EDGE_TIMEOUT_START)
    except Exception:
        with _session_lock:
            _active_session = None
        raise

    return RecordingStartResponse(status="recording", message="SVAN kaydı başladı")


@router.post("/stop", response_model=RecordingStopResponse)
def stop_session():
    global _active_session

    with _session_lock:
        if _active_session is None:
            raise HTTPException(status_code=409, detail="Aktif SVAN kaydı bulunamadı")
        session = dict(_active_session)

    result = _edge_post("/api/edge/stop-and-upload", session, timeout=EDGE_TIMEOUT_STOP)

    with _session_lock:
        _active_session = None

    recording = result.get("recording")
    if not recording:
        raise HTTPException(status_code=500, detail="Edge upload tamamlandı ama recording cevabı alınamadı")

    return RecordingStopResponse(status="stopped", recording=recording)
