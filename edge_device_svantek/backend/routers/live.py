import os
from typing import Optional

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/live", tags=["live"])

EDGE_BASE_URL = os.getenv("EDGE_BASE_URL", "http://127.0.0.1:8010").rstrip("/")


class LiveStartRequest(BaseModel):
    threshold_enabled: bool = False
    threshold_db: float = 120.0
    trigger_hold_sec: float = 1.0
    release_hold_sec: float = 5.0
    auto_rearm: bool = False
    rearm_cooldown_sec: float = 30.0
    setup_name: str = "AUTO120"
    folder: Optional[str] = "Otomatik"
    tag: Optional[str] = ""
    color: Optional[str] = "#ef4444"
    title: Optional[str] = None


class LiveStopRequest(BaseModel):
    finalize: bool = False


def _model_dict(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def edge_request(method: str, path: str, timeout: int = 60, payload: dict | None = None):
    url = f"{EDGE_BASE_URL}{path}"
    try:
        response = requests.request(method, url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Edge agent'a ulaşılamadı: {exc}") from exc

    if response.status_code >= 400:
        try:
            body = response.json()
            detail = body.get("detail", body) if isinstance(body, dict) else body
        except Exception:
            detail = response.text
        raise HTTPException(status_code=response.status_code, detail=detail)

    try:
        return response.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Edge agent geçersiz JSON döndürdü: {exc}") from exc


@router.get("/status")
def live_status():
    return edge_request("GET", "/api/edge/live/status", timeout=30)


@router.post("/start")
def live_start(payload: LiveStartRequest):
    return edge_request(
        "POST",
        "/api/edge/live/start",
        timeout=90,
        payload=_model_dict(payload),
    )


@router.post("/stop")
def live_stop(payload: LiveStopRequest):
    return edge_request(
        "POST",
        "/api/edge/live/stop",
        timeout=1800 if payload.finalize else 90,
        payload=_model_dict(payload),
    )


@router.get("/latest")
def live_latest():
    return edge_request("GET", "/api/edge/live/once", timeout=30)
