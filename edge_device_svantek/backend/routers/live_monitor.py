import os
from typing import Optional

import requests
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/api/live", tags=["live-monitor"])

EDGE_BASE_URL = os.getenv("EDGE_BASE_URL", "http://127.0.0.1:8010").rstrip("/")
EDGE_TIMEOUT_LIVE = float(os.getenv("EDGE_TIMEOUT_LIVE", "15"))

DEFAULT_PASSIVE_CMD = "#2,i,1,S?,R?,P?,M?,N?,T?,V?,v?;"


@router.get("/passive-test")
def passive_live_test(cmd: Optional[str] = Query(default=None)):
    """
    SVAN'a start/record komutu göndermeden #2 ölçüm sonucu okunabiliyor mu test eder.
    Bu endpoint kalıcı canlı sistem değildir; mimariyi kilitlemeden önce doğrulama amaçlıdır.
    """
    command = (cmd or DEFAULT_PASSIVE_CMD).strip()

    if not command.startswith("#2"):
        raise HTTPException(status_code=400, detail="Pasif canlı test için yalnızca #2 okuma komutu kullanılabilir.")

    try:
        res = requests.get(
            f"{EDGE_BASE_URL}/api/edge/live-once",
            params={"cmd": command},
            timeout=EDGE_TIMEOUT_LIVE,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Edge cihaza ulaşılamadı: {exc}") from exc

    if not res.ok:
        try:
            detail = res.json().get("detail", res.text)
        except Exception:
            detail = res.text
        raise HTTPException(status_code=res.status_code, detail=f"Edge hata: {detail}")

    return res.json()
