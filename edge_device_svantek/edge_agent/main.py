import json
import os
import re
import shutil
import subprocess
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
TOOLS_DIR = BACKEND_DIR / "tools"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from services import encryption_service
except Exception:
    encryption_service = None

EDGE_ID = os.getenv("EDGE_ID", "ubuntu-svan-edge")
MAIN_BACKEND_URL = os.getenv("MAIN_BACKEND_URL", "http://10.0.2.2:8000").rstrip("/")
EDGE_WORK_DIR = Path(os.getenv("EDGE_WORK_DIR", "/tmp/pi-ses-sistemi-edge"))
DELETE_AFTER_UPLOAD = os.getenv("DELETE_AFTER_UPLOAD", "1") == "1"
SVAN_SAMPLE_RATE = int(os.getenv("SVAN_SAMPLE_RATE", "8000"))
EDGE_ENCRYPTION_REQUIRED = os.getenv("EDGE_ENCRYPTION_REQUIRED", "1") == "1"
MANUAL_RECORD_SETUP = os.getenv("MANUAL_RECORD_SETUP", "RECORD").strip().upper()

active_session: dict | None = None
live_session: dict | None = None
usb_lock = threading.RLock()

threshold_monitor_stop_event = threading.Event()
threshold_monitor_thread: threading.Thread | None = None
latest_live_snapshot: dict | None = None

app = FastAPI(title="Pi Ses Sistemi Edge Agent")


class StartRequest(BaseModel):
    folder: Optional[str] = "Genel"
    tag: Optional[str] = ""
    color: Optional[str] = "#f2a65a"
    title: Optional[str] = None


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


def run_tool(args: list[str], timeout: int = 300) -> str:
    """
    backend/tools içindeki mevcut scriptleri çalıştırır.
    Edge agent sudo ile çalıştırıldığı için bu subprocess'ler de USB'ye erişebilir.
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # SVAN USB hattında aynı anda iki subprocess çalışırsa komutlar çakışabiliyor.
    # Özellikle canlı polling sürerken Stop'a basıldığında S0 komutu cihaza ulaşmayabiliyordu.
    # Bu lock bütün SVAN araçlarını sıraya alır.
    with usb_lock:
        result = subprocess.run(
            [sys.executable, *args],
            cwd=str(BACKEND_DIR),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )

    output = result.stdout or ""

    if result.returncode != 0:
        raise RuntimeError(
            f"Komut başarısız: {' '.join(args)}\n"
            f"Return code: {result.returncode}\n"
            f"Çıktı:\n{output}"
        )

    return output


def parse_last_logger_name(output: str) -> str:
    """
    svantek_control.py --stop çıktısından #7,LB,88; kısmını yakalar.
    """
    match = re.search(r"#7,LB,([^;\s]+);", output)
    if not match:
        raise RuntimeError(f"Son logger dosya adı bulunamadı. Çıktı:\n{output}")

    return match.group(1).strip()


def parse_svl_from_ls(output: str, logger_name: str) -> dict:
    """
    svantek_ls.py çıktısından ilgili SVL dosyasının cluster/size bilgisini yakalar.
    Beklenen satır örneği:
    FILE 88.SVL cluster=469 size=246182
    """
    wanted = f"{logger_name}.SVL".upper()

    for line in output.splitlines():
        line_upper = line.upper()

        if "FILE" not in line_upper or wanted not in line_upper:
            continue

        cluster_match = re.search(r"cluster=(\d+)", line, flags=re.IGNORECASE)
        size_match = re.search(r"size=(\d+)", line, flags=re.IGNORECASE)

        if not cluster_match or not size_match:
            continue

        return {
            "filename": wanted,
            "cluster": int(cluster_match.group(1)),
            "size": int(size_match.group(1)),
            "line": line,
        }

    raise RuntimeError(
        f"{wanted} svantek_ls.py çıktısında bulunamadı.\n"
        f"Çıktı:\n{output}"
    )


def safe_session_name(logger_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_logger = re.sub(r"[^a-zA-Z0-9_-]+", "_", logger_name)
    return f"svan_{clean_logger}_{timestamp}"


def _ensure_pi_encryption_ready() -> None:
    if encryption_service is None:
        raise RuntimeError(
            "Encryption service yüklenemedi. "
            "backend/services/encryption_service.py mevcut mu?"
        )
    encryption_service.get_key_from_env()


def _encrypt_and_verify_for_upload(path: Path) -> Path:
    _ensure_pi_encryption_ready()
    encrypted_path = encryption_service.encrypted_path_for(path)
    encryption_service.encrypt_file(path, encrypted_path)

    encrypted_bytes = encrypted_path.read_bytes()
    if not encryption_service.is_encrypted_blob(encrypted_bytes):
        raise RuntimeError(f"Şifreli dosya formatı doğrulanamadı: {encrypted_path}")

    plain_bytes = encryption_service.decrypt_bytes(encrypted_bytes)
    if len(plain_bytes) != path.stat().st_size:
        raise RuntimeError(f"Şifreleme doğrulamasında boyut uyuşmazlığı: {path}")

    return encrypted_path


def upload_recording_to_main_backend(
    session_dir: Path,
    title: str,
    folder: str,
    tag: str,
    color: str,
    logger_name: str,
) -> dict:
    audio_path = session_dir / "audio.wav"
    csv_path = session_dir / "data_all.csv"
    svl_path = session_dir / "raw.SVL"

    for path in (audio_path, csv_path, svl_path):
        if not path.exists():
            raise RuntimeError(f"Upload dosyası bulunamadı: {path}")

    if not EDGE_ENCRYPTION_REQUIRED:
        raise RuntimeError(
            "Pi-side encryption kapatılamaz. "
            "EDGE_ENCRYPTION_REQUIRED=1 ve AES_KEY_B64 zorunludur."
        )

    audio_enc_path = _encrypt_and_verify_for_upload(audio_path)
    csv_enc_path = _encrypt_and_verify_for_upload(csv_path)
    svl_enc_path = _encrypt_and_verify_for_upload(svl_path)

    url = f"{MAIN_BACKEND_URL}/api/edge-upload/recording"
    data = {
        "edge_id": EDGE_ID,
        "device_file_name": f"{logger_name}.SVL",
        "title": title,
        "folder": folder or "Genel",
        "tag": tag or "",
        "color": color or "#f2a65a",
        "sample_rate": str(SVAN_SAMPLE_RATE),
        "channels": "1",
        "encrypted_upload": "1",
        "encryption_algorithm": encryption_service.ALGORITHM_NAME,
    }

    with (
        audio_enc_path.open("rb") as audio_file,
        csv_enc_path.open("rb") as csv_file,
        svl_enc_path.open("rb") as svl_file,
    ):
        files = {
            "wav_file_enc": ("audio.wav.enc", audio_file, "application/octet-stream"),
            "csv_file_enc": ("data_all.csv.enc", csv_file, "application/octet-stream"),
            "svl_file_enc": ("raw.SVL.enc", svl_file, "application/octet-stream"),
        }
        response = requests.post(url, data=data, files=files, timeout=1800)

    if response.status_code >= 400:
        raise RuntimeError(
            f"Ana backend şifreli upload hatası: HTTP {response.status_code}\n"
            f"{response.text}"
        )

    result = response.json()
    if result.get("encryption_status") != "encrypted":
        raise RuntimeError(f"Backend encrypted durumu döndürmedi: {result}")

    return result


def read_live_once() -> dict:
    output = run_tool(["tools/svantek_live_once.py"], timeout=30)
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Canlı ölçüm JSON parse edilemedi:\n{output}") from exc


def update_threshold_trigger_from_live(data: dict) -> None:
    # Tetikleme:
    #   Kısa süreli anlık SPL eşik üzerinde trigger_hold_sec kadar kalır.
    #
    # Otomatik kapanış:
    #   Tetikleme oluştuktan sonra anlık SPL eşik altında release_hold_sec
    #   kadar kesintisiz kalırsa auto_stop_ready=True olur.
    global live_session

    if not live_session or live_session.get("mode") != "threshold_trigger":
        return

    live = data.get("live") if isinstance(data, dict) else None
    if not isinstance(live, dict) or not live.get("available"):
        return

    spl = live.get("spl")
    if not isinstance(spl, (int, float)):
        return

    threshold_db = float(live_session.get("threshold_db", 120.0))
    trigger_hold_sec = float(live_session.get("trigger_hold_sec", 1.0))
    release_hold_sec = float(live_session.get("release_hold_sec", 5.0))
    now_epoch = time.time()

    live_session["current_trigger_value_db"] = float(spl)
    live_session["trigger_source"] = "spl_short"

    if not live_session.get("triggered"):
        live_session["state"] = "armed"

        if float(spl) >= threshold_db:
            started_epoch = live_session.get("_over_threshold_started_epoch")
            if not isinstance(started_epoch, (int, float)):
                started_epoch = now_epoch
                live_session["_over_threshold_started_epoch"] = started_epoch
                live_session["over_threshold_started_at"] = datetime.now().isoformat()

            elapsed = max(0.0, now_epoch - float(started_epoch))
            live_session["over_threshold_elapsed_sec"] = round(elapsed, 3)

            if elapsed >= trigger_hold_sec:
                live_session["triggered"] = True
                live_session["state"] = "recording"
                live_session["triggered_at"] = datetime.now().isoformat()
                live_session["trigger_value_db"] = float(spl)
                live_session["over_threshold_elapsed_sec"] = round(trigger_hold_sec, 3)
                live_session["below_threshold_elapsed_sec"] = 0.0
                live_session["below_threshold_started_at"] = None
                live_session["auto_stop_ready"] = False
        else:
            live_session.pop("_over_threshold_started_epoch", None)
            live_session["over_threshold_started_at"] = None
            live_session["over_threshold_elapsed_sec"] = 0.0

        return

    if float(spl) < threshold_db:
        live_session["state"] = "release_wait"

        below_started_epoch = live_session.get("_below_threshold_started_epoch")
        if not isinstance(below_started_epoch, (int, float)):
            below_started_epoch = now_epoch
            live_session["_below_threshold_started_epoch"] = below_started_epoch
            live_session["below_threshold_started_at"] = datetime.now().isoformat()

        below_elapsed = max(0.0, now_epoch - float(below_started_epoch))
        live_session["below_threshold_elapsed_sec"] = round(below_elapsed, 3)

        if below_elapsed >= release_hold_sec:
            live_session["auto_stop_ready"] = True
            live_session["state"] = "auto_stop_ready"
            live_session["below_threshold_elapsed_sec"] = round(release_hold_sec, 3)
    else:
        live_session.pop("_below_threshold_started_epoch", None)
        live_session["below_threshold_started_at"] = None
        live_session["below_threshold_elapsed_sec"] = 0.0
        live_session["auto_stop_ready"] = False
        live_session["state"] = "recording"


def _public_session_copy(session: dict | None) -> dict | None:
    if not session:
        return None

    result = dict(session)
    result.pop("_over_threshold_started_epoch", None)
    result.pop("_below_threshold_started_epoch", None)
    result.pop("_rearm_deadline_epoch", None)
    result.pop("rearm_config", None)
    return result


def stop_threshold_monitor(wait: bool = False) -> None:
    global threshold_monitor_thread

    threshold_monitor_stop_event.set()
    thread = threshold_monitor_thread

    if (
        wait
        and thread is not None
        and thread.is_alive()
        and thread is not threading.current_thread()
    ):
        thread.join(timeout=2.0)

    if thread is not None and not thread.is_alive():
        threshold_monitor_thread = None


def threshold_monitor_loop() -> None:
    global active_session, live_session, latest_live_snapshot, threshold_monitor_thread

    try:
        while not threshold_monitor_stop_event.is_set():
            session = live_session
            if not session or not session.get("active"):
                return

            mode = session.get("mode")

            # Bir kayıt backend tarafından alındıktan sonra cihazı hemen tekrar
            # çalıştırmıyoruz. Kullanıcının belirlediği cooldown boyunca bekleyip
            # aynı setup ile yeni bir logger oturumu başlatıyoruz.
            if mode == "threshold_rearm":
                deadline = session.get("_rearm_deadline_epoch")
                if not isinstance(deadline, (int, float)):
                    cooldown = float(session.get("rearm_cooldown_sec", 30.0))
                    deadline = time.time() + cooldown
                    session["_rearm_deadline_epoch"] = deadline

                remaining = max(0.0, float(deadline) - time.time())
                session["cooldown_remaining_sec"] = round(remaining, 1)

                latest_live_snapshot = {
                    "edge_id": EDGE_ID,
                    "status": {"measurement_state": "stop"},
                    "live": {
                        "available": False,
                        "error": "Kayıt aktarıldı; yeniden eşik izleme cooldown'ı devam ediyor.",
                    },
                    "live_session": _public_session_copy(session),
                }

                if remaining > 0:
                    threshold_monitor_stop_event.wait(min(0.5, remaining))
                    continue

                session["state"] = "rearming"
                session["cooldown_remaining_sec"] = 0.0
                session["rearming_started_at"] = datetime.now().isoformat()

                latest_live_snapshot = {
                    "edge_id": EDGE_ID,
                    "status": {"measurement_state": "stop"},
                    "live": {
                        "available": False,
                        "error": "SVAN yeniden eşik izleme için hazırlanıyor.",
                    },
                    "live_session": _public_session_copy(session),
                }

                config = dict(session.get("rearm_config") or {})
                setup_name = str(config.get("setup_name") or session.get("setup_name") or "").strip()

                try:
                    rearm_outputs = {}
                    rearm_outputs["stop_before_setup"] = control_no_response("#1,S0", timeout=10)
                    if threshold_monitor_stop_event.is_set():
                        return

                    time.sleep(0.25)
                    rearm_outputs["load_setup"] = control_read(f"#7,LS,{setup_name}", timeout=30)
                    if threshold_monitor_stop_event.is_set():
                        return

                    rearm_outputs["logger_on"] = control_no_response("#1,T1", timeout=10)
                    if threshold_monitor_stop_event.is_set():
                        return

                    rearm_outputs["start"] = control_no_response("#1,S1", timeout=10)
                    if threshold_monitor_stop_event.is_set():
                        return

                except Exception as exc:
                    if live_session is not None:
                        live_session["state"] = "error"
                        live_session["active"] = False
                        live_session["rearm_error"] = str(exc)
                        live_session["completed_at"] = datetime.now().isoformat()

                    latest_live_snapshot = {
                        "edge_id": EDGE_ID,
                        "status": {"measurement_state": "unknown"},
                        "live": {
                            "available": False,
                            "error": f"Eşik izleme yeniden başlatılamadı: {exc}",
                        },
                        "live_session": _public_session_copy(live_session),
                    }
                    return

                cycle_index = int(session.get("next_cycle_index") or 1)
                completed_count = int(session.get("completed_count") or 0)
                workflow_started_at = (
                    session.get("workflow_started_at")
                    or session.get("started_at")
                    or datetime.now().isoformat()
                )
                last_recording = session.get("last_recording")
                last_logger_name = session.get("last_logger_name")

                threshold_db = float(config.get("threshold_db", 120.0))
                trigger_hold_sec = float(config.get("trigger_hold_sec", 1.0))
                release_hold_sec = float(config.get("release_hold_sec", 5.0))
                rearm_cooldown_sec = float(config.get("rearm_cooldown_sec", 30.0))

                active_session = {
                    "started_at": datetime.now().isoformat(),
                    "folder": config.get("folder") or "Otomatik",
                    "tag": config.get("tag") or f"threshold-{threshold_db:g}db",
                    "color": config.get("color") or "#ef4444",
                    "title": config.get("title") or f"Otomatik {threshold_db:g} dB",
                    "automatic_trigger": True,
                    "threshold_db": threshold_db,
                    "trigger_hold_sec": trigger_hold_sec,
                    "release_hold_sec": release_hold_sec,
                    "auto_rearm": True,
                    "rearm_cooldown_sec": rearm_cooldown_sec,
                    "setup_name": setup_name,
                    "cycle_index": cycle_index,
                }

                live_session = {
                    "active": True,
                    "started_at": datetime.now().isoformat(),
                    "workflow_started_at": workflow_started_at,
                    "mode": "threshold_trigger",
                    "state": "armed",
                    "threshold_db": threshold_db,
                    "trigger_hold_sec": trigger_hold_sec,
                    "release_hold_sec": release_hold_sec,
                    "auto_rearm": True,
                    "rearm_cooldown_sec": rearm_cooldown_sec,
                    "trigger_source": "spl_short",
                    "triggered": False,
                    "triggered_at": None,
                    "trigger_value_db": None,
                    "current_trigger_value_db": None,
                    "over_threshold_started_at": None,
                    "over_threshold_elapsed_sec": 0.0,
                    "below_threshold_started_at": None,
                    "below_threshold_elapsed_sec": 0.0,
                    "auto_stop_ready": False,
                    "setup_name": setup_name,
                    "cycle_index": cycle_index,
                    "completed_count": completed_count,
                    "last_recording": last_recording,
                    "last_logger_name": last_logger_name,
                    "rearmed_at": datetime.now().isoformat(),
                    "rearm_outputs": rearm_outputs,
                    "note": (
                        "Döngü modu: kayıt backend tarafından alındıktan ve cooldown "
                        "tamamlandıktan sonra aynı eşik setup'ı yeniden başlatıldı."
                    ),
                }

                time.sleep(0.5)
                first_data = read_live_once()
                update_threshold_trigger_from_live(first_data)
                first_data["edge_id"] = EDGE_ID
                first_data["live_session"] = _public_session_copy(live_session)
                latest_live_snapshot = first_data
                continue

            if mode != "threshold_trigger":
                return

            try:
                data = read_live_once()
                update_threshold_trigger_from_live(data)

                data["edge_id"] = EDGE_ID
                data["live_session"] = _public_session_copy(live_session)
                latest_live_snapshot = data

                if live_session and live_session.get("auto_stop_ready"):
                    live_session["state"] = "finalizing"
                    live_session["finalizing_started_at"] = datetime.now().isoformat()

                    trigger_info = _public_session_copy(live_session) or {}
                    recording_config = dict(active_session or {})

                    try:
                        # Bu fonksiyon ancak backend upload cevabını aldıktan sonra döner.
                        # Dolayısıyla cooldown, kayıt ana backend tarafından alındıktan sonra başlar.
                        result = edge_stop_and_upload()
                    except Exception as exc:
                        if live_session is not None:
                            live_session["state"] = "error"
                            live_session["active"] = False
                            live_session["auto_finalize_error"] = str(exc)
                            live_session["completed_at"] = datetime.now().isoformat()

                        latest_live_snapshot = {
                            **data,
                            "live_session": _public_session_copy(live_session),
                        }
                        return

                    completed_count = int(trigger_info.get("completed_count") or 0) + 1
                    current_cycle = int(trigger_info.get("cycle_index") or 1)
                    auto_rearm = bool(trigger_info.get("auto_rearm"))
                    cooldown = float(trigger_info.get("rearm_cooldown_sec", 30.0))

                    if auto_rearm:
                        rearm_config = {
                            "folder": recording_config.get("folder") or "Otomatik",
                            "tag": recording_config.get("tag") or "",
                            "color": recording_config.get("color") or "#ef4444",
                            "title": recording_config.get("title") or "",
                            "threshold_db": float(trigger_info.get("threshold_db", 120.0)),
                            "trigger_hold_sec": float(trigger_info.get("trigger_hold_sec", 1.0)),
                            "release_hold_sec": float(trigger_info.get("release_hold_sec", 5.0)),
                            "auto_rearm": True,
                            "rearm_cooldown_sec": cooldown,
                            "setup_name": trigger_info.get("setup_name") or "",
                        }

                        rearm_session = {
                            "active": True,
                            "started_at": trigger_info.get("started_at"),
                            "workflow_started_at": (
                                trigger_info.get("workflow_started_at")
                                or trigger_info.get("started_at")
                            ),
                            "mode": "threshold_rearm",
                            "state": "cooldown",
                            "threshold_db": trigger_info.get("threshold_db"),
                            "trigger_hold_sec": trigger_info.get("trigger_hold_sec"),
                            "release_hold_sec": trigger_info.get("release_hold_sec"),
                            "auto_rearm": True,
                            "rearm_cooldown_sec": cooldown,
                            "setup_name": trigger_info.get("setup_name"),
                            "triggered": False,
                            "auto_stop_ready": False,
                            "cycle_index": current_cycle,
                            "next_cycle_index": current_cycle + 1,
                            "completed_count": completed_count,
                            "last_recording": result.get("recording"),
                            "last_logger_name": result.get("logger_name"),
                            "last_completed_at": datetime.now().isoformat(),
                            "cooldown_total_sec": cooldown,
                            "cooldown_remaining_sec": cooldown,
                            "_rearm_deadline_epoch": time.time() + cooldown,
                            "rearm_config": rearm_config,
                            "note": (
                                "Kayıt backend tarafından alındı. Cooldown tamamlanınca "
                                "eşik izleme otomatik olarak yeniden başlayacak."
                            ),
                        }
                        live_session = rearm_session

                        latest_live_snapshot = {
                            "edge_id": EDGE_ID,
                            "status": {"measurement_state": "stop"},
                            "live": {
                                "available": False,
                                "error": "Kayıt aktarıldı; cooldown bekleniyor.",
                            },
                            "live_session": _public_session_copy(rearm_session),
                            "auto_upload": {
                                "status": "uploaded",
                                "recording": result.get("recording"),
                                "logger_name": result.get("logger_name"),
                            },
                        }
                        continue

                    completed_session = {
                        **trigger_info,
                        "active": False,
                        "mode": "threshold_complete",
                        "state": "uploaded",
                        "completed_at": datetime.now().isoformat(),
                        "completed_count": completed_count,
                        "recording": result.get("recording"),
                        "logger_name": result.get("logger_name"),
                        "auto_stop_ready": False,
                    }
                    live_session = completed_session

                    latest_live_snapshot = {
                        **data,
                        "live_session": _public_session_copy(completed_session),
                        "auto_upload": {
                            "status": "uploaded",
                            "recording": result.get("recording"),
                            "logger_name": result.get("logger_name"),
                        },
                    }
                    return

            except Exception as exc:
                if live_session is not None:
                    live_session["monitor_error"] = str(exc)
                    live_session["monitor_error_at"] = datetime.now().isoformat()

            threshold_monitor_stop_event.wait(0.5)
    finally:
        threshold_monitor_thread = None


def start_threshold_monitor() -> None:
    global threshold_monitor_thread

    stop_threshold_monitor(wait=True)
    threshold_monitor_stop_event.clear()

    threshold_monitor_thread = threading.Thread(
        target=threshold_monitor_loop,
        name="svan-threshold-monitor",
        daemon=True,
    )
    threshold_monitor_thread.start()


def parse_control_state(output: str | None) -> dict:
    output = output or ""

    # svantek_control.py çıktısında hem gönderilen komut hem cevap bulunur:
    #   Gönderiliyor: #1,S?,T?;
    #   Cevap: #1,S0,T1;
    # Eski kod ilk #1,...; eşleşmesini aldığı için #1,S?,T?; satırını state sanıyor
    # ve cihaz durmuş olsa bile measurement_state=unknown/running hatasına düşebiliyordu.
    # Burada mümkünse son gerçek cevap satırını, yoksa içinde sayısal S/T değeri olan son #1 cevabını seçiyoruz.
    response_matches = re.findall(r"Cevap:\s*(#1,[^;]+;)", output)
    if response_matches:
        raw = response_matches[-1].strip()
    else:
        all_matches = [m.strip() for m in re.findall(r"#1,[^;]+;", output)]
        numeric_matches = [m for m in all_matches if re.search(r"S[0-2]", m) or re.search(r"T[0-1]", m)]
        raw = (numeric_matches[-1] if numeric_matches else (all_matches[-1] if all_matches else output.strip()))

    state_match = re.search(r"S([0-2])", raw)
    logger_match = re.search(r"T([0-1])", raw)

    return {
        "raw": raw,
        "measurement_state": {
            "0": "stop",
            "1": "running",
            "2": "pause",
        }.get(state_match.group(1), "unknown") if state_match else "unknown",
        "logger_state": {
            "0": "off",
            "1": "on",
        }.get(logger_match.group(1), "unknown") if logger_match else "unknown",
    }


def control_no_response(cmd: str, timeout: int = 15) -> str:
    return run_tool(["tools/svantek_control.py", "--cmd", cmd, "--no-response"], timeout=timeout)


def control_read(cmd: str, timeout: int = 20) -> str:
    return run_tool(["tools/svantek_control.py", "--cmd", cmd], timeout=timeout)


def read_control_state() -> dict:
    output = control_read("#1,S?,T?", timeout=20)
    return parse_control_state(output)


def stop_live_mode() -> dict:
    """
    Canlı izleme modu cihazı S1 durumuna aldığı için kapatırken:
    1) Frontend polling'i durdurabilsin diye live_session'ı hemen kapat
    2) S0 komutunu cevap beklemeden hızlı gönder
    3) S?,T? ile gerçekten stop oldu mu doğrula
    4) T1 ile logger/kayıt davranışını geri aç

    Eski sürümde S0/T1 için cevap bekleniyordu. Cihaz bu komutlara çoğu zaman cevap
    vermediği için stop 6+ saniye gecikiyor ve polling ile USB çakışması yaşanabiliyordu.
    """
    global live_session

    stop_threshold_monitor(wait=False)

    old_session = live_session
    live_session = None

    outputs = {}
    errors = []
    final_state = None

    # S0 birkaç kez denenir; her denemeden sonra cihaz state'i okunur.
    for attempt in range(1, 4):
        try:
            outputs[f"stop_attempt_{attempt}"] = control_no_response("#1,S0", timeout=10)
        except Exception as exc:
            outputs[f"stop_attempt_{attempt}"] = str(exc)
            errors.append(str(exc))

        time.sleep(0.35)

        try:
            final_state = read_control_state()
            outputs[f"state_after_stop_{attempt}"] = final_state
            if final_state.get("measurement_state") == "stop":
                break
        except Exception as exc:
            outputs[f"state_after_stop_{attempt}"] = str(exc)
            errors.append(str(exc))

    try:
        outputs["logger_on"] = control_no_response("#1,T1", timeout=10)
    except Exception as exc:
        outputs["logger_on"] = str(exc)
        errors.append(str(exc))

    time.sleep(0.25)

    try:
        final_state = read_control_state()
        outputs["final_state"] = final_state
    except Exception as exc:
        outputs["final_state"] = str(exc)
        errors.append(str(exc))

    stopped = isinstance(final_state, dict) and final_state.get("measurement_state") == "stop"

    return {
        "status": "stopped" if stopped else "stop_requested_but_still_running",
        "stopped": stopped,
        "old_session": old_session,
        "final_state": final_state,
        "outputs": outputs,
        "errors": errors,
    }


@app.get("/api/edge/status")
def edge_status():
    status_output = None
    svan_ok = False
    svan_error = None

    try:
        status_output = run_tool(["tools/svantek_control.py", "--status"], timeout=30)
        svan_ok = True
    except Exception as exc:
        svan_error = str(exc)

    return {
        "status": "ok",
        "edge_id": EDGE_ID,
        "main_backend_url": MAIN_BACKEND_URL,
        "work_dir": str(EDGE_WORK_DIR),
        "delete_after_upload": DELETE_AFTER_UPLOAD,
        "sample_rate": SVAN_SAMPLE_RATE,
        "pi_side_encryption_required": EDGE_ENCRYPTION_REQUIRED,
        "encryption_key_configured": bool(os.getenv("AES_KEY_B64", "").strip()),
        "encryption_algorithm": (encryption_service.ALGORITHM_NAME if encryption_service is not None else None),
        "manual_record_setup": MANUAL_RECORD_SETUP,
        "active_session": active_session,
        "live_session": live_session,
        "svan_ok": svan_ok,
        "svan_status_output": status_output,
        "svan_error": svan_error,
    }


@app.get("/api/edge/live/status")
def edge_live_status():
    status = None
    measurement_state = None

    try:
        data = read_live_once()
        status = data.get("status")
        measurement_state = status.get("measurement_state") if isinstance(status, dict) else None
    except Exception as exc:
        return {
            "status": "error",
            "edge_id": EDGE_ID,
            "active": bool(live_session and live_session.get("active")),
            "live_session": live_session,
            "error": str(exc),
        }

    return {
        "status": "ok",
        "edge_id": EDGE_ID,
        "active": bool(live_session and live_session.get("active")),
        "live_session": live_session,
        "measurement_state": measurement_state,
        "svan_status": status,
    }


@app.post("/api/edge/live/start")
def edge_live_start(payload: LiveStartRequest):
    global live_session, active_session, latest_live_snapshot

    if active_session is not None:
        raise HTTPException(status_code=409, detail="Aktif kayıt varken canlı izleme başlatılamaz.")

    if live_session is not None and live_session.get("active"):
        return {
            "status": "already_running",
            "edge_id": EDGE_ID,
            "live_session": _public_session_copy(live_session),
            "latest": latest_live_snapshot or read_live_once(),
        }

    if live_session is not None and not live_session.get("active"):
        live_session = None

    outputs = {}

    try:
        if payload.threshold_enabled:
            _ensure_pi_encryption_ready()
            threshold_db = float(payload.threshold_db)
            if not 24 <= threshold_db <= 136:
                raise HTTPException(status_code=400, detail="Eşik değeri 24 ile 136 dB arasında olmalıdır.")

            trigger_hold_sec = float(payload.trigger_hold_sec)
            if not 0.5 <= trigger_hold_sec <= 30:
                raise HTTPException(
                    status_code=400,
                    detail="Tetikleme süresi 0.5 ile 30 saniye arasında olmalıdır.",
                )

            release_hold_sec = float(payload.release_hold_sec)
            if not 0.5 <= release_hold_sec <= 300:
                raise HTTPException(
                    status_code=400,
                    detail="Otomatik kapanış süresi 0.5 ile 300 saniye arasında olmalıdır.",
                )

            auto_rearm = bool(payload.auto_rearm)
            rearm_cooldown_sec = float(payload.rearm_cooldown_sec)
            if auto_rearm and not 5 <= rearm_cooldown_sec <= 900:
                raise HTTPException(
                    status_code=400,
                    detail="Yeniden başlatma cooldown süresi 5 ile 900 saniye arasında olmalıdır.",
                )

            setup_name = (payload.setup_name or "").strip()
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,8}", setup_name):
                raise HTTPException(
                    status_code=400,
                    detail="SVAN setup adı 1-8 karakter olmalı; yalnızca harf, rakam, _ ve - kullanılabilir.",
                )

            # Setup cihaz üzerinde önceden oluşturulmuş olmalı:
            # Event Recording = Level+, Level = threshold_db, Pre Trigger = 1 s.
            outputs["stop_before_setup"] = control_no_response("#1,S0", timeout=10)
            time.sleep(0.25)
            outputs["load_setup"] = control_read(f"#7,LS,{setup_name}", timeout=30)
            outputs["logger_on"] = control_no_response("#1,T1", timeout=10)
            outputs["start"] = control_no_response("#1,S1", timeout=10)

            active_session = {
                "started_at": datetime.now().isoformat(),
                "folder": payload.folder or "Otomatik",
                "tag": payload.tag or f"threshold-{threshold_db:g}db",
                "color": payload.color or "#ef4444",
                "title": payload.title or f"Otomatik {threshold_db:g} dB",
                "automatic_trigger": True,
                "threshold_db": threshold_db,
                "trigger_hold_sec": trigger_hold_sec,
                "release_hold_sec": release_hold_sec,
                "auto_rearm": auto_rearm,
                "rearm_cooldown_sec": rearm_cooldown_sec,
                "setup_name": setup_name,
                "cycle_index": 1,
            }

            workflow_started_at = datetime.now().isoformat()
            live_session = {
                "active": True,
                "started_at": workflow_started_at,
                "workflow_started_at": workflow_started_at,
                "mode": "threshold_trigger",
                "state": "armed",
                "threshold_db": threshold_db,
                "trigger_hold_sec": trigger_hold_sec,
                "release_hold_sec": release_hold_sec,
                "auto_rearm": auto_rearm,
                "rearm_cooldown_sec": rearm_cooldown_sec,
                "trigger_source": "spl_short",
                "triggered": False,
                "triggered_at": None,
                "trigger_value_db": None,
                "current_trigger_value_db": None,
                "over_threshold_started_at": None,
                "over_threshold_elapsed_sec": 0.0,
                "below_threshold_started_at": None,
                "below_threshold_elapsed_sec": 0.0,
                "auto_stop_ready": False,
                "setup_name": setup_name,
                "cycle_index": 1,
                "completed_count": 0,
                "last_recording": None,
                "last_logger_name": None,
                "note": (
                    "SVAN T1 + S1/Event Recording modunda çalışır. Web tetik durumu "
                    "uzun süreli Leq yerine kısa süreli anlık SPL üzerinden hesaplanır."
                ),
            }
        else:
            # Yalnızca canlı izleme: logger kapalı, dosya web sistemine aktarılmaz.
            outputs["logger_off"] = control_no_response("#1,T0", timeout=10)
            outputs["start"] = control_no_response("#1,S1", timeout=10)

            live_session = {
                "active": True,
                "started_at": datetime.now().isoformat(),
                "mode": "active_measurement_no_upload",
                "note": "T0 + S1 ile canlı dB okuma. Web kaydı oluşturulmaz.",
            }

        time.sleep(0.5)
        latest = read_live_once()
        update_threshold_trigger_from_live(latest)

        if payload.threshold_enabled:
            latest["edge_id"] = EDGE_ID
            latest["live_session"] = _public_session_copy(live_session)
            latest_live_snapshot = latest
            start_threshold_monitor()

        return {
            "status": "running",
            "edge_id": EDGE_ID,
            "live_session": live_session,
            "outputs": outputs,
            "latest": latest,
        }
    except HTTPException:
        active_session = None
        live_session = None
        raise
    except Exception as exc:
        active_session = None
        try:
            stop_live_mode()
        except Exception:
            live_session = None
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/edge/live/stop")
def edge_live_stop(payload: LiveStopRequest):
    global active_session, live_session

    try:
        if live_session and live_session.get("state") == "finalizing":
            return {
                "status": "finalizing",
                "edge_id": EDGE_ID,
                "live_session": _public_session_copy(live_session),
            }

        if live_session and live_session.get("state") == "rearming":
            return {
                "status": "rearming",
                "edge_id": EDGE_ID,
                "live_session": _public_session_copy(live_session),
            }

        if live_session and live_session.get("mode") == "threshold_complete":
            return {
                "status": live_session.get("state", "uploaded"),
                "edge_id": EDGE_ID,
                "live_session": _public_session_copy(live_session),
                "recording": live_session.get("recording"),
            }

        if live_session and live_session.get("mode") == "threshold_rearm":
            old_session = _public_session_copy(live_session)
            stop_threshold_monitor(wait=True)
            active_session = None
            live_session = None
            return {
                "status": "loop_cancelled",
                "stopped": True,
                "edge_id": EDGE_ID,
                "old_session": old_session,
            }

        stop_threshold_monitor(wait=True)

        is_threshold_mode = bool(
            live_session and live_session.get("mode") == "threshold_trigger"
        )

        if is_threshold_mode and live_session.get("triggered"):
            trigger_info = _public_session_copy(live_session) or {}
            result = edge_stop_and_upload()
            live_session = None
            result["edge_id"] = EDGE_ID
            result["trigger_session"] = trigger_info
            result["status"] = "uploaded"
            return result

        if is_threshold_mode:
            active_session = None

        result = stop_live_mode()
        result["edge_id"] = EDGE_ID
        if is_threshold_mode:
            result["status"] = "trigger_cancelled"
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/edge/live/once")
def edge_live_once():
    if live_session is not None and live_session.get("mode") in {
        "threshold_trigger",
        "threshold_rearm",
        "threshold_complete",
    }:
        if latest_live_snapshot is not None:
            data = dict(latest_live_snapshot)
        else:
            data = {
                "edge_id": EDGE_ID,
                "status": {"measurement_state": "unknown"},
                "live": {
                    "available": False,
                    "error": "İlk threshold ölçümü bekleniyor.",
                },
            }

        data["edge_id"] = EDGE_ID
        data["live_session"] = _public_session_copy(live_session)
        return data

    if live_session is None:
        try:
            state = read_control_state()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        return {
            "edge_id": EDGE_ID,
            "live_session": None,
            "status": state,
            "live": {
                "available": False,
                "error": "Canlı izleme kapalı.",
            },
        }

    try:
        data = read_live_once()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    data["edge_id"] = EDGE_ID
    data["live_session"] = _public_session_copy(live_session)
    return data


@app.post("/api/edge/start")
def edge_start(payload: StartRequest):
    global active_session

    try:
        _ensure_pi_encryption_ready()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    if active_session is not None:
        raise HTTPException(status_code=409, detail="Edge üzerinde zaten aktif kayıt var.")

    # Canlı izleme açıksa kayıt öncesi otomatik kapat.
    if live_session is not None:
        stop_live_mode()
        time.sleep(0.5)

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,8}", MANUAL_RECORD_SETUP):
        raise HTTPException(
            status_code=500,
            detail="MANUAL_RECORD_SETUP 1-8 karakter olmalıdır.",
        )

    outputs = {}

    try:
        outputs["stop_before_setup"] = control_no_response("#1,S0", timeout=10)
        time.sleep(0.25)
        outputs["load_setup"] = control_read(
            f"#7,LS,{MANUAL_RECORD_SETUP}",
            timeout=30,
        )
        outputs["logger_on"] = control_no_response("#1,T1", timeout=10)
        outputs["start"] = control_no_response("#1,S1", timeout=10)
        time.sleep(0.5)

        final_state = read_control_state()
        outputs["final_state"] = final_state

        if final_state.get("measurement_state") != "running":
            raise RuntimeError(
                f"Ölçüm running durumuna geçmedi: {final_state}"
            )
        if final_state.get("logger_state") != "on":
            raise RuntimeError(
                f"Logger açık görünmüyor: {final_state}"
            )
    except Exception as exc:
        active_session = None
        try:
            control_no_response("#1,S0", timeout=10)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=str(exc))

    active_session = {
        "started_at": datetime.now().isoformat(),
        "folder": payload.folder or "Genel",
        "tag": payload.tag or "",
        "color": payload.color or "#f2a65a",
        "title": payload.title or "",
        "setup_name": MANUAL_RECORD_SETUP,
        "automatic_trigger": False,
        "start_output": outputs.get("start"),
        "start_outputs": outputs,
    }

    return {
        "status": "recording",
        "edge_id": EDGE_ID,
        "message": f"SVAN kaydı {MANUAL_RECORD_SETUP} setup'ı ile başladı.",
        "setup_name": MANUAL_RECORD_SETUP,
        "session": active_session,
        "outputs": outputs,
    }


@app.post("/api/edge/stop-and-upload")
def edge_stop_and_upload():
    global active_session

    if active_session is None:
        raise HTTPException(status_code=409, detail="Aktif kayıt yok.")

    session = active_session

    try:
        stop_output = run_tool(["tools/svantek_control.py", "--stop"], timeout=90)
        logger_name = parse_last_logger_name(stop_output)

        ls_output = run_tool(["tools/svantek_ls.py"], timeout=120)
        svl_info = parse_svl_from_ls(ls_output, logger_name)

        session_name = safe_session_name(logger_name)
        session_dir = EDGE_WORK_DIR / session_name
        session_dir.mkdir(parents=True, exist_ok=True)

        raw_svl_path = session_dir / "raw.SVL"
        audio_path = session_dir / "audio.wav"
        csv_path = session_dir / "data_all.csv"

        download_output = run_tool(
            [
                "tools/svantek_download_file.py",
                "--cluster", str(svl_info["cluster"]),
                "--size", str(svl_info["size"]),
                "--output", str(raw_svl_path),
            ],
            timeout=1800,
        )

        wav_output = run_tool(
            [
                "tools/svl_extract_wav_raw24.py",
                str(raw_svl_path),
                "--output", str(audio_path),
                "--rate", str(SVAN_SAMPLE_RATE),
            ],
            timeout=600,
        )

        csv_output = run_tool(
            [
                "tools/svl_extract_all_csv.py",
                str(raw_svl_path),
                "--output", str(csv_path),
                "--record-id", logger_name,
                "--rate", str(SVAN_SAMPLE_RATE),
            ],
            timeout=900,
        )

        title = session.get("title") or session_name

        upload_result = upload_recording_to_main_backend(
            session_dir=session_dir,
            title=title,
            folder=session.get("folder") or "Genel",
            tag=session.get("tag") or "",
            color=session.get("color") or "#f2a65a",
            logger_name=logger_name,
        )

        if DELETE_AFTER_UPLOAD:
            shutil.rmtree(session_dir, ignore_errors=True)

        active_session = None

        return {
            "status": "uploaded",
            "edge_id": EDGE_ID,
            "logger_name": logger_name,
            "svl_info": svl_info,
            "recording": upload_result,
            "outputs": {
                "stop": stop_output,
                "ls": ls_output,
                "download": download_output,
                "wav": wav_output,
                "csv": csv_output,
            },
        }

    except Exception as exc:
        # Hata olursa active_session'ı silmiyoruz.
        # Böylece tekrar stop-and-upload denenebilir.
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/edge/reset-session")
def edge_reset_session():
    global active_session
    old = active_session
    active_session = None
    return {
        "status": "reset",
        "old_session": old,
    }
