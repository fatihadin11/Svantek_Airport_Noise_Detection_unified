import argparse
import json
import re
import sys
import time

import usb.core
import usb.util

VID = 0x0017
PID = 0x0001
OUT_ENDPOINT = 0x01
IN_ENDPOINT = 0x83
INTERFACE = 0

LIVE_COMMAND = "#2,i,1,S?,R?,P?,M?,N?,T?,V?,v?;"


class SvantekUSB:
    def __init__(self):
        self.dev = usb.core.find(idVendor=VID, idProduct=PID)

        if self.dev is None:
            raise RuntimeError("SVAN 971 bulunamadı. VirtualBox USB passthrough kontrol et.")

        try:
            self.dev.set_configuration()
        except usb.core.USBError:
            pass

        try:
            if self.dev.is_kernel_driver_active(INTERFACE):
                self.dev.detach_kernel_driver(INTERFACE)
        except Exception:
            pass

        try:
            usb.util.claim_interface(self.dev, INTERFACE)
        except usb.core.USBError as exc:
            raise RuntimeError(
                f"USB interface kullanılıyor: {exc}\n"
                "Çözüm: VirtualBox USB menüsünden 0017:0001 cihazını çıkarıp tekrar seç veya SVAN USB kablosunu çıkar-tak yap."
            ) from exc

    def close(self):
        try:
            usb.util.release_interface(self.dev, INTERFACE)
        except Exception:
            pass
        try:
            usb.util.dispose_resources(self.dev)
        except Exception:
            pass

    def send(self, command: str, timeout: int = 3000, delay: float = 0.15) -> str | None:
        if not command.endswith(";"):
            command += ";"

        self.dev.write(OUT_ENDPOINT, command.encode("ascii"), timeout=timeout)
        time.sleep(delay)

        chunks = []
        deadline = time.time() + (timeout / 1000.0)

        while time.time() < deadline:
            try:
                data = self.dev.read(IN_ENDPOINT, 1024, timeout=350)
                part = bytes(data)
                if part:
                    chunks.append(part)
                    if b";" in part:
                        break
            except usb.core.USBTimeoutError:
                if chunks:
                    break
                return None

        if not chunks:
            return None

        return b"".join(chunks).decode("ascii", errors="replace").strip()


def parse_status(raw: str | None) -> dict:
    out = {"raw": raw}
    if not raw:
        out["measurement_state"] = "unknown"
        return out

    state_match = re.search(r"S([0-2])", raw)
    if state_match:
        state_code = state_match.group(1)
        out["measurement_state"] = {
            "0": "stop",
            "1": "running",
            "2": "pause",
        }.get(state_code, state_code)

    logger_match = re.search(r"T([0-1])", raw)
    if logger_match:
        out["logger_state"] = "on" if logger_match.group(1) == "1" else "off"

    return out


def parse_live(raw: str | None, command: str) -> dict:
    out = {
        "command": command,
        "raw": raw,
        "available": False,
    }

    if not raw:
        out["error"] = "Cevap gelmedi."
        return out

    if raw.startswith("#2,?"):
        out["error"] = "Cihaz #2 sonucu döndürmedi."
        return out

    out["available"] = True

    patterns = {
        "time_sec": r",T(-?\d+(?:\.\d+)?)",
        "lpeak": r",P(-?\d+(?:\.\d+)?)",
        "lmax": r",M(-?\d+(?:\.\d+)?)",
        "lmin": r",N(-?\d+(?:\.\d+)?)",
        "spl": r",S(-?\d+(?:\.\d+)?)",
        "leq": r",R(-?\d+(?:\.\d+)?)",
        "overload": r",V([01])",
        "underrange": r",v([0-3])",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, raw)
        if not match:
            continue

        value = match.group(1)

        if key == "overload":
            out[key] = bool(int(value))
        elif key == "underrange":
            out[key] = int(value)
        else:
            out[key] = float(value)

    return out


def main():
    parser = argparse.ArgumentParser(description="SVAN 971 #2 canlı SLM sonucu için tek okuma testi.")
    parser.add_argument("--cmd", default=LIVE_COMMAND)
    parser.add_argument("--status-cmd", default="#1,S?,T?")
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()

    svan = SvantekUSB()

    try:
        status_raw = svan.send(args.status_cmd)
        live_raw = svan.send(args.cmd)

        if args.raw:
            print(status_raw or "")
            print(live_raw or "")
            return

        result = {
            "status": parse_status(status_raw),
            "live": parse_live(live_raw, args.cmd),
            "fallback": None,
            "note": "Bu okuma tek başına kayıt upload etmez. Gerçek canlı güncelleme için cihaz running durumda olmalıdır.",
        }

        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        svan.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        sys.exit(1)
