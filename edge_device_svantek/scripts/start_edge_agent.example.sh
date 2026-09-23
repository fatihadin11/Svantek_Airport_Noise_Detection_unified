#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$(basename "$SCRIPT_DIR")" = "scripts" ]; then
  PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
else
  PROJECT_ROOT="$SCRIPT_DIR"
fi

cd "$PROJECT_ROOT"

# Bu dosyayı start_edge_agent.local.sh adıyla kopyalayın.
# .local.sh dosyaları Git tarafından yok sayılır; gerçek IP ve AES anahtarını
# yalnızca yerel kopyada tutun.
#
# WINDOWS_IP örneği: 192.168.1.20
# AES_KEY_B64, Windows backend ile birebir aynı 32-byte base64 anahtar olmalıdır.

sudo env   MAIN_BACKEND_URL="http://WINDOWS_IP:8000"   EDGE_ID="raspberry-pi-svan-edge"   EDGE_WORK_DIR="/tmp/pi-ses-sistemi-edge"   DELETE_AFTER_UPLOAD="1"   SVAN_SAMPLE_RATE="8000"   MANUAL_RECORD_SETUP="RECORD"   EDGE_ENCRYPTION_REQUIRED="1"   AES_KEY_B64="PASTE_THE_SAME_AES_KEY_B64_HERE"   backend/venv/bin/python -m uvicorn     edge_agent.main:app     --host 0.0.0.0     --port 8010
