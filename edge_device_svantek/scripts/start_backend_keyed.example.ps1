$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ((Split-Path -Leaf $ScriptDir) -eq "scripts") {
  $ProjectRoot = Split-Path -Parent $ScriptDir
} else {
  $ProjectRoot = $ScriptDir
}

Set-Location "$ProjectRoot\backend"

# Bu dosyayı start_backend_keyed.local.ps1 adıyla kopyalayın.
# .local.ps1 dosyaları Git tarafından yok sayılır.
#
# Pi edge agent ile aynı AES_KEY_B64 kullanılmalıdır.
$env:AES_KEY_B64 = "PASTE_THE_SAME_AES_KEY_B64_HERE"

# Raspberry Pi örneği: http://192.168.1.35:8010
$env:EDGE_BASE_URL = "http://PI_OR_EDGE_IP:8010"

..\.venv\Scripts\python.exe -m uvicorn `
  main:app `
  --host 0.0.0.0 `
  --port 8000
