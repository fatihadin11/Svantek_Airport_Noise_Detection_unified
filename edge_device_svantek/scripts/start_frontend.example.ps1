$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if ((Split-Path -Leaf $ScriptDir) -eq "scripts") {
  $ProjectRoot = Split-Path -Parent $ScriptDir
} else {
  $ProjectRoot = $ScriptDir
}

Set-Location "$ProjectRoot\frontend"

# Bu dosyayı start_frontend.local.ps1 adıyla kopyalayabilirsiniz.
# Frontend, Vite geliştirme sunucusunu başlatır.
npm run dev
