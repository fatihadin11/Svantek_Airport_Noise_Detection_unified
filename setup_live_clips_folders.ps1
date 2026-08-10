# ============================================================
# setup_live_clips_folders.ps1
# D:\Airport_Live_Clips klasör yapısını YENİ sınıf taksonomisine
# göre yeniden oluşturur (9 aktif + OTHER = 10 sınıf).
#
# NOT: Bu script eski klasörleri SİLMEZ, sadece yenilerini oluşturur.
# Eski taksonomiyle toplanmış klipler (varsa) D:\Airport_Live_Clips
# altında AIRCRAFT\, AMBIENT\ vb. eski klasörlerde kalmaya devam eder;
# "eski örneklerle işimiz kalmadı" dediğin için bunlara dokunmadım —
# istersen elle silebilirsin.
#
# Çalıştırma: PowerShell'i yönetici olarak aç, bu dosyanın olduğu
# klasörde:  .\setup_live_clips_folders.ps1
# ============================================================

$base = "D:\Airport_Live_Clips"

$classes = @(
    "JET_AIRCRAFT",
    "HELICOPTER",
    "APU_GSE",
    "WIND",
    "PRECIPITATION",
    "NATURE",
    "TRAFFIC",
    "SIREN_ALARM",
    "SPEECH",
    "OTHER"
)

foreach ($folder in "pending", "approved", "rejected") {
    foreach ($cls in $classes) {
        New-Item -ItemType Directory -Force -Path "$base\$folder\$cls" | Out-Null
    }
}

Write-Host "Tamamlandi: $base altinda pending/approved/rejected x 10 sinif klasoru olusturuldu." -ForegroundColor Green
Write-Host ""
Write-Host "Not: gui_main.py::PendingClipManager klasörleri zaten dinamik olarak"
Write-Host "kendisi de oluşturuyor (os.makedirs(..., exist_ok=True)) — bu script"
Write-Host "sadece GUI'yi ilk çalıştırmadan önce yapıyı gözle görmek/hazırlamak içindir."
