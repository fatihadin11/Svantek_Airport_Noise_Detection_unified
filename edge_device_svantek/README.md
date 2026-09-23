# SVANTEK Airport Noise - Unified System

SVAN 971 ile alınan kayıtları güvenli şekilde merkezi panele aktaran ve
kayıt tamamlandıktan sonra Airport Noise Detection modelleriyle inceleyen
deneysel sistem.

## Ne yapar?

1. Raspberry Pi üzerindeki edge agent, SVAN 971 kaydını yönetir.
2. Kayıt bitince WAV, CSV ve ham SVL dosyalarını AES-256-GCM ile şifreleyip
   backend'e gönderir.
3. Backend şifreli kaydı kalıcı olarak `.enc` biçiminde tutar.
4. Analiz sırasında ses yalnız geçici olarak açılır, mono/22.050 Hz'e
   dönüştürülür ve sınıflandırılır.
5. Panel, algılanan olayları zaman çizelgesinde gösterir; olaya tıklanınca
   kayıt ilgili saniyeye gider.

Panel ayrıca daha önce alınmış `audio.wav.enc` kayıtlarını doğrudan yükleyebilir.
Varsa `data_all.csv.enc` ve `raw.SVL.enc` dosyaları aynı seçimde eklenebilir.

## Mimari

```text
SVAN 971 -> Raspberry Pi edge agent -> encrypted upload -> FastAPI backend
                                                     -> SQLite + encrypted storage
                                                     -> Airport AI analysis
React/Vite panel <---------------------------------- events, audio, CSV, waveform
```

## Kurulum

Python 3.10 ve Node.js gerekir.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-unified.txt

cd frontend
npm ci
```

## Yerel yapılandırma ve çalıştırma

AES anahtarı hiçbir zaman Git'e eklenmez. Yerel başlangıç şablonunu kopyalayın,
`AES_KEY_B64` ve `EDGE_BASE_URL` değerlerini kendi ortamınıza göre düzenleyin:

```powershell
Copy-Item .\scripts\start_backend_keyed.example.ps1 .\start_backend_keyed.local.ps1
.\start_backend_keyed.local.ps1
```

İkinci terminalde:

```powershell
cd frontend
npm run dev
```

## AI modelleri

Model ağırlıkları ve eğitim verileri GitHub'a dahil edilmez. Kullanacağınız
model dosyalarını `airport_ai/models/` klasörüne yerleştirin; beklenen dosyalar
için [model klasörü notlarına](airport_ai/models/README.md) bakın.

Mevcut model dosyaları eski altı sınıflı taksonomiyi kullanır:

`AIRCRAFT`, `AMBIENT`, `OTHER`, `SPEECH`, `TRAFFIC`, `WIND`

Bu sonuçlar doğrulama amaçlıdır. Güvenilir saha kullanımı için SVANTEK
kayıtlarıyla etiketli veri hazırlanmalı ve model yeniden eğitilmelidir.

## Bilinen sınırlama: SVL içindeki event sesi

SVL içindeki gömülü ses için mevcut Python extractor deneysel niteliktedir.
SvanPC++ ile üretilen WAV çıktısı farklı bir dönüşüm zinciri kullanabildiğinden,
bu sesler cızırtılı veya boğuk duyulabilir ve AI doğruluğunu düşürebilir.

Önerilen üretim yaklaşımı, SVAN setup'ında **Audio Recording: Wave** modunu
kullanmak ve cihazın oluşturduğu ayrı WAV dosyasını edge agent ile doğrudan
aktarmaktır. Böylece özel SVL ses decoder'ı devreden çıkar.

## Git politikası

Depoya kayıt, `.enc` dosyası, SQLite veritabanı, AES anahtarı, yerel `.local`
başlatma dosyası veya model ağırlığı eklenmez. Bunlar `.gitignore` ile dışarıda
tutulur.
