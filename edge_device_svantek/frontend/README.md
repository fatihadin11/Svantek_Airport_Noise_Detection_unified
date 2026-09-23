# Frontend — Web Panel (Faz 2)

React + Vite ile geliştirilmiş web panel. Backend'in (Faz 1) sunduğu REST API'yi
kullanır: kayıt listesi, waveform görüntüleme/oynatma, mikrofon ile kayıt alma,
dosya yükleme ve silme.

## Tasarım Notları

- Waveform için wavesurfer.js gibi bir kütüphane yerine, backend'in zaten ürettiği
  downsample edilmiş genlik verisi (`/api/recordings/{id}/waveform`) doğrudan bir
  `<canvas>` üzerine çubuklar halinde çiziliyor. Bu, tarayıcıda ayrıca ses dosyası
  çözümlemeyi (decode) gerektirmiyor ve daha hafif.
- Sesin kendisi oynatma için native `<audio>` elemanıyla, `/api/recordings/{id}/audio`
  endpoint'inden stream ediliyor; waveform üzerindeki ilerleme çubuğu bu elemanla
  senkronize çalışıyor.
- Tasarım dili: koyu, "stüdyo konsolu" estetiği. Kayıt tuşu kırmızı (VU-metre
  rengi), oynatma/ilerleme vurgusu kehribar tonunda, süre/zaman damgaları
  monospace fontla (tape counter hissi).

## Kurulum

```bash
cd frontend
npm install
```

## Geliştirme

Backend'in ayrıca çalışıyor olması gerekir (bkz. `../backend/README.md`):

```bash
# Terminal 1 — backend
cd ../backend && uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 2 — frontend
cd frontend && npm run dev
```

`vite.config.js` içinde `/api` istekleri otomatik olarak `localhost:8000`'e
proxy'leniyor, bu yüzden ayrıca bir ayar gerekmiyor.

Tarayıcıdan `http://localhost:5173` adresine gidin (ya da Pi'nin yerel ağ IP'si
üzerinden, `--host` ayarı sayesinde diğer cihazlardan da erişilebilir).

## Üretim Build'i (Pi üzerinde çalıştırmak için)

```bash
npm run build
```

`dist/` klasörü statik dosyalar olarak oluşur. Bunu backend ile aynı origin'den
servis etmek için FastAPI'de bir `StaticFiles` mount'u eklenebilir (Faz 3'te
ele alınacak), ya da Nginx ile ayrı servis edilebilir.

## Font Notu

`index.html` Google Fonts üzerinden (Space Grotesk, Inter, JetBrains Mono)
font yüklüyor. Pi internete erişemiyorsa bu fontlar sistem fontlarına
otomatik geri düşer (fallback), ancak en iyi görünüm için fontların
self-host edilmesi (örn. `public/fonts/` altına indirilip `@font-face`
ile tanımlanması) önerilir.

## Test Edilen Akış

- Kayıt listesi backend'den çekilip gösteriliyor
- Dosya yükleme → liste anında güncelleniyor
- Waveform verisi çekilip canvas'a çiziliyor
- Ses oynatma, durdurma, ilerleme çubuğu senkronizasyonu
- Silme (onay diyaloğu ile)

Tüm bu akış, geliştirme ortamında backend + frontend dev sunucuları birlikte
çalıştırılarak uçtan uca (curl ile API seviyesinde) doğrulandı. `npm run build`
hatasız tamamlanıyor.

Mikrofon ile gerçek zamanlı kayıt (Kayıt Al tuşu), gerçek bir mikrofon ve
tarayıcı ortamı gerektirdiğinden bu ortamda görsel olarak test edilemedi;
Pi'ye mikrofon takıldığında ilk doğrulanması gereken adım budur.
