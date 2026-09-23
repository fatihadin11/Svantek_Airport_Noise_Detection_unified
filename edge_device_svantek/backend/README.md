# Backend — Pi Ses Kayıt Sistemi (Faz 1)

## Kurulum (Raspberry Pi üzerinde)

```bash
# Sistem bağımlılıkları (PortAudio - mikrofon erişimi için gerekli)
sudo apt update
sudo apt install -y libportaudio2 portaudio19-dev python3-pip python3-venv

# Sanal ortam oluştur
cd pi-ses-sistemi/backend
python3 -m venv venv
source venv/bin/activate

# Python paketlerini kur
pip install -r requirements.txt
```

## Çalıştırma

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Çalıştıktan sonra:
- API: `http://<pi-ip>:8000/api/...`
- Otomatik oluşan API dokümantasyonu (Swagger UI): `http://<pi-ip>:8000/docs`

## Mikrofon Kontrolü

Bağlı ses giriş cihazlarını görmek için:

```
GET /api/devices
```

Eğer mikrofon listede görünmüyorsa:
```bash
arecord -l   # ALSA'nın gördüğü cihazları listeler
```
komutu ile donanım seviyesinde de kontrol edilebilir.

## Test Edilen Uçlar (Faz 1 kapsamında)

- `GET  /api/health` — sağlık kontrolü
- `GET  /api/recordings` — kayıt listesi
- `GET  /api/recordings/{id}` — tek kayıt detayı
- `GET  /api/recordings/{id}/audio` — ses dosyasını indir/oynat
- `GET  /api/recordings/{id}/waveform` — waveform verisi (görselleştirme için)
- `POST /api/recordings/start` — mikrofon ile kayda başla
- `POST /api/recordings/stop` — kaydı durdur, kaydet
- `POST /api/recordings/upload` — dışarıdan ses dosyası yükle
- `PATCH /api/recordings/{id}` — başlık güncelle
- `DELETE /api/recordings/{id}` — kaydı sil

`start`/`stop` uçları gerçek bir mikrofon gerektirdiği için geliştirme ortamında
otomatik test edilemedi; Pi'ye mikrofon takıldığında ilk doğrulama bu ikisiyle yapılmalı.
Diğer tüm uçlar (liste, yükleme, waveform, güncelleme, silme, hatalı format reddi)
uçtan uca test edildi ve sorunsuz çalışıyor.

## Sıradaki Adım

Faz 2: Web panel (React + wavesurfer.js) bu API'yi tüketecek şekilde geliştirilecek.
