# Airport Noise Detection — Birleşik Sistem (SVANTEK Entegrasyonlu)

Havalimanı ortamında çalışan uçtan uca gerçek zamanlı çevresel ses sınıflandırma sistemi. EfficientNet-B0 ve BEATs (Microsoft) foundation modelini paralel çalıştırır; SVANTEK SV 971 profesyonel ses seviye ölçerden gelen kayıtları şifreli olarak toplayıp aynı yapay zekâyla otomatik analiz eden bir web panelini de içerir.

Repo: [Svantek_Airport_Noise_Detection_unified](https://github.com/fatihadin11/Svantek_Airport_Noise_Detection_unified)

> Bu dosya, projenin iki ayrı bileşenine ait README'lerin (ana sınıflandırma sistemi ve SVANTEK/edge alt sistemi) birleştirilmiş hâlidir. İki bileşen aynı Python koduna (`noise_detector.py`) dayanır ama farklı model ağırlığı klasörleri kullanabilir — detaylar aşağıda.

---

## Proje Bileşenleri

- **Ana sistem** (repo kökü) — PyQt6 GUI, dosya/mikrofon üzerinden gerçek zamanlı sınıflandırma, model eğitim scriptleri.
- **`edge_device_svantek/`** — SVANTEK SV 971 entegrasyonu: Raspberry Pi kaydı yükler, FastAPI backend şifreli kaydı çözüp aynı AI ile arka planda analiz eder, React tabanlı tek panelde sonuçları gösterir.

Her iki bileşen de sınıflandırma için **aynı kaynak kodu** (`noise_detector.py`, `class_config.py`, `BEATs.py` vb., repo kökünde) kullanır; `edge_device_svantek` bu kodu import eder, kopyalamaz.

---

## Repo Yapısı

```
Airport_Noise/                       (repo kökü)
├── beats/                           # microsoft/unilm BEATs kaynak kopyası (BEATs.py bağımlılığı)
├── cache/
│   └── manifest_v6.csv              # Eğitim manifesti (collector SQLite + onaylı live klipler)
├── edge_device/                     # Ana projenin kendi edge firmware'i (SVANTEK'ten bağımsız, ayrı iş)
├── edge_device_svantek/             # SVANTEK alt sistemi
│   ├── analysis_outputs/            # Airport AI analiz çıktıları (backend üretir)
│   ├── backend/                     # FastAPI backend
│   │   └── services/
│   │       └── noise_analysis_service.py   # AirportNoiseSystem'i çağıran servis
│   ├── edge_agent/
│   ├── frontend/                    # React panel
│   ├── models/                      # Bu alt sisteme özel model ağırlıkları (bkz. Kurulum §2)
│   ├── scripts/
│   ├── .gitignore
│   ├── README_UNIFIED.md
│   ├── requirements-unified.txt
│   └── start_backend_keyed.local.ps1   # Git'e eklenmez, .example.ps1'den kopyalanır
├── models/                          # Ana sistemin varsayılan model klasörü
├── outputs/
│   └── training_beats/              # Eğitim grafikleri, confusion matrix
├── class_config.py                  # ★ TEK sınıf kaynağı — isim/renk/ağırlık burada
├── BEATs.py / backbone.py / modules.py / quantizer.py   # BEATs model tanımı
├── noise_detector.py                # Ana sistem sınıfı — GUI ve edge backend ortak kullanır
├── gui_main.py                      # PyQt6 arayüzü
├── mic_map.py
├── dataset_builder.py                # Manifest oluşturucu (v6)
├── env_audio_processor.py           # ⚠ KULLANILMIYOR — eski AMBIENT veri seti iptal edildi
├── train_beats.py / train_efficientnet.py / train_cnn.py
├── setup_live_clips_folders.ps1
└── requirements.txt
```

---

## Sınıflandırılan Kategoriler

İki seviyeli taksonomi — sınıflandırma/eğitim hep **alt sınıf** düzeyinde çalışır, ana sınıf sadece gruplama/dokümantasyon amaçlıdır (bkz. `class_config.py::CLASS_GROUPS`). Bu taksonomi, veri toplama tarafındaki kardeş proje **airport-audio-collector** ile birebir aynıdır.

| Ana Sınıf | Alt Sınıf | Açıklama |
|---|---|---|
| AIRCRAFT | `JET_AIRCRAFT` | Uçak motoru, kalkış/iniş, geniş bant, Doppler etkili sesler |
| AIRCRAFT | `HELICOPTER` | Döner kanat: düşük frekanslı (<100 Hz) periyodik darbe sesleri |
| AIRCRAFT | `APU_GSE` | Yer güç ünitesi / apron destek ekipmanı: sürekli tonal sesler |
| ENVIRONMENT | `WIND` | Rüzgarlığa çarpan rüzgar: türbülanslı, düşük frekanslı sesler |
| ENVIRONMENT | `PRECIPITATION` | Yağmur, dolu, gök gürültüsü |
| ENVIRONMENT | `NATURE` | Kuş, köpek, kurbağa gibi vahşi yaşam kaynaklı yüksek frekanslı sesler |
| CITY_LIFE | `TRAFFIC` | Karayolu taşıtları: yuvarlanma ve motor sesleri |
| CITY_LIFE | `SIREN_ALARM` | İtfaiye/ambulans/yer aracı geri vites ikaz tonları + siren |
| CITY_LIFE | `SPEECH` | Yakın çevre insan konuşması, anons, bağırma |
| OTHER | `OTHER` | Yukarıdakilerin hiçbirine uymayan, arka plan gürültüsünü aşan anomaliler |

> Sınıf listesi, renkleri ve eğitim/inference ağırlıkları **tek kaynaktan** gelir: `class_config.py`. Yeni bir sınıf eklemek/çıkarmak için SADECE bu dosya değişir.

---

## Sistem Gereksinimleri

- **OS:** Windows 10/11 (64-bit)
- **GPU:** CUDA destekli NVIDIA GPU (önerilir — BEATs encoder GPU olmadan çok yavaş çalışır)
- **RAM:** 16 GB+
- **Disk:** C: sürücüsünde ~2 GB, D: sürücüsünde ~500 MB (ana sistemin model ağırlıkları için)
- **Python:** 3.9 – 3.11 (ana sistem), `edge_device_svantek` alt sistemi için ayrıca **Python 3.10** ve **Node.js** (frontend)

> ⚠️ Ana sistem bazı yolları `D:\` sürücüsünde sabit kodlanmış olarak bekler (`D:\models`, `D:\Airport_Live_Clips`). `D:` sürücünüz yoksa `noise_detector.py`, `train_beats.py` ve `dataset_builder.py` içindeki ilgili sabitleri güncelleyin. `edge_device_svantek` alt sistemi bu sabitlere bağlı değildir — kendi model klasörünü kullanır (aşağıya bakın).

---

## Kurulum

### 1. Ana Sistem

```bash
git clone https://github.com/fatihadin11/Svantek_Airport_Noise_Detection_unified.git
cd Svantek_Airport_Noise_Detection_unified

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

CUDA ile PyTorch kurmak için önce [pytorch.org](https://pytorch.org/get-started/locally/) adresinden uygun komutu alın:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**BEATs encoder'ı indirin** (~90 MB, [microsoft/unilm](https://github.com/microsoft/unilm/tree/master/beats)) ve şuraya yerleştirin: `D:\models\BEATs_iter3_plus_AS2M.pt`

**`D:\` klasör yapısını oluşturun:**
```
D:\
├── models\
│   └── BEATs_iter3_plus_AS2M.pt
└── Airport_Live_Clips\
    ├── pending\   (JET_AIRCRAFT, HELICOPTER, APU_GSE, WIND, PRECIPITATION, NATURE, TRAFFIC, SIREN_ALARM, SPEECH, OTHER)
    ├── approved\  (aynı 10 alt klasör)
    └── rejected\  (aynı 10 alt klasör)
```
`gui_main.py` bu alt klasörleri kendisi de oluşturur (`os.makedirs(..., exist_ok=True)`); elle hazırlamak isterseniz `setup_live_clips_folders.ps1` betiğini çalıştırabilirsiniz. Eski taksonomiyle toplanmış klipler varsa script onlara dokunmaz, elle silinmesi gerekir.

### 2. SVANTEK Edge Alt Sistemi (`edge_device_svantek/`)

```powershell
cd edge_device_svantek
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-unified.txt

cd frontend
npm install
```

`requirements-unified.txt` hem backend'in kendi bağımlılıklarını (`backend/requirements.txt`) hem de ana sistemin bağımlılıklarını (`../requirements.txt`) kurar.

**Model ağırlıkları:** `edge_device_svantek/models/` klasörünü oluşturup şu dosyaları buraya kopyalayın (ana sistemin `D:\models\` ve `models\` klasörlerinden):
```
edge_device_svantek/models/
├── best_model.pkl               ├── best_efficientnet.pt
├── label_encoder.pkl            ├── efficientnet_label_encoder.pkl
├── best_cnn.pt                  ├── beats_mlp.pt
├── cnn_label_encoder.pkl        └── BEATs_iter3_plus_AS2M.pt
```
Bu klasör ana sistemin `models/` ve `D:\models\` klasörlerinden **bağımsızdır** — iki bileşen farklı model sürümleri kullanabilir. `noise_analysis_service.py`, `AirportNoiseSystem`'i başlatırken bu klasörü açıkça belirtir (`models_dir`, `beats_encoder_path`, `beats_mlp_path` parametreleri).

**Yerel ayar dosyasını hazırlayın:**
```powershell
Copy-Item .\scripts\start_backend_keyed.example.ps1 .\start_backend_keyed.local.ps1
```
`start_backend_keyed.local.ps1` içine `AES_KEY_B64` ve edge adresini girin. Bu dosya Git'e eklenmez.

---

## Eğitim Verisi Kaynağı

Eski harici veri setleri (ESC-50, AeroSonicDB, Generic Audio Classifier) **tamamen iptal edildi**. Model artık sıfırdan, iki kaynaktan eğitiliyor:

1. **airport-audio-collector SQLite pipeline'ı** — kardeş proje, YouTube'dan otonom veri toplayıp CLAP ile doğruluyor. `dataset_builder.py::load_from_collector_db()` bu projenin `pipeline.sqlite3`'ünden `status='accepted'` örnekleri okur. Kendi `pipeline.sqlite3` yolunuzu `dataset_builder.py::COLLECTOR_DB_PATH` sabitinde (veya aynı isimli ortam değişkeninde) belirtmeniz gerekir.
2. **Onaylı canlı mikrofon klipleri** — GUI Faz 2'de onaylanan klipler.

İsteğe bağlı ek kaynak: `D:\Svantek_Recordings\` altına sınıf ismiyle eşleşen klasörler (`JET_AIRCRAFT\`, `HELICOPTER\`, ...) halinde gerçek mikrofon kayıtları koyarsanız `train_beats.py` bunları otomatik dahil eder.

Sadece inference yapacaksanız bu adımı atlayabilirsiniz.

---

## Çalıştırma

### A) Ana Sistem — GUI (Dosya/Mikrofon Analizi)

```bash
python gui_main.py
```
- **Faz 1 — Dosya Analizi:** Ses dosyası yükle, sınıflandır, haritada görselleştir
- **Faz 2 — Canlı Kayıt:** Mikrofondan gerçek zamanlı sınıflandırma ve aktif öğrenme

### B) SVANTEK Unified Panel

```powershell
cd edge_device_svantek
.\start_backend_keyed.local.ps1

cd frontend
npm run dev
```

**Akış:**
1. SVANTEK kaydı biter; Pi, WAV/CSV/SVL çıktısını şifreli olarak backend'e yükler.
2. Kayıt merkeze gelir gelmez AI analizi arka planda başlar (panelden gerekirse yeniden tetiklenebilir).
3. Backend, şifreli WAV'i geçici olarak çözüp mono/22.050 Hz'e dönüştürür.
4. Airport AI, kaydı 5 saniyelik / 2,5 saniye ilerlemeli pencerelerde inceler.
5. Ardışık aynı sınıflar tek olayda birleştirilir (ör. `04:10–04:28 JET_AIRCRAFT %94`).
6. Geçici dosyalar silinir; olaylar SQLite'a yazılır ve panelde ilgili ses zamanına atlanabilir.

**Daha önce alınmış Pi kayıtlarını içe aktarma:** Panelde **Ses / Şifreli Kayıt Yükle** ile en az `audio.wav.enc` seçin (`data_all.csv.enc`, `raw.SVL.enc` isteğe bağlı eklenebilir). Sistem AES anahtarını doğrular, diskte yalnızca `.enc` kopyalarını tutar ve analizi otomatik başlatır. Düz WAV yükleme seçeneği de korunur.

---

## Eğitim

```bash
python dataset_builder.py       # 1. cache/manifest_v6.csv oluşturur
python train_beats.py           # 2. BEATs MLP (önerilen) — ilk çalıştırma ~45-50 dk (GPU), çıktı: D:\models\beats_mlp.pt
python train_efficientnet.py    # 3. EfficientNet (isteğe bağlı)
python train_cnn.py             # 4. CNN/SVM (isteğe bağlı)
```
> ⚠ `train_cnn.py` güncel ensemble'da **kullanılmıyor** — hâlâ eski 6-sınıf taksonomiyi kullanıyor, `manifest_v6.csv` ile doğrudan uyumlu değil.

---

## Mimari Özeti

```
Mikrofon / Dosya / SVANTEK Kaydı
      │
      ▼
  Rolling Buffer (5s pencere, 2.5s hop)
      │
      ├──► EfficientNet-B0 ──► Softmax     (Mel Spectrogram)
      │
      └──► BEATs Encoder (frozen) ──► MLP ──► Softmax   (768-dim embedding)
                       │
                       ▼
                Ensemble (α=0.5)
                       │
                       ▼
      GUI: Majority Voting (n=5) → Tahmin + Güven Skoru
      Edge: Ardışık pencere birleştirme → Olay (start/end/label/confidence) → SQLite
```
Her iki uç da aynı `noise_detector.py::AirportNoiseSystem` sınıfını kullanır; edge tarafı `backend/services/noise_analysis_service.py` üzerinden çağırır ve kendi `models/` klasörünü (bkz. Kurulum §2) parametre olarak geçer.

---

## Model Performansı

> ⚠ Aşağıdaki sayılar **eski 6 sınıf taksonomisiyle** ölçülmüştü, artık geçerli değil — yeni 10 sınıf taksonomisiyle henüz eğitim tamamlanmadı.

| Model | F1 Macro | Split Yöntemi |
|---|---|---|
| SVM | — | — |
| CNN | — | — |
| EfficientNet-B0 | — (yeniden eğitim bekliyor) | Random |
| **BEATs MLP** | — (yeniden eğitim bekliyor) | Group-aware (leakage-free) |

---

## Notlar / Bilinen Sorunlar

- `manifest_v6.csv` içindeki dosya yolları bu makineye özgüdür; başka makinede `dataset_builder.py::COLLECTOR_DB_PATH` güncellenmeli.
- Sınıf ismi/renk/ağırlık her zaman `class_config.py`'den gelir — başka hiçbir dosyada elle kopyalanmamalı.
- BEATs embedding cache dosyaları (~186 MB, `.pkl`) repoya dahil değildir; `train_beats.py` ilk çalıştırmada oluşturur.
- CUDA bulunamazsa sistem CPU moduna düşer; BEATs embedding hesaplama çok uzar.
- `env_audio_processor.py` kullanılmıyor (eski AMBIENT veri seti iptal edildi).
- `edge_device_svantek/models/` ana sistemin `models/`/`D:\models\` klasörlerinden bağımsız bir kopyadır — birini güncelleyip diğerini unutmamaya dikkat edin.
- `start_backend_keyed.local.ps1` ve benzeri yerel ayar dosyaları Git'e eklenmez.

---

*Bu README, önceden ayrı duran iki proje (ana AI sistemi ve SVANTEK edge entegrasyonu) tek repo altında birleştirildikten sonra güncel klasör yapısını yansıtacak şekilde yeniden yazılmıştır. Kod üzerinde ayrıca yakın zamanda başka değişiklikler yapılmış olabilir; yukarıdaki dosya/parametre adlarını mevcut kodunuzla hızlıca karşılaştırmanız önerilir.*
