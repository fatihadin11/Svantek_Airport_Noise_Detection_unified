# Airport Noise — Gerçek Zamanlı Çevresel Ses Sınıflandırma Sistemi

Havalimanı ortamında çalışan, uçtan uca gerçek zamanlı ses sınıflandırma sistemi.  
EfficientNet-B0 ve BEATs (Microsoft) foundation model'ini paralel olarak çalıştırır.

## Sınıflandırılan Kategoriler

İki seviyeli taksonomi — sınıflandırma/eğitim hep **alt sınıf** düzeyinde
çalışır, ana sınıf sadece gruplama/dokümantasyon amaçlıdır (bkz.
`class_config.py::CLASS_GROUPS`). Bu taksonomi, veri toplama tarafındaki
kardeş proje **airport-audio-collector** ile birebir aynıdır.

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

> Sınıf listesi, renkleri ve eğitim/inference ağırlıkları **tek kaynaktan**
> gelir: `class_config.py`. Yeni bir sınıf eklemek/çıkarmak için SADECE bu
> dosya değişir; başka hiçbir dosyada sınıf ismi elle kopyalanmamalıdır.

---

## Sistem Gereksinimleri

- **OS:** Windows 10/11 (64-bit)
- **GPU:** CUDA destekli NVIDIA GPU (önerilir — BEATs encoder GPU olmadan çok yavaş çalışır)
- **RAM:** 16 GB+
- **Disk:** C: sürücüsünde ~2 GB, D: sürücüsünde ~500 MB (model ağırlıkları için)
- **Python:** 3.9 – 3.11

> ⚠️ Proje bazı yolları `D:\` sürücüsünde sabit kodlanmış olarak bekler.  
> `D:` sürücünüz yoksa ilgili yolları `noise_detector.py`, `train_beats.py` ve `dataset_builder.py` içinde arayıp güncelleyin (`D:\models`, `D:\Airport_Live_Clips`).

---

## Kurulum

### 1. Repoyu klonla

```bash
git clone https://github.com/<fatihadin11>/<repo>.git
cd Airport_Noise
```

### 2. Sanal ortam oluştur ve bağımlılıkları kur

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

> PyTorch'u CUDA ile kurmak için önce [pytorch.org](https://pytorch.org/get-started/locally/) adresinden sisteminize uygun komutu alın:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```

### 3. BEATs Encoder'ı İndir

BEATs frozen encoder ağırlıklarını (~90 MB) Microsoft'un resmi kaynağından indir:

```
https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt
```

İndirilen dosyayı şu konuma yerleştir:

```
D:\models\BEATs_iter3_plus_AS2M.pt
```

### 4. D:\ Klasör Yapısını Oluştur

```
D:\
├── models\                    ← Yukarıda indirilen BEATs checkpoint buraya
│   └── BEATs_iter3_plus_AS2M.pt
└── Airport_Live_Clips\        ← Canlı kayıt oturumları için (GUI tarafından otomatik kullanılır)
    ├── pending\
    │   └── JET_AIRCRAFT\  HELICOPTER\  APU_GSE\  WIND\  PRECIPITATION\
    │       NATURE\  TRAFFIC\  SIREN_ALARM\  SPEECH\  OTHER\
    ├── approved\
    │   └── (aynı 10 alt klasör)
    └── rejected\
        └── (aynı 10 alt klasör)
```

> `PendingClipManager` (gui_main.py) bu alt klasörleri zaten dinamik
> olarak kendisi oluşturur (`os.makedirs(..., exist_ok=True)`) — elle
> oluşturman şart değil. Yine de baştan hazırlamak istersen depoda
> gelen `setup_live_clips_folders.ps1` betiğini çalıştırabilirsin:

```powershell
.\setup_live_clips_folders.ps1
```

> ⚠ Eski taksonomiyle (AIRCRAFT/AMBIENT/SPEECH/TRAFFIC/WIND/OTHER)
> toplanmış klipler varsa, bu script onlara dokunmaz — eski klasörler
> olduğu gibi kalır. Eski veri seti tamamen iptal edildiği için bunları
> kullanmaya devam etmeyeceksen elle silebilirsin.

---

## Eğitim Verisi Kaynağı

Eski harici veri setleri (ESC-50, AeroSonicDB, Generic Audio Classifier)
**tamamen iptal edildi** — yeni taksonomiyle anlamlı biçimde eşleşmiyorlardı.
Model artık sıfırdan, aşağıdaki iki kaynaktan eğitiliyor:

1. **airport-audio-collector SQLite pipeline'ı** — kardeş proje, YouTube'dan
   otonom veri toplayıp CLAP ile doğruluyor. `dataset_builder.py` bu projenin
   `pipeline.sqlite3`'ünden `status='accepted'` ve kalite eşiğini geçen
   örnekleri doğrudan okur (`load_from_collector_db()`). Kendi
   `pipeline.sqlite3` yolunu `dataset_builder.py` içindeki
   `COLLECTOR_DB_PATH` sabitinde (veya `COLLECTOR_DB_PATH` ortam
   değişkeninde) belirtmen gerekir.
2. **Onaylı canlı mikrofon klipleri** — GUI'den toplanıp Faz 2'de
   onaylanan klipler (değişmedi).

İsteğe bağlı ek kaynak: `D:\Svantek_Recordings\` altına, klasör adı sınıf
ismiyle eşleşen (`JET_AIRCRAFT\`, `HELICOPTER\`, ...) gerçek mikrofon
kayıtları koyarsan `train_beats.py` bunları da otomatik dahil eder
(CSV gerekmez). Bu klasörler hâlâ eski isimlerdeyse yeniden adlandırman
gerekir — script eski isimlendirmeleri sessizce atlar, hata vermez.

Sadece GUI'yi çalıştırıp inference yapacaksan bu adımı atlayabilirsin —
ama yeni taksonomi için henüz eğitilmiş ağırlık YOK, önce eğitim
gerekiyor (bkz. aşağıdaki "Eğitim" bölümü).

---

## Çalıştırma

### GUI (Canlı Sınıflandırma + Dosya Analizi)

```bash
python gui_main.py
```

Arayüz iki ana sekme içerir:
- **Faz 1 — Dosya Analizi:** Ses dosyası yükle, sınıflandır, haritada görselleştir
- **Faz 2 — Canlı Kayıt:** Mikrofondan gerçek zamanlı sınıflandırma ve aktif öğrenme

---

## Eğitim

Eğitim scriptlerini bu sırayla çalıştır:

### 1. Manifest Oluştur

```bash
python dataset_builder.py
```

`cache/manifest_v6.csv` oluşturur (collector SQLite + onaylı live klipler).

### 2. BEATs MLP Eğit (Önerilen)

```bash
python train_beats.py
```

- İlk çalıştırmada BASE embedding cache'i oluşturur (~45–50 dk, GPU gerekli)
- Cache oluştuktan sonraki çalıştırmalar çok daha hızlı
- Eğitilmiş MLP: `D:\models\beats_mlp.pt`

### 3. EfficientNet Eğit (İsteğe Bağlı)

```bash
python train_efficientnet.py
```

### 4. CNN / SVM Eğit (İsteğe Bağlı)

```bash
python train_cnn.py
```

> ⚠ `train_cnn.py` bu güncellemenin **dışında** bırakıldı (canlı ensemble'da
> kullanılmıyor — bkz. Mimari Özeti). Hâlâ eski 6-sınıf taksonomiyi ve eski
> `MANUAL_CLASS_WEIGHTS`'i kullanıyor; `manifest_v6.csv`'yi okursa eski
> sınıflarla eşleşmeyen etiketler nedeniyle hatalı/eksik çalışır. CNN/SVM'i
> de yeni taksonomiye taşımak istersen ayrıca söyle.

---

## Proje Yapısı

```
Airport_Noise/
│
├── class_config.py              # ★ TEK sınıf kaynağı — isim/renk/ağırlık burada
│
├── BEATs.py                    # Microsoft/unilm BEATs model tanımı
├── backbone.py                 # BEATs backbone
├── modules.py                  # BEATs yardımcı modüller
├── quantizer.py                # BEATs quantizer
│
├── noise_detector.py           # Ana sistem sınıfı — tüm model inference burada
├── gui_main.py                 # PyQt6 arayüzü
├── mic_map.py                  # Harita bileşeni
│
├── dataset_builder.py          # Manifest oluşturucu (v6 — collector SQLite + live)
├── env_audio_processor.py      # ⚠ KULLANILMIYOR — eski AMBIENT veri seti iptal edildi
├── train_beats.py              # BEATs MLP eğitim scripti
├── train_efficientnet.py       # EfficientNet eğitim scripti
├── train_cnn.py                # CNN eğitim scripti — ⚠ eski taksonomide kaldı (güncellenmedi)
├── setup_live_clips_folders.ps1 # D:\Airport_Live_Clips klasör yapısını kurar
│
├── cache/
│   └── manifest_v6.csv         # Collector SQLite + onaylı live klipler (tek, birleşik)
│
├── models/                     # Eğitilmiş model ağırlıkları — ⚠ hepsi ESKİ taksonomiyle
│   │                            eğitilmiş, yeni sınıflarla yeniden eğitilmesi gerekiyor
│   ├── beats_mlp.pt            # BEATs MLP (aktif)
│   ├── best_efficientnet.pt    # EfficientNet-B0
│   ├── best_efficientnet_finetune.pt
│   ├── efficientnet_label_encoder.pkl
│   ├── efficientnet_meta.pkl
│   ├── best_cnn.pt
│   ├── cnn_label_encoder.pkl
│   ├── best_model.pkl          # SVM
│   └── label_encoder.pkl
│
└── outputs/
    └── training_beats/         # Eğitim grafikleri ve confusion matrix'ler
```

---

## Model Performansı

> ⚠ Aşağıdaki sayılar **eski 6 sınıf taksonomisiyle** ölçülmüştü ve artık
> geçerli değil — yeni 10 sınıf taksonomisiyle henüz eğitim yapılmadı.
> Yeni veri seti toplanıp eğitim tamamlandıktan sonra bu tabloyu güncelle.

| Model | F1 Macro | Split Yöntemi |
|---|---|---|
| SVM | — | — |
| CNN | — | — |
| EfficientNet-B0 | — (yeniden eğitim bekliyor) | Random |
| **BEATs MLP** | — (yeniden eğitim bekliyor) | Group-aware (leakage-free) |

---

## Mimari Özeti

```
Mikrofon / Dosya
      │
      ▼
  Rolling Buffer (5s pencere)
      │
      ├──► EfficientNet-B0 ──► Softmax
      │         (Mel Spectrogram)
      │
      └──► BEATs Encoder (frozen) ──► MLP ──► Softmax
                (768-dim embedding)
                       │
                       ▼
                Ensemble (α=0.5)
                       │
                       ▼
            Majority Voting (n=5)
                       │
                       ▼
              Tahmin + Güven Skoru
```

---

## Notlar

- `manifest_v6.csv` içindeki canlı klip yolları (`D:\Airport_Live_Clips\approved\...`) ve collector DB'den gelen dosya yolları bu makineye özgüdür. Başka bir makinede eğitim yapılacaksa collector DB'yi/klipleri o makineye taşımak ya da `dataset_builder.py::COLLECTOR_DB_PATH`'i güncellemek gerekir.
- `dataset_builder.py` çalıştırmadan önce `COLLECTOR_DB_PATH` sabitini (veya aynı isimde bir ortam değişkenini) kendi airport-audio-collector `pipeline.sqlite3` yoluna ayarlaman gerekir — repo bunu tahmin edemez.
- Sınıf ismi/renk/ağırlık her zaman `class_config.py`'den gelir. Yeni bir sınıf eklemek/çıkarmak istersen SADECE bu dosyayı değiştir; diğer dosyalar otomatik senkron kalır.
- BEATs embedding cache dosyaları (`.pkl`, toplam ~186 MB) repoya dahil edilmemiştir. `train_beats.py` ilk çalıştırmada otomatik oluşturur. Taksonomi değiştiği için eski cache dosyaları (varsa) geçersizdir — script bunu otomatik algılayıp yeniden hesaplar.
- CUDA bulunamazsa sistem CPU moduna düşer; BEATs embedding hesaplama çok uzun sürer.
