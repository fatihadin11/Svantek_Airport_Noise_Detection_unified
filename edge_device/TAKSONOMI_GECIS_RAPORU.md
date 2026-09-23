# Airport Noise — Edge Device: Taksonomi Geçiş Raporu

**Tarih:** 2026-08-11
**Kapsam:** `edge_device/` klasöründeki `firmware.py`, `web_server.py`, `center_server.py`
**Değişmeyen:** `class_config.py` (ana projeden geldiği için dokunulmadı, zaten yeni taksonomiyi tanımlıyor)

---

## 1. Neden bu değişiklik

Ana proje (havalimanı ses toplama pipeline'ı) düz 6 sınıflı taksonomiden 2 seviyeli,
10 sınıflı olana geçti. `edge_device/` klasörü bu geçişte **geride kalmıştı** —
firmware, dashboard ve merkez sunucu hâlâ eski sınıf isimlerini kullanıyordu.

| | Eski (6 sınıf) | Yeni (10 sınıf, 4 ana grup) |
|---|---|---|
| AIRCRAFT | tek sınıf | `JET_AIRCRAFT`, `HELICOPTER`, `APU_GSE` |
| ENVIRONMENT | `WIND` | `WIND`, `PRECIPITATION`, `NATURE` |
| CITY_LIFE | `TRAFFIC`, `SPEECH` | `TRAFFIC`, `SIREN_ALARM`, `SPEECH` |
| OTHER | `OTHER`, `AMBIENT` | `OTHER` (AMBIENT kaldırıldı) |

---

## 2. Klasör yapısı (değişmedi)

```
Airport_Noise\
├── firmware.py           ← GÜNCELLENDİ
├── web_server.py         ← GÜNCELLENDİ
├── center_server.py      ← GÜNCELLENDİ
├── class_config.py       ← (ana projeden, referans — edge_device'a kopyalanmadı)
│
├── recordings\
│   └── rec_2026-08-11_13-15-00_JET_AIRCRAFT.wav   ← dosya adları artık yeni sınıf isimleriyle
├── device_data.db
├── center_data.db
├── center_recordings\
└── firmware.log
```

---

## 3. Alınan mimari kararlar

Değişikliğe başlamadan önce üç açık soru netleştirildi:

1. **Checkpoint durumu:** Yeni 10 sınıflı `beats_mlp.pt` **henüz eğitilmedi**.
   Kod ileriye dönük hazırlandı; gerçek checkpoint gelene kadar firmware
   **kasıtlı olarak çalışmayacak** (bkz. §5).
2. **Sınıf listesinin kaynağı:** `class_config.py` **import edilmedi** —
   `edge_device/` klasörünün bağımsız/taşınabilir kalması tercih edildi.
   Sınıf listesi ve renkler `class_config.py`'den **elle kopyalandı**.
   ⚠️ Ana projede taksonomi tekrar değişirse bu üç dosya elle güncellenmeli.
3. **OTHER'ın kaydı:** Eski davranış korundu — `OTHER` majority'yi kazansa
   bile hâlâ **kaydedilmiyor / merkeze gönderilmiyor** (sessizce geçiliyor).

---

## 4. Dosya bazlı değişiklikler

### `firmware.py`
- `CLASSES` listesi eski 6 sınıftan yeni 10 sınıfa güncellendi; sıralama
  **alfabetik** korundu (ana projedeki `LabelEncoder` davranışıyla uyumlu
  olması için zorunlu).
- Eski `AMBIENT_LABELS` sabiti `SKIP_LABELS` olarak yeniden adlandırıldı,
  çünkü artık bir "sınıf" değil, sadece "kaydetme" mantığını taşıyan bir
  küme: `{"SILENCE", "UNKNOWN", "OTHER"}`.
- RMS eşiğinin altındaki sessiz pencereler artık `"AMBIENT"` yerine yeni,
  CLASSES listesinde **yer almayan** bir dahili sentinel olan
  `SILENCE_SENTINEL = "SILENCE"` ile majority buffer'a işleniyor. Bu,
  gerçek bir model çıktısı değil, sadece kararlılık mekanizması.
- `_build_mlp()` içine checkpoint/sınıf-sayısı uyuşmazlığında **net,
  açıklayıcı bir hata mesajı** eklendi (eskiden ham PyTorch shape-mismatch
  hatası verirdi).
- `classify()` içindeki `[1, 6]` / `[6]` yorumları `[1, N_CLASSES]` /
  `[N_CLASSES]` olarak güncellendi.
- **Değiştirilmedi (bilinçli):** `CONFIDENCE_THR=0.75`, `MAJORITY_LEN=7`,
  `MAJORITY_MIN_VOTES=5`. Sınıf sayısı artışı bu eşiklerin yeniden kalibre
  edilmesini gerektirebilir (bkz. §6), ama bu ayrı bir tuning kararı —
  taksonomi değişikliğiyle otomatik çözülmüyor.

### `web_server.py`
- Python `LABEL_COLORS` sözlüğü yeni 10 sınıfa güncellendi
  (`class_config.py::CLASS_COLORS` ile birebir aynı hex değerleri).
- Dashboard'da fiilen kullanılan JS `LABEL_COLORS` objesi güncellendi.
- Hiçbir yerde kullanılmadığı doğrulanan (`grep` ile teyit edildi) ölü CSS
  değişkenleri (`--aircraft`, `--speech`, `--traffic`, `--wind`, `--other`,
  `--ambient`, `--unknown`) temizlendi.

### `center_server.py`
- Merkez dashboard'daki JS `LABEL_COLORS` objesi aynı 10 sınıfa güncellendi.
- Python tarafında sınıf ismine bağlı hiçbir kod yoktu (`label` sütunu
  serbest metin olarak saklanıyor) — başka değişiklik gerekmedi.

### `class_config.py`
- Değiştirilmedi. Zaten güncel taksonomiyi tanımlıyor; edge_device
  dosyaları buradan **elle** senkronize edildi.

---

## 5. ⚠️ Bilinen risk — devreye almadan önce

`firmware.py` şu an **10 sınıflı bir MLP mimarisi bekliyor**
(`nn.Linear(256, 10)`), ama gerçek `beats_mlp.pt` checkpoint'i hâlâ eski
6 sınıfla eğitilmiş durumda. Yeni checkpoint hazır olmadan firmware
çalıştırılırsa, `_build_mlp()` artık şu şekilde **net bir hatayla durur**
(sessizce yanlış sonuç üretmek yerine):

```
MLP checkpoint (...) beklenen 10 sınıflı mimariyle UYUMSUZ.
Bu genelde checkpoint'in henüz yeni taksonomiyle eğitilmediği anlamına gelir...
```

**Yapılacak:** Ana projede 10 sınıflı model eğitildikten sonra yeni
`beats_mlp.pt` checkpoint'i `edge_device`'a taşınmalı.

---

## 6. Rapor için not — henüz yapılmayan ama gerekebilecek işler

- **Eşik kalibrasyonu:** `CONFIDENCE_THR`, `MAJORITY_LEN`,
  `MAJORITY_MIN_VOTES` yeni model eğitildikten sonra per-class confusion
  matrix'e bakılarak yeniden ayarlanmalı — özellikle akustik olarak
  birbirine yakın alt sınıflar (`JET_AIRCRAFT` / `HELICOPTER` / `APU_GSE`)
  için.
- **Doküman/kod uyuşmazlığı (bu migrasyondan bağımsız, önceden var):**
  `README_edge.md`, `CONFIDENCE_THR=0.45` ve `MAJORITY_LEN=5` olarak
  belgeliyor; gerçek kodda değerler `0.75` ve `7`/`5` (`MAJORITY_MIN_VOTES`).
  Bu migrasyon kapsamında dokunulmadı, ayrıca ele alınmalı.
- **Checkpoint dosya adı:** `DEFAULT_MLP_CKPT = "beats_mlp.pt"` sabit
  kaldı — yeni checkpoint farklı bir adla geliyorsa (ör. `beats_mlp_v2.pt`)
  bu satır ayrıca güncellenmeli.
