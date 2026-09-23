# SVANTEK Airport Noise — Birleşik Sistem

Bu klasör, iki kaynak projeye dokunmadan oluşturulmuş çalışma kopyasıdır.

## Akış

1. SVANTEK kaydı biter; Pi WAV, CSV ve SVL çıktısını şifreli olarak backend'e yükler.
2. SVANTEK kaydı merkeze gelir gelmez AI analizi arka planda başlar. Tek
   paneldeki düğme, gerekirse analizi yeniden çalıştırmak için kullanılır.
3. Backend, gerekiyorsa şifreli WAV'i yalnızca geçici dosyaya çözer ve kaydı
   mono/22.050 Hz WAV biçimine dönüştürür.
4. Airport AI, kaydı 5 saniyelik ve 2,5 saniye ilerlemeli pencerelerde inceler.
5. Ardışık aynı sınıflar tek olay olarak birleştirilir. Örnek: `04:10–04:28 JET_AIRCRAFT %94`.
6. Geçici çözülmüş ve dönüştürülmüş dosyalar analiz sonunda silinir; olaylar
   SQLite'a kaydedilir ve aynı panelde ilgili ses zamanına atlanabilir.

### Daha önce alınmış Pi kayıtlarını içe aktarma

Panelde **Ses / Şifreli Kayıt Yükle** düğmesine basıp en az `audio.wav.enc`
dosyasını seçin. Aynı seçime varsa `data_all.csv.enc` ve `raw.SVL.enc`
dosyalarını da ekleyebilirsiniz. Sistem, dosyaların bu sistemin AES anahtarıyla
açılabildiğini doğrular; diskte yalnızca `.enc` kopyalarını tutar ve AI analizini
otomatik başlatır. WAV yükleme seçeneği ayrıca korunur.

## Kurulum

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-unified.txt
cd frontend
npm install
```

Model dosyaları GitHub'a dahil edilmez. Kullanılacak EfficientNet/CNN/SVM
ağırlıklarını ve BEATs encoder checkpoint'ini `airport_ai/models/` içine
yerleştirin; gerekli dosyaların listesi o klasördeki README'de bulunur.
Modelin sınıf taksonomisi yeniden eğitilene kadar sonuçlar test/doğrulama
amaçlı değerlendirilmelidir.

## Çalıştırma

Önce şablonu yerel ayar dosyasına kopyalayın ve `AES_KEY_B64` ile edge adresini
doldurun. Bu dosya Git'e eklenmez.

```powershell
Copy-Item .\scripts\start_backend_keyed.example.ps1 .\start_backend_keyed.local.ps1
.\start_backend_keyed.local.ps1

cd frontend
npm run dev
```
