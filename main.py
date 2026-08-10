"""
=============================================================
 HAVALIMANL ÇEVRESEL GÜRÜLTÜ TESPİT SİSTEMİ – DEMO ÇALIŞTIRICI
=============================================================

Kullanım örnekleri:

  # 1) Kendi ses dosyanızla:
  python main.py --audio benim_sesim.wav

  # 2) Demo modu (sentetik WAV üretir):
  python main.py --demo

  # 3) Özel parametreler:
  python main.py --audio ses.wav --sr 44100 --out sonuclar/
"""

import argparse
import sys
import os

# Proje dizinini Python yoluna ekle
sys.path.insert(0, os.path.dirname(__file__))

from noise_detector import (F 
    AirportNoiseSystem,
    _synth_wav,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Havalimanı Çevresel Gürültü Tespit Sistemi"
    )
    p.add_argument("--audio", type=str, default=None,
                   help="WAV ses dosyası yolu")
    p.add_argument("--demo", action="store_true",
                   help="Sentetik WAV ile demo çalıştır")
    p.add_argument("--sr", type=int, default=22050,
                   help="Hedef örnekleme hızı (varsayılan: 22050)")
    p.add_argument("--out", type=str, default="outputs",
                   help="Çıktı klasörü (varsayılan: outputs/)")
    p.add_argument("--prefix", type=str, default="airport_noise",
                   help="Çıktı dosya öneki")
    return p.parse_args()


def main():
    args = parse_args()

    # ─ Demo modu ──────────────────────────────────────
    if args.demo or args.audio is None:
        print("[DEMO MODU] Sentetik havalimanı sesi üretiliyor...")
        audio_path = "demo_airport_sound.wav"
        _synth_wav(audio_path, duration=10.0, sr=args.sr)
    else:
        audio_path = args.audio
        if not os.path.exists(audio_path):
            print(f"[HATA] Dosya bulunamadı: {audio_path}")
            sys.exit(1)

    # ─ Sistemi başlat ve çalıştır ─────────────────────
    system = AirportNoiseSystem(
        target_sr=args.sr,
        output_dir=args.out
    )
    report = system.run(audio_path, prefix=args.prefix)

    # ─ Sonuç özeti yazdır ─────────────────────────────
    print("\n📊  RAPOR ÖZETİ")
    print("─" * 40)
    print(f"  Dosya     : {report['audio_path']}")
    print(f"  Süre      : {report['duration_s']:.2f} s")
    print(f"  SR        : {report['sample_rate']} Hz")
    print(f"  Örnekler  : {report['n_samples']:,}")
    print("\n  📂 Oluşturulan dosyalar:")
    for f in report["output_files"]:
        print(f"    → {f}")

    print("\n  🎯 Gürültü Kaynak Dağılımı:")
    for label, pct in sorted(report["classification"].items(),
                              key=lambda x: -x[1]):
        bar = "█" * int(pct / 2)
        print(f"    {label:10s} {bar:50s} {pct:5.1f}%")

    return report


if __name__ == "__main__":
    system = AirportNoiseSystem(target_sr=22050, output_dir="outputs")
    report = system.run(
        r"C:\Users\Fatih\Desktop\TUBITAK\Airport_Noise\sounds\Flying Plane Sound Effect.wav"
    )