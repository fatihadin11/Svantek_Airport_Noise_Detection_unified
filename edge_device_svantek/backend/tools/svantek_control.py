import argparse
import time
import usb.core
import usb.util

VID = 0x0017
PID = 0x0001
OUT_ENDPOINT = 0x01
IN_ENDPOINT = 0x83
INTERFACE = 0


class SvantekUSB:
    def __init__(self):
        self.dev = usb.core.find(idVendor=VID, idProduct=PID)

        if self.dev is None:
            raise SystemExit("SVAN 971 bulunamadı. VirtualBox USB passthrough kontrol et.")

        try:
            self.dev.set_configuration()
        except usb.core.USBError:
            pass

        try:
            if self.dev.is_kernel_driver_active(INTERFACE):
                self.dev.detach_kernel_driver(INTERFACE)
        except Exception:
            pass

        try:
            usb.util.claim_interface(self.dev, INTERFACE)
        except usb.core.USBError as e:
            raise SystemExit(
                f"USB interface kullanılıyor: {e}\n"
                "Çözüm: VirtualBox USB menüsünden 0017:0001 cihazını çıkarıp tekrar seç veya SVAN USB kablosunu çıkar-tak yap."
            )

    def close(self):
        try:
            usb.util.release_interface(self.dev, INTERFACE)
        except Exception:
            pass

        try:
            usb.util.dispose_resources(self.dev)
        except Exception:
            pass

    def send(self, command, expect_response=True, timeout=3000, delay=0.3):
        if not command.endswith(";"):
            command += ";"

        print(f"Gönderiliyor: {command}")

        written = self.dev.write(
            OUT_ENDPOINT,
            command.encode("ascii"),
            timeout=timeout,
        )

        print(f"Yazılan byte: {written}")

        if not expect_response:
            print("Bu komut için cevap beklenmedi.")
            return None

        time.sleep(delay)

        try:
            data = self.dev.read(
                IN_ENDPOINT,
                512,
                timeout=timeout,
            )
            response = bytes(data).decode("ascii", errors="replace").strip()
            print(f"Cevap: {response}")
            return response
        except usb.core.USBTimeoutError:
            print("Cevap gelmedi / timeout.")
            return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--cmd", default=None)
    parser.add_argument(
        "--no-response",
        action="store_true",
        help="--cmd için cevap bekleme. #1,S0/#1,S1/#1,T0/#1,T1 gibi set komutlarında hızlıdır.",
    )
    args = parser.parse_args()

    svan = SvantekUSB()

    try:
        if args.status:
            svan.send("#1,S?", expect_response=True)

        if args.start:
            svan.send("#1,S1", expect_response=False)
            time.sleep(1.0)
            svan.send("#1,S?", expect_response=True)

        if args.stop:
            svan.send("#1,S0", expect_response=False)
            time.sleep(1.0)
            svan.send("#1,S?", expect_response=True)

            print("\nSon dosya adları deneniyor:")
            for cmd in ["#7,LB", "#7,LW"]:
                try:
                    svan.send(cmd, expect_response=True)
                except Exception as e:
                    print(f"{cmd} hata verdi: {e}")

        if args.all:
            svan.send("#1", expect_response=True)

        if args.cmd:
            svan.send(args.cmd, expect_response=not args.no_response)

    finally:
        svan.close()


if __name__ == "__main__":
    main()
