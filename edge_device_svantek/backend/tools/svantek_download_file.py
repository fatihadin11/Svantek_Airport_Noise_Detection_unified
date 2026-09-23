import argparse
import os
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
            raise SystemExit("SVAN 971 bulunamadı.")

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
                f"USB interface meşgul: {e}\n"
                "VirtualBox USB menüsünden 0017:0001 tikini kaldırıp tekrar seç."
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

    def write_cmd(self, cmd):
        if not cmd.endswith(";"):
            cmd += ";"

        self.dev.write(
            OUT_ENDPOINT,
            cmd.encode("ascii"),
            timeout=3000,
        )

    def read_some(self, timeout=1000):
        try:
            return bytes(self.dev.read(IN_ENDPOINT, 4096, timeout=timeout))
        except usb.core.USBTimeoutError:
            return b""

    def read_block(self, disk, address, offset, nbytes):
        cmd = f"#D,r,{disk},{address},{offset},{nbytes};"
        self.write_cmd(cmd)

        buf = b""
        header_end = None
        deadline = time.time() + 8

        while time.time() < deadline:
            chunk = self.read_some(timeout=700)

            if chunk:
                buf += chunk

                if header_end is None and b";" in buf:
                    header_end = buf.find(b";") + 1

                if header_end is not None:
                    payload_len = len(buf) - header_end
                    if payload_len >= nbytes:
                        break

        if header_end is None:
            print("Ham cevap ilk 100 byte:", buf[:100])
            raise RuntimeError("Header bulunamadı.")

        header = buf[:header_end].decode("ascii", errors="replace")
        payload = buf[header_end:header_end + nbytes]

        if len(payload) < nbytes:
            raise RuntimeError(
                f"Eksik veri geldi. Header={header!r}, beklenen={nbytes}, gelen={len(payload)}"
            )

        return header, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disk", type=int, default=0)
    parser.add_argument("--cluster", type=int, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk", type=int, default=512)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    svan = SvantekUSB()

    try:
        offset = 0

        with open(args.output, "wb") as f:
            while offset < args.size:
                nbytes = min(args.chunk, args.size - offset)

                header, payload = svan.read_block(
                    disk=args.disk,
                    address=args.cluster,
                    offset=offset,
                    nbytes=nbytes,
                )

                f.write(payload)
                offset += len(payload)

                percent = offset * 100 / args.size
                print(f"\rİndiriliyor: {offset}/{args.size} byte %{percent:.1f}", end="", flush=True)

        print()
        print(f"Tamamlandı: {args.output}")

    finally:
        svan.close()


if __name__ == "__main__":
    main()
