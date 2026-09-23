import re
import struct
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

    def write(self, cmd):
        if not cmd.endswith(";"):
            cmd += ";"
        self.dev.write(OUT_ENDPOINT, cmd.encode("ascii"), timeout=3000)

    def read_some(self, timeout=1000):
        try:
            return bytes(self.dev.read(IN_ENDPOINT, 4096, timeout=timeout))
        except usb.core.USBTimeoutError:
            return b""

    def ascii_cmd(self, cmd):
        self.write(cmd)
        time.sleep(0.2)
        data = self.read_some(timeout=3000)
        return data.decode("ascii", errors="replace").strip()

    def read_block(self, disk, address, offset, nbytes):
        cmd = f"#D,r,{disk},{address},{offset},{nbytes};"
        self.write(cmd)

        buf = b""
        deadline = time.time() + 5

        while time.time() < deadline:
            chunk = self.read_some(timeout=700)
            if chunk:
                buf += chunk

            if b";" in buf:
                header_end = buf.find(b";") + 1
                payload_len = len(buf) - header_end
                if payload_len >= nbytes:
                    break

        if b";" not in buf:
            print("Ham cevap:", buf[:100])
            raise SystemExit("Header bulunamadı.")

        header_end = buf.find(b";") + 1
        header = buf[:header_end].decode("ascii", errors="replace")
        payload = buf[header_end:header_end + nbytes]

        return header, payload


def parse_entry(entry):
    first = entry[0]

    if first == 0x00:
        return "END", None

    if first == 0xE5:
        return "SKIP", None

    attr = entry[11]

    # Long filename entry
    if attr == 0x0F:
        return "SKIP", None

    name_raw = entry[0:8].decode("ascii", errors="replace").strip()
    ext_raw = entry[8:11].decode("ascii", errors="replace").strip()

    if not name_raw:
        return "SKIP", None

    full_name = name_raw
    if ext_raw:
        full_name += "." + ext_raw

    cluster_high = struct.unpack_from("<H", entry, 20)[0]
    cluster_low = struct.unpack_from("<H", entry, 26)[0]
    cluster = (cluster_high << 16) | cluster_low

    size = struct.unpack_from("<I", entry, 28)[0]

    is_dir = bool(attr & 0x10)

    return "OK", {
        "name": full_name,
        "type": "DIR" if is_dir else "FILE",
        "cluster": cluster,
        "size": size,
        "attr": attr,
    }


def main():
    svan = SvantekUSB()

    try:
        wd = svan.ascii_cmd("#D,d,?")
        print("Working directory:", wd)

        m = re.search(r"#D,d,(\d+),(\d+),(\d+);", wd)

        if not m:
            raise SystemExit("Working directory cevabı çözümlenemedi.")

        disk = int(m.group(1))
        address = int(m.group(2))
        count = int(m.group(3))

        print(f"Disk={disk}, Address={address}, Count={count}")
        print("\nDosyalar:\n")

        found_any = False

        offset = 0
        block_size = 512

        while offset < count:
            header, data = svan.read_block(disk, address, offset, block_size)

            for i in range(0, len(data), 32):
                entry = data[i:i + 32]

                if len(entry) < 32:
                    continue

                status, item = parse_entry(entry)

                if status == "END":
                    return

                if status == "OK":
                    found_any = True
                    print(
                        f"{item['type']:4} "
                        f"{item['name']:15} "
                        f"cluster={item['cluster']:<8} "
                        f"size={item['size']}"
                    )

            offset += block_size

        if not found_any:
            print("Dosya bulunamadı veya dizin parse edilemedi.")

    finally:
        svan.close()


if __name__ == "__main__":
    main()
