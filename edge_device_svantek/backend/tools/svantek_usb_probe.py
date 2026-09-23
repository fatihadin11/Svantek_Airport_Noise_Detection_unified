import usb.core
import usb.util
import time

VID = 0x0017
PID = 0x0001

dev = usb.core.find(idVendor=VID, idProduct=PID)

if dev is None:
    raise SystemExit("SVAN 971 USB cihazı bulunamadı.")

print("SVAN 971 bulundu.")
print(dev)

try:
    dev.set_configuration()
except usb.core.USBError as e:
    print("set_configuration uyarısı:", e)

cfg = dev.get_active_configuration()
print("Aktif configuration:", cfg.bConfigurationValue)

intf = cfg[(0, 0)]
print("Interface:", intf.bInterfaceNumber, "Alt:", intf.bAlternateSetting)

# Kernel driver bağlıysa ayırmayı dene
try:
    if dev.is_kernel_driver_active(intf.bInterfaceNumber):
        print("Kernel driver aktif, ayrılıyor...")
        dev.detach_kernel_driver(intf.bInterfaceNumber)
except (NotImplementedError, usb.core.USBError) as e:
    print("Kernel driver kontrol uyarısı:", e)

out_ep = None
in_ep = None

for ep in intf:
    direction = usb.util.endpoint_direction(ep.bEndpointAddress)
    endpoint_type = usb.util.endpoint_type(ep.bmAttributes)

    print(
        "Endpoint:",
        hex(ep.bEndpointAddress),
        "direction:",
        "IN" if direction == usb.util.ENDPOINT_IN else "OUT",
        "type:",
        endpoint_type,
        "maxpacket:",
        ep.wMaxPacketSize,
    )

    if direction == usb.util.ENDPOINT_OUT:
        out_ep = ep
    elif direction == usb.util.ENDPOINT_IN:
        in_ep = ep

if out_ep is None or in_ep is None:
    raise SystemExit("IN/OUT endpoint bulunamadı.")

cmd = b"#1,S?;"
print("Gönderiliyor:", cmd)

try:
    written = out_ep.write(cmd, timeout=2000)
    print("Yazılan byte:", written)
except usb.core.USBError as e:
    raise SystemExit(f"Yazma hatası: {e}")

time.sleep(0.3)

try:
    data = in_ep.read(512, timeout=3000)
    raw = bytes(data)
    print("Cevap raw:", raw)
    print("Cevap text:", raw.decode("ascii", errors="replace"))
except usb.core.USBError as e:
    print("Cevap okunamadı:", e)
