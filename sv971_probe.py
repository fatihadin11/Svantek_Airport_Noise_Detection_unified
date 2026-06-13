"""
sv971_probe2.py  —  Windows WinAPI ile Svantek SV 971 Keşif
============================================================
libusb veya hidapi gerektirmez.
Windows Device Manager üzerinden cihazı bulur ve pipe'ları dener.

Çalıştırma:
    python sv971_probe2.py
"""

import sys
import time
import ctypes
import ctypes.wintypes as wt

if sys.platform != "win32":
    print("Bu script sadece Windows'ta çalışır.")
    sys.exit(1)

# ── Win32 sabitleri ───────────────────────────────────────────────────────────
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
GENERIC_READ         = 0x80000000
GENERIC_WRITE        = 0x40000000
OPEN_EXISTING        = 3
FILE_SHARE_READ      = 0x00000001
FILE_SHARE_WRITE     = 0x00000002
FILE_FLAG_OVERLAPPED = 0x40000000

kernel32 = ctypes.windll.kernel32
setupapi = ctypes.windll.setupapi

# ── 1. SetupAPI ile USB cihazları listele ─────────────────────────────────────
print("=== Sistemdeki USB cihazları (VID_0017 veya Svantek) ===")

try:
    import winreg
    # HKLM\SYSTEM\CurrentControlSet\Enum\USB altına bak
    base = r"SYSTEM\CurrentControlSet\Enum\USB"
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as usb_key:
        i = 0
        while True:
            try:
                vid_pid = winreg.EnumKey(usb_key, i)
                if "VID_0017" in vid_pid.upper() or "SVAN" in vid_pid.upper():
                    print(f"\n  [{vid_pid}]")
                    with winreg.OpenKey(usb_key, vid_pid) as vp_key:
                        j = 0
                        while True:
                            try:
                                instance = winreg.EnumKey(vp_key, j)
                                with winreg.OpenKey(vp_key, instance) as inst_key:
                                    try:
                                        desc, _ = winreg.QueryValueEx(inst_key, "DeviceDesc")
                                        print(f"    Instance: {instance}")
                                        print(f"    Desc:     {desc}")
                                    except FileNotFoundError:
                                        pass
                                    # DevicePath / SymbolicName
                                    for val in ["SymbolicName", "DeviceDesc", "Service", "Driver"]:
                                        try:
                                            v, _ = winreg.QueryValueEx(inst_key, val)
                                            print(f"    {val}: {v}")
                                        except FileNotFoundError:
                                            pass
                                    # \Device Parameters altına bak
                                    try:
                                        with winreg.OpenKey(inst_key, "Device Parameters") as dp:
                                            k = 0
                                            while True:
                                                try:
                                                    name, val, _ = winreg.EnumValue(dp, k)
                                                    print(f"    DevParam.{name}: {val}")
                                                    k += 1
                                                except OSError:
                                                    break
                                    except FileNotFoundError:
                                        pass
                                j += 1
                            except OSError:
                                break
                i += 1
            except OSError:
                break
except Exception as e:
    print(f"  Registry okuma hatası: {e}")

# ── 2. WMI ile USB cihaz yolu bul ─────────────────────────────────────────────
print("\n=== WMI USB sorgusu ===")
try:
    import subprocess
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         """
         Get-WmiObject Win32_USBControllerDevice | ForEach-Object {
             $dev = [wmi]($_.Dependent)
             if ($dev.DeviceID -like '*VID_0017*') {
                 Write-Output "DeviceID: $($dev.DeviceID)"
                 Write-Output "Name: $($dev.Name)"
                 Write-Output "Service: $($dev.Service)"
                 Write-Output "Status: $($dev.Status)"
                 Write-Output "---"
             }
         }
         """],
        capture_output=True, text=True, timeout=15
    )
    if result.stdout.strip():
        print(result.stdout)
    else:
        print("  WMI sorgusu sonuç döndürmedi.")
    if result.stderr.strip():
        print(f"  Stderr: {result.stderr[:200]}")
except Exception as e:
    print(f"  WMI hatası: {e}")

# ── 3. Cihaz symbolic link'i bul ve açmayı dene ───────────────────────────────
print("\n=== Cihaz yolu bulma (SetupDiGetDevicePath) ===")

DIGCF_PRESENT         = 0x02
DIGCF_DEVICEINTERFACE = 0x10

# USB device interface GUID (genel)
USB_DEVICE_GUIDS = [
    # WinUSB
    "{a5dcbf10-6530-11d2-901f-00c04fb951ed}",
    # libusb-win32 / genel USB cihaz
    "{f18a0e88-c30c-11d0-8815-00a0c906bed8}",
    # Svantek / SvUSB — vendor tanımlı olabilir, genel de dene
    "{36fc9e60-c465-11cf-8056-444553540000}",  # USB hub
    "{745a17a0-74d3-11d0-b6fe-00a0c90f57da}",  # HID
]

# setupapi ile tara
class GUID(ctypes.Structure):
    _fields_ = [("Data1", wt.DWORD),
                ("Data2", wt.WORD),
                ("Data3", wt.WORD),
                ("Data4", ctypes.c_byte * 8)]

class SP_DEVICE_INTERFACE_DATA(ctypes.Structure):
    _fields_ = [("cbSize",             wt.DWORD),
                ("InterfaceClassGuid", GUID),
                ("Flags",              wt.DWORD),
                ("Reserved",           ctypes.POINTER(ctypes.c_ulong))]

class SP_DEVINFO_DATA(ctypes.Structure):
    _fields_ = [("cbSize",    wt.DWORD),
                ("ClassGuid", GUID),
                ("DevInst",   wt.DWORD),
                ("Reserved",  ctypes.POINTER(ctypes.c_ulong))]

def str_to_guid(s):
    import re
    s = s.strip("{}")
    parts = re.split(r"[-]", s)
    d1 = int(parts[0], 16)
    d2 = int(parts[1], 16)
    d3 = int(parts[2], 16)
    d4 = bytes.fromhex(parts[3] + parts[4])
    g = GUID()
    g.Data1 = d1; g.Data2 = d2; g.Data3 = d3
    for i, b in enumerate(d4): g.Data4[i] = b
    return g

found_paths = []
for guid_str in USB_DEVICE_GUIDS:
    guid = str_to_guid(guid_str)
    hdev = setupapi.SetupDiGetClassDevsW(
        ctypes.byref(guid), None, None,
        DIGCF_PRESENT | DIGCF_DEVICEINTERFACE
    )
    if hdev == INVALID_HANDLE_VALUE:
        continue
    idx = 0
    while True:
        iface_data = SP_DEVICE_INTERFACE_DATA()
        iface_data.cbSize = ctypes.sizeof(SP_DEVICE_INTERFACE_DATA)
        ok = setupapi.SetupDiEnumDeviceInterfaces(
            hdev, None, ctypes.byref(guid), idx, ctypes.byref(iface_data)
        )
        if not ok:
            break
        # Yol boyutunu öğren
        size = wt.DWORD(0)
        setupapi.SetupDiGetDeviceInterfaceDetailW(
            hdev, ctypes.byref(iface_data), None, 0, ctypes.byref(size), None
        )
        if size.value == 0:
            idx += 1; continue
        # Buffer oluştur
        buf = ctypes.create_string_buffer(size.value)
        ctypes.cast(buf, ctypes.POINTER(wt.DWORD))[0] = 8  # cbSize (x64)
        ok2 = setupapi.SetupDiGetDeviceInterfaceDetailW(
            hdev, ctypes.byref(iface_data), buf, size, None, None
        )
        if ok2:
            path = ctypes.cast(
                ctypes.byref(buf, 4), ctypes.c_wchar_p
            ).value or ""
            if "0017" in path.lower() or "svan" in path.lower():
                print(f"  Yol bulundu: {path}")
                found_paths.append(path)
        idx += 1
    setupapi.SetupDiDestroyDeviceInfoList(hdev)

# ── 4. Bulunan yolları CreateFile ile açmayı dene ─────────────────────────────
if found_paths:
    print("\n=== CreateFile denemeleri ===")
    for path in found_paths:
        h = kernel32.CreateFileW(
            path,
            GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            None, OPEN_EXISTING, 0, None
        )
        if h != INVALID_HANDLE_VALUE:
            print(f"  ✓ Açıldı: {path}")
            # Küçük okuma denemesi
            buf = ctypes.create_string_buffer(64)
            read = wt.DWORD(0)
            ok = kernel32.ReadFile(h, buf, 64, ctypes.byref(read), None)
            if ok and read.value > 0:
                print(f"    {read.value} byte okundu: {buf.raw[:read.value].hex(' ')}")
            else:
                err = kernel32.GetLastError()
                print(f"    ReadFile başarısız (err={err}) — muhtemelen write-first protokol")
            kernel32.CloseHandle(h)
        else:
            err = kernel32.GetLastError()
            print(f"  ✗ Açılamadı (err={err}): {path[:80]}")
else:
    print("\n  SetupAPI ile yol bulunamadı.")

# ── 5. PowerShell ile cihaz yolu ─────────────────────────────────────────────
print("\n=== PowerShell ile tam cihaz yolu ===")
try:
    ps_cmd = r"""
    $ErrorActionPreference = 'SilentlyContinue'
    Get-PnpDevice | Where-Object { $_.InstanceId -like '*VID_0017*' } | ForEach-Object {
        Write-Output "InstanceId : $($_.InstanceId)"
        Write-Output "FriendlyName: $($_.FriendlyName)"
        Write-Output "Class: $($_.Class)"
        Write-Output "Status: $($_.Status)"
        $props = Get-PnpDeviceProperty -InstanceId $_.InstanceId -KeyName 'DEVPKEY_Device_PDOName' -EA SilentlyContinue
        if ($props) { Write-Output "PDOName: $($props.Data)" }
        Write-Output "---"
    }
    """
    r = subprocess.run(
        ["powershell", "-NoProfile", "-Command", ps_cmd],
        capture_output=True, text=True, timeout=20
    )
    print(r.stdout if r.stdout.strip() else "  Sonuç yok.")
    if r.stderr.strip():
        print(f"  Stderr: {r.stderr[:300]}")
except Exception as e:
    print(f"  Hata: {e}")

print("\n=== Tamamlandı ===")
print("Bu çıktıyı paylaş.")