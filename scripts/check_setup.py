#!/usr/bin/env python3
"""Check the dashboard's Python packages and list the serial devices (PPK2, Arduino).

    python scripts/check_setup.py              # packages, ports, and ask the PPK2 for its calibration
    python scripts/check_setup.py --no-connect # don't open the PPK2

Talking to the PPK2 only reads its calibration data; it does not switch its
output on. Close the dashboard and the nRF Connect Power Profiler app first
(only one program can use the port).
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# (import name, pip name, needed for)
PACKAGES = [
    ("tkinter", "(part of Python)", "the dashboard window"),
    ("serial", "pyserial", "Arduino and PPK2"),
    ("numpy", "numpy", "everything"),
    ("matplotlib", "matplotlib", "plots"),
    ("PIL", "Pillow", "camera preview"),
    ("ppk2_api", "ppk2-api", "the PPK2 power meter"),
    ("Fluigent.SDK", "fluigent_sdk-23.0.0 folder", "the Fluigent flow sensor"),
    ("cv2", "opencv-python", "camera (optional)"),
    ("psutil", "psutil", "memory watchdog (optional)"),
    ("websockets", "websockets", "network flow backend (optional)"),
]
REQUIRED = {"tkinter", "serial", "numpy", "matplotlib", "ppk2_api"}
ARDUINO_VIDS = {0x2341: "Arduino", 0x2A03: "Arduino", 0x1A86: "CH340 (Arduino clone)",
                0x0403: "FTDI", 0x10C4: "CP210x"}


def main():
    print(f"Python {sys.version.split()[0]}  ({sys.executable})\n")
    missing = []
    for mod, pip_name, why in PACKAGES:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "") or getattr(m, "VERSION", "")
            print(f"  OK       {mod:<14} {str(ver):<10} {why}")
        except Exception as e:
            flag = "MISSING " if mod in REQUIRED else "missing "
            print(f"  {flag} {mod:<14} {'':<10} {why}  -> pip install {pip_name}  ({type(e).__name__})")
            if mod in REQUIRED:
                missing.append(pip_name)

    print("\nSerial ports:")
    ppk2 = []
    try:
        import serial.tools.list_ports
        from power_ppk2 import find_ppk2_ports
        ppk2 = find_ppk2_ports()
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            vidpid = f"{p.vid:04x}:{p.pid:04x}" if p.vid is not None else "    -    "
            tag = "<- PPK2" if p.device in ppk2 else (f"<- {ARDUINO_VIDS[p.vid]}?" if p.vid in ARDUINO_VIDS else "")
            print(f"  {p.device:<7} {vidpid}  {p.description:<45} {tag}")
        if not ports:
            print("  (none)")
    except Exception as e:
        print(f"  could not list ports: {e!r}")

    if not ppk2:
        print("\nNo PPK2 found. Plug it in with the USB DATA/POWER port and switch it on.")
    elif "--no-connect" not in sys.argv:
        print(f"\nAsking the PPK2 on {ppk2[0]} for its calibration data (output stays off) ...")
        try:
            from ppk2_api.ppk2_api import PPK2_API
            ppk = PPK2_API(ppk2[0], timeout=1, write_timeout=1, exclusive=True)
            try:
                ok = ppk.get_modifiers()
                mods = ppk.modifiers
                print(f"  {'OK' if ok else 'NO ANSWER'}: calibrated={mods.get('Calibrated')} HW={mods.get('HW')} "
                      f"R={[round(float(mods['R'][str(i)]), 3) for i in range(5)]}")
                if not ok:
                    print("  Update the PPK2 firmware once with nRF Connect for Desktop -> Power Profiler, then retry.")
            finally:
                ppk.ser.close()
        except Exception as e:
            print(f"  could not talk to the PPK2: {e}\n  (Is the dashboard or nRF Connect Power Profiler using it?)")

    if missing:
        print(f"\nMissing required packages: {', '.join(missing)} -> run install.bat")
        return 1
    print("\nSetup OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
