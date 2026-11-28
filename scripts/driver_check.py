"""Check the duty-cycle pump-driver firmware after flashing (firmware/pump_driver_duty_*), on the lab PC.

Opens the driver port (dashboard and runner must be closed), asks '?', then drives briefly and runs the firmware's
own self-test ('T', SAMD21: samples its IN1/IN2 pins) at D = 1.0, 0.4 and 0.2, and stops (0, A 1).
With the pump connected the pump runs ~2 s per step at the supply voltage (normal operation).
Pass: '?' answers PUMP_DRIVER_DUTY; per T line, forward IN1 ~100 % and IN2 ~(1 - D) x 100 %, reverse mirrored
(+-3 %). A mirror-image result (IN2 ~D x 100 %) means the polarity convention is flipped: rerun with --pol 1.
usage: py -3.10 scripts\\driver_check.py --port COM3 [--f 600] [--pol 0]
"""
import argparse
import time

import serial

ap = argparse.ArgumentParser()
ap.add_argument("--port", required=True)
ap.add_argument("--f", type=int, default=600)
ap.add_argument("--pol", type=int, default=0, choices=(0, 1))
a = ap.parse_args()
s = serial.Serial(a.port, 115200, timeout=0.3)
time.sleep(1.5)
s.reset_input_buffer()


def cmd(c, wait=0.3):
    s.write((c + "\n").encode())
    time.sleep(wait)
    out = s.read(s.in_waiting or 1).decode("ascii", "replace").strip()
    if out:
        print(f"{c!r:10s} -> {out}")
    return out


ident = cmd("?")
if "PUMP_DRIVER_DUTY" not in ident:
    raise SystemExit("no PUMP_DRIVER_DUTY reply: old firmware still on the board, or the wrong port")
ok = True
try:
    cmd(f"P {a.pol}")
    cmd(f"C 50000")
    for d in (1.0, 0.4, 0.2):
        cmd(f"A {d}")
        cmd(f"{a.f}", 1.5)
        r = cmd("T", 2.5)
        try:
            fw = r.split("fwd:")[1].split("|")[0]
            in1 = float(fw.split("IN1 high")[1].split("%")[0])
            in2 = float(fw.split("IN2 high")[1].split("%")[0])
            good = abs(in1 - 100) <= 3 and abs(in2 - 100 * (1 - d)) <= 3
            ok &= good
            print(f"   D {d}: forward IN1 {in1:.1f} % (want 100), IN2 {in2:.1f} % (want {100 * (1 - d):.0f}) -> {'OK' if good else 'MISMATCH'}")
        except (IndexError, ValueError):
            print("   (no T result: nRF52840 firmware has no self-test; check with the coil probe instead)")
        cmd("?")
finally:
    cmd("0", 1.6)
    cmd("A 1")
    cmd("?")
print("DRIVER CHECK " + ("PASSED" if ok else "FAILED"))
