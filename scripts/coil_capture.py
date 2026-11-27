"""Capture coil current from the XIAO coil probe (firmware/coil_sense_xiao), alongside a run (NM, 27 Nov 2026).

Each capture: N samples of the differential shunt voltage at the measured rate (~100-200 kS/s), saved as
<out>/cap_<k>.bin (int16 LE counts) plus a row in <out>/captures.csv with the PC wall time (sync with a run via
meta.json 'created' + merged.csv t_s; see scripts/coil_fold.py) and quick numbers (mean, p-p in mA, clipping).
counts -> mA: count x (1000 mV / gain) / 2048 / R_shunt.

The pump driver is a XIAO too (same USB IDs) on COM3: this script refuses COM3 and checks the '?' reply
("COIL_SENSE_XIAO") before capturing.

usage (lab PC, py -3.10):
  py -3.10 scripts\\coil_capture.py --port COM9 --check                     # one capture, print numbers (gain check)
  py -3.10 scripts\\coil_capture.py --port COM9 --every 5 --duration 1500 --out pump_runs\\coil_<name>
      [--gain 8] [--n 10000] [--rshunt 1.0]
"""
import argparse
import csv
import datetime
import os
import struct
import sys
import time

import serial

ap = argparse.ArgumentParser(description="XIAO coil-current captures")
ap.add_argument("--port", required=True)
ap.add_argument("--gain", type=int, default=8, choices=(1, 2, 4, 8, 16))
ap.add_argument("--n", type=int, default=10000)
ap.add_argument("--rshunt", type=float, default=1.0, help="shunt (ohm)")
ap.add_argument("--check", action="store_true", help="one capture, print, nothing saved")
ap.add_argument("--every", type=float, default=5.0, help="seconds between captures")
ap.add_argument("--duration", type=float, default=60.0, help="total seconds")
ap.add_argument("--out", default=None)
a = ap.parse_args()
if a.port.upper() == "COM3":
    raise SystemExit("COM3 is the pump driver: refusing")
if not a.check and not a.out:
    raise SystemExit("--out <dir> is needed (or --check)")


def readline(s, timeout=3.0):
    t_end = time.time() + timeout
    line = b""
    while time.time() < t_end:
        ch = s.read(1)
        if ch:
            line += ch
            if ch == b"\n":
                return line.decode("ascii", "replace").strip()
    return line.decode("ascii", "replace").strip()


s = serial.Serial(a.port, 115200, timeout=0.2)
time.sleep(1.5)
s.reset_input_buffer()
s.write(b"?")
ident = readline(s)
if "COIL_SENSE_XIAO" not in ident:
    raise SystemExit(f"{a.port} did not answer as the coil probe: {ident!r}")
for cmd, want in ((f"g{a.gain}\n", "OK gain"), (f"n{a.n}\n", "OK n")):
    s.write(cmd.encode())
    r = readline(s)
    if not r.startswith(want):
        raise SystemExit(f"{cmd.strip()} -> {r!r}")
lsb_ma = 1000.0 / a.gain / 2048.0 / a.rshunt
print(f"probe: {ident}; gain {a.gain} -> {lsb_ma * 1000:.1f} uA per count, full scale +-{2048 * lsb_ma:.0f} mA")


def capture():
    s.reset_input_buffer()
    s.write(b"c")
    hdr = readline(s, 5.0)
    if not hdr.startswith("CAP"):
        raise RuntimeError(f"bad header {hdr!r}")
    kv = dict(x.split("=") for x in hdr.split()[1:])
    n = int(kv["n"])
    raw = b""
    t_end = time.time() + 10.0
    while len(raw) < 2 * n and time.time() < t_end:
        raw += s.read(2 * n - len(raw))
    tail = readline(s) + readline(s)
    if len(raw) != 2 * n or "END" not in tail:
        raise RuntimeError(f"short capture ({len(raw)} of {2 * n} bytes, tail {tail!r})")
    counts = struct.unpack(f"<{n}h", raw)
    fs = n / (int(kv["us"]) * 1e-6)
    return counts, fs, raw, kv


if a.check:
    counts, fs, _, kv = capture()
    ma = [c * lsb_ma for c in counts]
    clip = sum(1 for c in counts if abs(c) >= 2040)
    print(f"{len(counts)} samples at {fs / 1000:.1f} kS/s: mean {sum(ma) / len(ma):+.2f} mA, min {min(ma):+.1f}, "
          f"max {max(ma):+.1f} mA, clipped {clip}" + ("  -> lower --gain" if clip else ""))
    sys.exit(0)

os.makedirs(a.out, exist_ok=True)
idx_path = os.path.join(a.out, "captures.csv")
new = not os.path.exists(idx_path)
fh = open(idx_path, "a", newline="")
w = csv.writer(fh)
if new:
    w.writerow(["k", "wall_iso", "t_unix", "n", "fs_hz", "gain", "rshunt_ohm", "lsb_ma", "mean_ma", "min_ma", "max_ma",
                "clipped"])
k0 = sum(1 for f in os.listdir(a.out) if f.startswith("cap_"))
t_start = time.time()
k = k0
while time.time() - t_start < a.duration:
    t_cap = time.time()
    try:
        counts, fs, raw, kv = capture()
    except Exception as e:
        print(f"capture failed: {e}")
        time.sleep(a.every)
        continue
    with open(os.path.join(a.out, f"cap_{k:05d}.bin"), "wb") as f:
        f.write(raw)
    ma = [c * lsb_ma for c in counts]
    clip = sum(1 for c in counts if abs(c) >= 2040)
    w.writerow([k, datetime.datetime.fromtimestamp(t_cap).isoformat(timespec="milliseconds"), f"{t_cap:.3f}", len(counts),
                f"{fs:.1f}", a.gain, a.rshunt, f"{lsb_ma:.6f}", f"{sum(ma) / len(ma):.3f}", f"{min(ma):.2f}",
                f"{max(ma):.2f}", clip])
    fh.flush()
    print(f"cap {k}: {fs / 1000:.1f} kS/s, mean {sum(ma) / len(ma):+.2f} mA, p-p {max(ma) - min(ma):.1f} mA"
          + (f", CLIPPED {clip}" if clip else ""))
    k += 1
    time.sleep(max(0.0, a.every - (time.time() - t_cap)))
fh.close()
print(f"{k - k0} captures in {a.out}")
