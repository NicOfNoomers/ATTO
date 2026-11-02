"""Quick free-flow frequency scan, any pump (NM, 2 Nov 2026).

Why: a fixed priming script does not fit every pump. 261030-2 (prime_plus run 20261102_082858) pumped 72-80 uL/min
at 600 Hz but fell off above it (1000 Hz 41 -> 15, 1300 Hz 23 -> 12, 1600 Hz ~0), so 1000-1600 Hz priming steps sat
where it barely pumps. Scan first, then prime near the pump's own best frequency (make_config_prime_at.py --f <Hz>).

Outlet VENTED, no syringe pressure, 2.5 V on the RSD3305P, PPK2 on, Pump ID required (typed in; no default).
Free flow at 100, 150, 200, 300, 400, 500, 600, 700, 800, 1000, 1300, 1600 Hz, 10 s each, 5 s zeros, ascending.
Run it before AND after priming: if the peak moves up after priming, it was gas.

usage: python make_config_scan_quick.py [--address COM7 | host:5025 | ""] [--v 2500]   -> config_scan_quick.json
"""
import argparse
import json

ap = argparse.ArgumentParser(description="Quick free-flow frequency scan (any pump)")
ap.add_argument("--address", default="COM7", help="RSD3305P address (default COM7, the lab PC); '' = Power panel")
ap.add_argument("--v", type=int, default=2500, help="supply voltage in mV (0 < v <= 2500)")
args = ap.parse_args()
if not 0 < args.v <= 2500:
    raise SystemExit("--v must be between 1 and 2500 mV (the pump's limit)")

FREQS = [100, 150, 200, 300, 400, 500, 600, 700, 800, 1000, 1300, 1600]
DRIVE_S, ZERO_S = 10.0, 5.0


def zero(s):
    return {"type": "constant", "frequency_hz": 0, "duration_s": float(s), "repeat": 1, "settle_ms": 0}


def drive(hz, s, **meta):
    return {"type": "constant", "frequency_hz": int(hz), "duration_s": float(s), "repeat": 1, "settle_ms": 0, **meta}


supply = {"mode": "rsd3305p", "channel": 1, "current_limit_ma": 60, "max_mv": 2500, "initial_mv": args.v,
          "tolerance_mv": 50, "step_timeout_s": 10, "off_on_exit": True, "eol": "LF"}
if args.address:
    supply["address"] = args.address
tests = [zero(10.0), {"type": "voltage", "voltage_mv": args.v, "while_waiting": "zero", "settle_s": 5,
                      "note": f"{args.v / 1000:.2f} V by the supply"}, zero(ZERO_S)]
for f in FREQS:
    tests += [drive(f, DRIVE_S, section="scan", level_mv=args.v), zero(ZERO_S)]
cfg = {
    "port": "COM3", "baud": 115200, "read_timeout_s": 0.2,
    "start_time": None, "end_time": None,
    "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
    "require_pump_id": True,
    "tests": tests,
    "meta": {"name": "Quick frequency scan", "operator": "", "pump_id": "", "freqs_hz": FREQS,
             "description": (f"Quick free-flow frequency scan, any pump: outlet VENTED, no syringe pressure. "
                             f"{', '.join(map(str, FREQS))} Hz, {DRIVE_S:g} s each with {ZERO_S:g} s zeros, ascending, "
                             f"at {args.v / 1000:.2f} V (RSD3305P), PPK2 on. Type the Pump ID. Run it before AND after "
                             f"priming: if the peak moves up after priming, it was gas. Prime near the peak with "
                             f"make_config_prime_at.py --f <Hz>.")},
    "serial": {"protocol": "freq_only"},
    "protocol": "freq_only",
    "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
    "supply": supply,
    "power": {"enabled": True, "backend": "ppk2", "port": "auto", "bin_ms": 10.0, "watch_ma": 57,
              "watch_hold_ms": 300, "raw_every_min": 10, "raw_length_s": 2.0, "post_run_tail_s": 10.0},
}
with open("config_scan_quick.json", "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, indent=2)
nom = sum(t.get("duration_s", 0) for t in tests if t["type"] == "constant") + 5
print(f"config_scan_quick.json: {len(FREQS)} frequencies, {nom / 60:.1f} min + 10 s tail; supply "
      f"{args.address or 'from the Power panel'}")
