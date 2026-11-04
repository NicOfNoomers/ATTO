"""High-frequency bubble break with low-frequency checks, any pump (NM, 4 Nov 2026).

Why (4 Nov, pump 261030-1): flow fell to ~1/3 of the 1-2 Nov run at the same current (600 Hz 2.5 V: ~70-80 vs ~200
uL/min). Priming at 1000 and 1600 Hz (prime_at rounds) brought 1300-1600 Hz back to ~200 uL/min but left 100-1000 Hz
where they were: something (gas) still eats the large slow strokes. Nic: break the bubbles with the higher frequencies
and wait until it cleans up for the lower ones.

One run: REPS x [each break frequency BREAK_S on, 5 s off] (the on/off edges are the "burst" part), then checks at
600 / 300 / 100 Hz, 20 s each with 15 s zeros (offset reference), so every run shows whether the low end is coming
back. Repeat the run until 600 Hz recovers (scripts/flow_per_block.py on the run folder), then a quick scan.
Outlet VENTED, no syringe pressure, RSD3305P at 2.5 V, PPK2 on (trip 57 mA / 300 ms), Pump ID required.

usage: python make_config_bubble_break.py [--freqs 1300,1600,2000,2500] [--reps 3] [--break-s 25] [--v 2500]
                                          [--pump 261030-1] [--address COM7]      -> config_bubble_break.json
"""
import argparse
import json

ap = argparse.ArgumentParser(description="High-frequency bubble break + low-frequency checks")
ap.add_argument("--freqs", default="1300,1600,2000,2500", help="break frequencies in Hz (<= 3000)")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--break-s", type=float, default=25.0, help="seconds per break frequency per rep")
ap.add_argument("--checks", default="600,300,100", help="check frequencies at the end (Hz)")
ap.add_argument("--v", type=int, default=2500, help="supply voltage in mV (0 < v <= 2500)")
ap.add_argument("--pump", default="261030-1")
ap.add_argument("--address", default="COM7", help="RSD3305P address (default COM7, the lab PC); '' = Power panel")
ap.add_argument("--out", default="config_bubble_break.json")
args = ap.parse_args()
if not 0 < args.v <= 2500:
    raise SystemExit("--v must be between 1 and 2500 mV (the pump's limit)")
FREQS = [int(x) for x in args.freqs.split(",") if x.strip()]
CHECKS = [int(x) for x in args.checks.split(",") if x.strip()]
if not FREQS or not all(1 <= f <= 3000 for f in FREQS + CHECKS):
    raise SystemExit("frequencies must be within 1..3000 Hz")
OFF_S, CHECK_S, CHECK_ZERO_S = 5.0, 20.0, 15.0


def zero(s):
    return {"type": "constant", "frequency_hz": 0, "duration_s": float(s), "repeat": 1, "settle_ms": 0}


def drive(hz, s, **meta):
    return {"type": "constant", "frequency_hz": int(hz), "duration_s": float(s), "repeat": 1, "settle_ms": 0, **meta}


supply = {"mode": "rsd3305p", "channel": 1, "current_limit_ma": 60, "max_mv": 2500, "initial_mv": args.v,
          "tolerance_mv": 50, "step_timeout_s": 10, "off_on_exit": True, "eol": "LF"}
if args.address:
    supply["address"] = args.address
tests = [zero(10.0), {"type": "voltage", "voltage_mv": args.v, "while_waiting": "zero", "settle_s": 5,
                      "note": f"{args.v / 1000:.2f} V by the supply"}, zero(10.0)]
for r in range(args.reps):
    for f in FREQS:
        tests += [drive(f, args.break_s, section="break", level_mv=args.v, rep=r + 1), zero(OFF_S)]
tests.append(zero(CHECK_ZERO_S - OFF_S))
for f in CHECKS:
    tests += [drive(f, CHECK_S, section="check", level_mv=args.v, reference=(f == 600)), zero(CHECK_ZERO_S)]

cfg = {
    "port": "COM3", "baud": 115200, "read_timeout_s": 0.2,
    "start_time": None, "end_time": None,
    "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
    "require_pump_id": True,
    "tests": tests,
    "meta": {"name": "Bubble break", "operator": "", "pump_id": args.pump, "break_hz": FREQS, "check_hz": CHECKS,
             "description": (f"Bubble break: outlet VENTED, no syringe pressure. {args.reps} x [{', '.join(map(str, FREQS))} Hz, "
                             f"{args.break_s:g} s each, 5 s off] at {args.v / 1000:.2f} V (RSD3305P), PPK2 on, then checks "
                             f"{', '.join(map(str, CHECKS))} Hz 20 s with 15 s zeros. Repeat until 600 Hz recovers; then a "
                             f"quick scan.")},
    "serial": {"protocol": "freq_only"},
    "protocol": "freq_only",
    "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
    "supply": supply,
    "power": {"enabled": True, "backend": "ppk2", "port": "auto", "bin_ms": 10.0, "watch_ma": 57,
              "watch_hold_ms": 300, "raw_every_min": 10, "raw_length_s": 2.0, "post_run_tail_s": 10.0},
}
with open(args.out, "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, indent=2)
nom = sum(t.get("duration_s", 0) for t in tests if t["type"] == "constant") + 5
print(f"{args.out}: {args.reps} x break at {FREQS} Hz ({args.break_s:g} s each), checks {CHECKS} Hz; "
      f"{nom / 60:.1f} min + 10 s tail; pump {args.pump}; supply {args.address or 'from the Power panel'}")
