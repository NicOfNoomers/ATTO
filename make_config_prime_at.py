"""Priming at a pump's own best frequency (NM, 2 Nov 2026).

Why: each pump has its own response (261030-2 falls off above 600 Hz; 261030-1 pumped best at 1300 Hz this morning),
so fixed 1000-1600 Hz priming steps can sit where a pump barely pumps. Run config_scan_quick.json first, pick f near
the peak, then prime here around f.

Outlet VENTED, no syringe pressure, RSD3305P (default 2.5 V), PPK2 on, Pump ID required.
3 x [f 30 s, 0 Hz 5 s, round(0.8 f) 10 s, 0 Hz 5 s, round(1.2 f) 10 s, 0 Hz 10 s], then checks: f 20 s, 0 Hz 15 s,
600 Hz 20 s (the common reference across pumps), 0 Hz 15 s. About 5 min. Run the scan again afterwards: a peak
that moved up was gas.

usage: python make_config_prime_at.py --f 450 [--pump 261030-2] [--v 2500] [--address COM7 | host:5025 | ""]
  -> config_prime_at_<f>Hz.json (config_prime_at_<f>Hz_<v>mV.json when --v is not 2500)
"""
import argparse
import json

ap = argparse.ArgumentParser(description="Priming around a chosen frequency (per pump)")
ap.add_argument("--f", type=int, required=True, help="priming frequency in Hz (near the scan's peak)")
ap.add_argument("--pump", default="", help="pump ID written into the config (still asked for at the start)")
ap.add_argument("--v", type=int, default=2500, help="supply voltage in mV (0 < v <= 2500)")
ap.add_argument("--address", default="COM7", help="RSD3305P address (default COM7, the lab PC); '' = Power panel")
args = ap.parse_args()
if not 0 < args.v <= 2500:
    raise SystemExit("--v must be between 1 and 2500 mV (the pump's limit)")
F_MAX = 3000
f_lo, f_hi = round(0.8 * args.f), round(1.2 * args.f)
if not (1 <= f_lo and f_hi <= F_MAX):
    raise SystemExit(f"--f {args.f}: 0.8 f .. 1.2 f = {f_lo}..{f_hi} Hz must lie within 1..{F_MAX} Hz")

ROUND = [(args.f, 30.0), (0, 5.0), (f_lo, 10.0), (0, 5.0), (f_hi, 10.0), (0, 10.0)]
REPS = 3
CHECKS = [(args.f, 20.0), (0, 15.0), (600, 20.0), (0, 15.0)]


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
for r in range(REPS):
    for f, dur in ROUND:
        tests.append(zero(dur) if f == 0 else drive(f, dur, section="prime_at", level_mv=args.v, rep=r + 1))
for f, dur in CHECKS:
    tests.append(zero(dur) if f == 0 else drive(f, dur, section="prime_check", level_mv=args.v,
                                                reference=(f == 600)))
rnd = ", ".join(f"{f} Hz {d:g} s" for f, d in ROUND)
cfg = {
    "port": "COM3", "baud": 115200, "read_timeout_s": 0.2,
    "start_time": None, "end_time": None,
    "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
    "require_pump_id": True,
    "tests": tests,
    "meta": {"name": f"Prime at {args.f} Hz", "operator": "", "pump_id": args.pump, "prime_f_hz": args.f,
             "description": (f"Priming around {args.f} Hz (near this pump's best frequency from config_scan_quick.json): "
                             f"outlet VENTED, no syringe pressure. {REPS} x [{rnd}] at {args.v / 1000:.2f} V (RSD3305P), "
                             f"PPK2 on, then checks: {args.f} Hz 20 s and 600 Hz 20 s (common reference across pumps) "
                             f"with 15 s zeros. Type the Pump ID. Run the quick scan again afterwards: a peak that moved "
                             f"up was gas.")},
    "serial": {"protocol": "freq_only"},
    "protocol": "freq_only",
    "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
    "supply": supply,
    "power": {"enabled": True, "backend": "ppk2", "port": "auto", "bin_ms": 10.0, "watch_ma": 57,
              "watch_hold_ms": 300, "raw_every_min": 10, "raw_length_s": 2.0, "post_run_tail_s": 10.0},
}
name = f"config_prime_at_{args.f}Hz" + ("" if args.v == 2500 else f"_{args.v}mV") + ".json"
with open(name, "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, indent=2)
nom = sum(t.get("duration_s", 0) for t in tests if t["type"] == "constant") + 5
print(f"{name}: prime at {args.f} Hz (with {f_lo} and {f_hi} Hz), {nom / 60:.1f} min + 10 s tail; checks {args.f} and "
      f"600 Hz; pump {args.pump or '(typed at the start)'}; supply {args.address or 'from the Power panel'}")
