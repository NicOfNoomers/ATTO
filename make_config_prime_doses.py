"""Priming dose-response at a pump's peak frequency, any pump (NM via atto_test_configs, 5 Nov 2026).

Question: does the pump recover with more priming at its peak, how fast, and does the recovery hold at rest?
Why (pump 260930-2):
- 4 Nov: priming at 1600 then 1300 Hz brought 1300 Hz from 97 to 254 uL/min, ~65 % of its 30 Oct 351-425.
- 5 Nov: prime_plus pushed it back (1300 Hz checks 175 -> 128 -> 35). Its 2 Hz strokes hurt the high frequencies on
  two pumps now, so this test has no 2 Hz and no prime_plus.
- 600 Hz is this pump's dip; its peak band is 1000-1300 Hz.

Setup: outlet VENTED to waste, pump TILTED OUTLET-UP, no syringe pressure; RSD3305P (COM7) at 2.5 V, PPK2 on (trip
57 mA / 300 ms), Pump ID required. Run on the lab PC with video of the outlet:
    py -3.10 scripts\\run_headless.py config_prime_doses_1300Hz.json --name prime_doses_1300 --pump-id 260930-2 \\
        --video --camera 1 --show

Sequence
- Check panel C: each --checks frequency 10 s with 5 s zeros (a 5 s zero first, so every check has an offset
  reference on both sides), ~80 s.
- C0, then for each dose d (minutes): priming at f for d minutes, then C. Doses are separate blocks, so cumulative
  priming is 1, 3, 7, 15 min with the defaults (whole rounds; the exact minutes are printed and stored).
- Priming pattern as prime_at: rounds of [f 30 s, 0 Hz 5 s, round(0.8 f) 10 s, 0 Hz 5 s, round(1.2 f) 10 s,
  0 Hz 10 s] = 70 s, as many whole rounds as fit d minutes (at least one).
- Then --rest-min at 0 Hz (pump idle, everything logging), then C again: does the recovery decay at rest?
  --rest-min 0 skips it.
- Finally a config_scan_quick-style scan: 100 ... 1600 Hz (12 frequencies), 10 s each, 5 s zeros, ascending.
- Gate for 260930-2 (for reading the result, not an automatic stop): 1300 Hz >= ~250 uL/min in a check panel counts
  as recovered; its best is 351-425 (30 Oct).
Every block carries "section" (check / prime / rest / scan), check blocks "panel" (C0, C1 ..., C_rest), the dose and
the cumulative priming minutes, so scripts/flow_per_block.py rows can be grouped per panel.

usage: python make_config_prime_doses.py --f 1300 [--doses 1,2,4,8] [--checks 400,700,1000,1300,1600]
                                         [--rest-min 10] [--v 2500] [--pump 260930-2] [--address COM7]
    -> config_prime_doses_<f>Hz.json
"""
import argparse
import json

ap = argparse.ArgumentParser(description="Priming dose-response at a pump's peak frequency")
ap.add_argument("--f", type=int, default=1300, help="priming frequency in Hz (the pump's peak)")
ap.add_argument("--doses", default="1,2,4,8", help="priming doses in minutes, in order")
ap.add_argument("--checks", default="400,700,1000,1300,1600", help="check panel frequencies (Hz)")
ap.add_argument("--rest-min", type=float, default=10.0, help="rest at 0 Hz before the last panel (0 = none)")
ap.add_argument("--v", type=int, default=2500, help="supply voltage in mV (0 < v <= 2500)")
ap.add_argument("--pump", default="260930-2", help="pump ID written into the config (the runner passes --pump-id too)")
ap.add_argument("--address", default="COM7", help="RSD3305P address (default COM7, the lab PC); '' = Power panel")
args = ap.parse_args()

F_MAX = 3000
if not 0 < args.v <= 2500:
    raise SystemExit("--v must be between 1 and 2500 mV (the pump's limit)")
DOSES = [float(x) for x in args.doses.split(",") if x.strip()]
CHECKS = [int(x) for x in args.checks.split(",") if x.strip()]
f_lo, f_hi = round(0.8 * args.f), round(1.2 * args.f)
if not DOSES or min(DOSES) <= 0:
    raise SystemExit("--doses must be positive minutes")
if not (1 <= f_lo and f_hi <= F_MAX) or not all(1 <= f <= F_MAX for f in CHECKS):
    raise SystemExit(f"frequencies must lie within 1..{F_MAX} Hz (0.8 f .. 1.2 f = {f_lo}..{f_hi})")
if args.rest_min < 0:
    raise SystemExit("--rest-min must be >= 0")

ROUND = [(args.f, 30.0), (0, 5.0), (f_lo, 10.0), (0, 5.0), (f_hi, 10.0), (0, 10.0)]
ROUND_S = sum(d for _, d in ROUND)
CHECK_S, CHECK_ZERO_S = 10.0, 5.0
SCAN = [100, 150, 200, 300, 400, 500, 600, 700, 800, 1000, 1300, 1600]
SCAN_S, SCAN_ZERO_S = 10.0, 5.0


def zero(s, **meta):
    return {"type": "constant", "frequency_hz": 0, "duration_s": float(s), "repeat": 1, "settle_ms": 0, **meta}


def drive(hz, s, **meta):
    return {"type": "constant", "frequency_hz": int(hz), "duration_s": float(s), "repeat": 1, "settle_ms": 0, **meta}


def panel(name, dose_min, cum_min):
    meta = {"section": "check", "panel": name, "dose_min": dose_min, "cumulative_prime_min": cum_min,
            "level_mv": args.v}
    out = [zero(CHECK_ZERO_S)]
    for f in CHECKS:
        out += [drive(f, CHECK_S, **meta), zero(CHECK_ZERO_S)]
    return out


supply = {"mode": "rsd3305p", "channel": 1, "current_limit_ma": 60, "max_mv": 2500, "initial_mv": args.v,
          "tolerance_mv": 50, "step_timeout_s": 10, "off_on_exit": True, "eol": "LF"}
if args.address:
    supply["address"] = args.address
tests = [zero(10.0), {"type": "voltage", "voltage_mv": args.v, "while_waiting": "zero", "settle_s": 5,
                      "note": f"{args.v / 1000:.2f} V by the supply"}, zero(10.0)]
tests += panel("C0", 0.0, 0.0)
cum = 0.0
dose_table = []
for k, d in enumerate(DOSES, start=1):
    n = max(1, round(d * 60.0 / ROUND_S))
    actual = n * ROUND_S / 60.0
    cum += actual
    dose_table.append((d, n, actual, cum))
    for r in range(n):
        for f, dur in ROUND:
            tests.append(zero(dur) if f == 0 else
                         drive(f, dur, section="prime", dose=k, dose_min=round(actual, 2), round=r + 1, level_mv=args.v))
    tests += panel(f"C{k}", round(actual, 2), round(cum, 2))
if args.rest_min > 0:
    tests.append(zero(args.rest_min * 60.0, section="rest", rest_min=args.rest_min))
    tests += panel("C_rest", 0.0, round(cum, 2))
tests.append(zero(SCAN_ZERO_S))
for f in SCAN:
    tests += [drive(f, SCAN_S, section="scan", level_mv=args.v), zero(SCAN_ZERO_S)]

cfg = {
    "port": "COM3", "baud": 115200, "read_timeout_s": 0.2,
    "start_time": None, "end_time": None,
    "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
    "require_pump_id": True,
    "tests": tests,
    "meta": {"name": f"Priming doses at {args.f} Hz", "operator": "", "pump_id": args.pump, "prime_f_hz": args.f,
             "doses_min": [round(a, 2) for _, _, a, _ in dose_table],
             "cumulative_prime_min": [round(c, 2) for _, _, _, c in dose_table],
             "check_hz": CHECKS, "rest_min": args.rest_min, "gate": "1300 Hz >= ~250 uL/min in a check panel "
                                                                     "(260930-2; best 351-425 on 30 Oct)",
             "description": (f"PRIMING DOSE-RESPONSE at {args.f} Hz: outlet VENTED to waste, pump TILTED OUTLET-UP, "
                             f"no syringe pressure, {args.v / 1000:.2f} V (RSD3305P), PPK2 on. Check panel "
                             f"({', '.join(map(str, CHECKS))} Hz, 10 s each, 5 s zeros) before priming and after each "
                             f"dose ({', '.join(f'{a:.2f}' for _, _, a, _ in dose_table)} min of [{args.f} Hz 30 s, "
                             f"{f_lo} Hz 10 s, {f_hi} Hz 10 s] rounds; cumulative "
                             f"{', '.join(f'{c:.2f}' for _, _, _, c in dose_table)} min)"
                             + (f", then {args.rest_min:g} min rest at 0 Hz and the panel again" if args.rest_min > 0 else "")
                             + ", then a 100-1600 Hz scan. Recovered for 260930-2: 1300 Hz >= ~250 uL/min. "
                               "No 2 Hz strokes (they hurt the high frequencies).")},
    "serial": {"protocol": "freq_only"},
    "protocol": "freq_only",
    "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
    "supply": supply,
    "power": {"enabled": True, "backend": "ppk2", "port": "auto", "bin_ms": 10.0, "watch_ma": 57,
              "watch_hold_ms": 300, "raw_every_min": 10, "raw_length_s": 2.0, "post_run_tail_s": 10.0},
    "video": {"codec": "mjpg", "quality": 95, "overlay": True, "chunk_s": 60.0, "crop": None},
}
name = f"config_prime_doses_{args.f}Hz.json"
with open(name, "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, indent=2)
total = sum(t.get("duration_s", 0) for t in tests if t["type"] == "constant") + 5
print(f"{name}: {len(tests)} blocks, {total / 60:.1f} min + 10 s tail; pump {args.pump}; supply {args.address or 'Power panel'}")
for d, n, a, c in dose_table:
    print(f"  dose {d:g} min -> {n} rounds = {a:.2f} min (cumulative {c:.2f} min)")
print(f"  {len(DOSES) + 1 + (1 if args.rest_min > 0 else 0)} check panels; rest {args.rest_min:g} min; final scan 12 frequencies")
