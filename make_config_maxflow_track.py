"""Maximum-flow point tracking config with flushes, free flow (NM, 18 Nov 2026).

Why: NM expects the maximum-flow frequency to sit near the maximum-pressure one, and free flow answers in
seconds. So first track the frequency that gives the most free flow (this config, block 'maxflow_track',
maxflow_tracker.py), then run the pressure tracker there (make_config_maxp_track.py --starts <that Hz>).
Flushes are built in, as NM asked: prime_at-style rounds at the start (--flush-rounds), one round at the
current best frequency every --flush-every-s, and one when the flow falls below --drop-frac x the best.

Per start: 10 s at 0 Hz (zero), the tracker, 10 s at 0 Hz. Each tracker:
  flush -> measure the centre -> probe f +- step, each compared with the centre measured just before and just
  after it -> move if it beats the centre by max(3 uL/min, 3 %), else halve the step -> stop at step_min or
  max_s -> hold the tracked frequency (its mean flow is the result).
Roughly 10-15 min per start. One start per suspected peak (e.g. 650,1250).

Setup: outlet VENTED (free flow, no syringe pressure), flow sensor in line. PPK2 watch 57 mA / 300 ms.
--supply manual (bench supply by hand; the default since 6 Nov) or rsd3305p (COM7). Type the Pump ID.
Results: maxflow_track.csv (one row per window and flush block) and maxflow_track_results.json.

usage:
  python make_config_maxflow_track.py --pump <id> [--starts 650,1250] [--supply manual|rsd3305p] [--v 2500]
      [--fmin 200 --fmax 2000] [--step0 50 --step-min 5] [--settle-s 3 --window-s 10]
      [--gain-abs 3 --gain-frac 0.03] [--max-s 1800] [--hold-s 60]
      [--flush-rounds 3] [--flush-every-s 300] [--flush-extra 1300] [--drop-frac 0.8] [--out <file>]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maxflow_tracker  # noqa: E402

ap = argparse.ArgumentParser(description="Maximum-flow point tracking with flushes (free flow), any pump")
ap.add_argument("--pump", required=True)
ap.add_argument("--starts", default="650")
ap.add_argument("--supply", choices=("manual", "rsd3305p"), default="manual")
ap.add_argument("--address", default="COM7")
ap.add_argument("--v", type=int, default=2500)
ap.add_argument("--fmin", type=int, default=200)
ap.add_argument("--fmax", type=int, default=2000)
ap.add_argument("--step0", type=float, default=50.0)
ap.add_argument("--step-min", type=float, default=5.0)
ap.add_argument("--settle-s", type=float, default=3.0)
ap.add_argument("--window-s", type=float, default=10.0)
ap.add_argument("--gain-abs", type=float, default=3.0, help="uL/min")
ap.add_argument("--gain-frac", type=float, default=0.03)
ap.add_argument("--max-s", type=float, default=1800.0)
ap.add_argument("--hold-s", type=float, default=60.0)
ap.add_argument("--flush-rounds", type=int, default=3)
ap.add_argument("--flush-every-s", type=float, default=300.0)
ap.add_argument("--flush-extra", default="", help="extra flush frequencies, e.g. 1300 or 1300,1600")
ap.add_argument("--drop-frac", type=float, default=0.8)
ap.add_argument("--out", default=None)
args = ap.parse_args()

if not 0 < args.v <= 2500:
    raise SystemExit("--v must be between 1 and 2500 mV (the pump's limit)")
starts = [int(s) for s in args.starts.split(",") if s.strip()]
extra = [int(s) for s in args.flush_extra.split(",") if s.strip()]
if not starts:
    raise SystemExit("--starts needs at least one frequency")

tests = []
if args.supply == "rsd3305p":
    tests.append({"type": "voltage", "voltage_mv": args.v, "while_waiting": "zero", "settle_s": 5,
                  "note": f"{args.v / 1000:.2f} V by the supply"})
for f0 in starts:
    blk = {"type": "maxflow_track", "start_hz": f0, "f_min": args.fmin, "f_max": args.fmax,
           "step0_hz": args.step0, "step_min_hz": args.step_min, "settle_s": args.settle_s,
           "window_s": args.window_s, "gain_abs": args.gain_abs, "gain_frac": args.gain_frac,
           "max_s": args.max_s, "hold_s": args.hold_s, "flush_rounds": args.flush_rounds,
           "flush_every_s": args.flush_every_s, "flush_extra_hz": extra, "drop_frac": args.drop_frac,
           "section": "maxflow_track", "level_mv": args.v}
    maxflow_tracker.FlowParams.from_block(blk)        # refuses bad values here, not on the rig
    tests += [{"type": "constant", "frequency_hz": 0, "duration_s": 10.0, "repeat": 1, "settle_ms": 0},
              blk,
              {"type": "constant", "frequency_hz": 0, "duration_s": 10.0, "repeat": 1, "settle_ms": 0}]

if args.supply == "manual":
    supply = {"mode": "manual", "current_limit_ma": 60, "max_mv": 2500, "initial_mv": args.v, "lead_min": 2}
    supply_txt = f"MANUAL bench supply set by hand to {args.v / 1000:.2f} V (switch it off by hand after the run)"
else:
    supply = {"mode": "rsd3305p", "channel": 1, "current_limit_ma": 60, "max_mv": 2500, "initial_mv": args.v,
              "tolerance_mv": 50, "step_timeout_s": 10, "off_on_exit": True, "eol": "LF", "address": args.address}
    supply_txt = f"{args.v / 1000:.2f} V by the RSD3305P ({args.address})"

desc = (f"Maximum-flow point tracking with flushes, free flow, from {', '.join(str(s) for s in starts)} Hz "
        f"({args.fmin}-{args.fmax} Hz, step {args.step0:g} -> {args.step_min:g} Hz), {supply_txt}, PPK2 on. "
        f"Flush {args.flush_rounds} round(s) at the start, 1 round every {args.flush_every_s:g} s and after a drop "
        f"below {args.drop_frac:g} x best; each probe is compared with the centre measured before and after it. "
        f"Result: the tracked frequency held {args.hold_s:g} s. Outlet VENTED, no syringe. Type the Pump ID. "
        f"Next: make_config_maxp_track.py --starts <the tracked Hz>.")
cfg = {
    "port": "COM3", "baud": 115200, "read_timeout_s": 0.2, "start_time": None, "end_time": None,
    "break_seconds_between_tests": 0.0, "send_zero_on_exit": True, "require_pump_id": True,
    "meta": {"name": "Max-flow tracking", "description": desc, "operator": "", "pump_id": args.pump,
             "starts_hz": starts},
    "serial": {"protocol": "freq_only"}, "protocol": "freq_only",
    "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
    "supply": supply,
    "power": {"enabled": True, "backend": "ppk2", "port": "auto", "bin_ms": 10.0, "watch_ma": 57,
              "watch_hold_ms": 300, "raw_every_min": 10, "raw_length_s": 2.0, "post_run_tail_s": 10.0},
    "tests": tests,
}
out = args.out or f"config_maxflow_track_{'_'.join(str(s) for s in starts)}Hz_{args.supply}.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
print(f"wrote {out}: {len(tests)} blocks, {len(starts)} tracking(s); at most "
      f"{len(starts) * (args.max_s + args.hold_s + 20) / 60:.0f} min (usually 10-15 min per start)")
