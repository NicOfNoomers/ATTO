"""Maximum-pressure point tracking config: the blocked outlet climbs to the frequency that holds the most pressure
(NM, 18 Nov 2026).

Why: on 18 Nov NM reached 378 mbar by hand at 650 Hz, 2.5 V, blocked outlet. That is between the 600 / 800 Hz
points of the fixed-grid max-pressure scans (make_config_maxp_scan.py), which also stopped every step at 270 mbar.
Fixed grids miss narrow peaks; the tracker (maxp_tracker.py, block type 'maxp_track') searches the frequency
instead of assuming it.

Per start frequency: gentle release (pull only, to the floor), hold_position (the syringe motor holds the plunger:
closed line), then the tracker:
  build to the plateau -> probe f +- step -> move if the pressure keeps rising there, else halve the step -> stop at
  step_min, max_s or stop_above -> hold the best frequency (the mean of the second half of the hold is the maximum)
  -> 10 s at 0 Hz (leak decay).
It finds the peak nearest each start, so give one start per suspected peak (e.g. 650,1250).

Limits:
  - The controller faults above min(--limit, 95 % of the pressure sensor's range): with a +-344.74 mbar
    (5 psi) sensor that is 327.5 mbar, whatever --limit says.
  - The tracker ends its drive 5 mbar below that (or at --stop-above if lower) and reports
    "maximum >= stop". A pump that holds more than the sensor can read needs a bigger sensor; that is a
    hardware change, not a config one.
  - PPK2 watch 57 mA / 300 ms. Supply: --supply manual (bench supply set by hand, default since 6 Nov) or
    rsd3305p (COM7).

Setup: prime first with the outlet vented. Connect the syringe through the stopcock, syringe EMPTY (the releases
only pull), stopcock closed to waste. Type the Pump ID. Headless runs need --on-prompt wait (syringe clears).
Results: maxp_track.csv (one row per window) and maxp_track_results.json (one result per start) in the run folder.

usage:
  python make_config_maxp_track.py --pump 261030-1 [--starts 650,1250] [--supply manual|rsd3305p] [--v 2500]
      [--fmin 200 --fmax 2000] [--step0 50 --step-min 5] [--probe-s 4 --settle-s 1] [--plateau-slope 0.5]
      [--gain-slope 1.0] [--build-max-s 40] [--max-s 600] [--stop-above 330] [--limit 340] [--hold-s 20]
      [--leak-s 10] [--release-rate 30 --release-floor 5] [--out <file>]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import maxp_tracker  # noqa: E402

ap = argparse.ArgumentParser(description="Maximum-pressure point tracking (blocked outlet), any pump")
ap.add_argument("--pump", required=True, help="pump ID (also passed to run_headless --pump-id)")
ap.add_argument("--starts", default="650", help="start frequencies, one tracking per start, e.g. 650,1250")
ap.add_argument("--supply", choices=("manual", "rsd3305p"), default="manual")
ap.add_argument("--address", default="COM7", help="RSD3305P address (--supply rsd3305p)")
ap.add_argument("--v", type=int, default=2500, help="supply voltage in mV (0 < v <= 2500)")
ap.add_argument("--fmin", type=int, default=200)
ap.add_argument("--fmax", type=int, default=2000)
ap.add_argument("--step0", type=float, default=50.0)
ap.add_argument("--step-min", type=float, default=5.0)
ap.add_argument("--probe-s", type=float, default=4.0)
ap.add_argument("--settle-s", type=float, default=1.0)
ap.add_argument("--plateau-slope", type=float, default=0.5, help="mbar/s")
ap.add_argument("--gain-slope", type=float, default=1.0, help="mbar/s")
ap.add_argument("--build-max-s", type=float, default=40.0)
ap.add_argument("--max-s", type=float, default=600.0, help="tracking time cap per start (hold and leak on top)")
ap.add_argument("--stop-above", type=float, default=330.0, help="mbar; also capped 5 mbar below the controller limit")
ap.add_argument("--limit", type=float, default=340.0, help="controller limit, mbar (it also caps at 95 %% of the sensor range)")
ap.add_argument("--hold-s", type=float, default=20.0)
ap.add_argument("--leak-s", type=float, default=10.0)
ap.add_argument("--release-rate", type=float, default=30.0, help="release ramp, mbar/s")
ap.add_argument("--release-floor", type=float, default=5.0, help="release stops at this pressure, mbar")
ap.add_argument("--out", default=None, help="output file (default config_maxp_track_<starts>Hz_<supply>.json)")
args = ap.parse_args()

if not 0 < args.v <= 2500:
    raise SystemExit("--v must be between 1 and 2500 mV (the pump's limit)")
if not 50.0 <= args.limit <= 400.0:
    raise SystemExit("--limit must be within 50..400 mbar")
if not (0.0 < args.release_floor <= 20.0 and 5.0 <= args.release_rate <= 40.0):
    raise SystemExit("--release-floor must be in (0, 20] mbar and --release-rate in [5, 40] mbar/s")
starts = [int(s) for s in args.starts.split(",") if s.strip()]
if not starts:
    raise SystemExit("--starts needs at least one frequency")

tests = [{"type": "constant", "frequency_hz": 0, "duration_s": 10.0, "repeat": 1, "settle_ms": 0}]
if args.supply == "rsd3305p":
    tests.insert(0, {"type": "voltage", "voltage_mv": args.v, "while_waiting": "zero", "settle_s": 5,
                     "note": f"{args.v / 1000:.2f} V by the supply"})
for f0 in starts:
    blk = {"type": "maxp_track", "start_hz": f0, "f_min": args.fmin, "f_max": args.fmax, "step0_hz": args.step0,
           "step_min_hz": args.step_min, "settle_s": args.settle_s, "probe_s": args.probe_s,
           "plateau_slope": args.plateau_slope, "gain_slope": args.gain_slope, "build_max_s": args.build_max_s,
           "max_s": args.max_s, "stop_above_mbar": args.stop_above, "hold_s": args.hold_s, "leak_s": args.leak_s,
           "section": "maxp_track", "level_mv": args.v}
    maxp_tracker.Params.from_block(blk)          # refuses bad values here, not on the rig
    tests += [
        {"type": "pressure", "target_mbar": "release", "timeout_s": 30.0,
         "release_rate_mbar_s": args.release_rate, "release_floor_mbar": args.release_floor},
        {"type": "pressure", "target_mbar": "hold_position", "min_pull_room": 0.3},
        blk,
        {"type": "constant", "frequency_hz": 0, "duration_s": 3.0, "repeat": 1, "settle_ms": 0},
    ]
tests.append({"type": "pressure", "target_mbar": "release", "timeout_s": 30.0,
              "release_rate_mbar_s": args.release_rate, "release_floor_mbar": args.release_floor})

if args.supply == "manual":
    supply = {"mode": "manual", "current_limit_ma": 60, "max_mv": 2500, "initial_mv": args.v, "lead_min": 2}
    supply_txt = f"MANUAL bench supply set by hand to {args.v / 1000:.2f} V (switch it off by hand after the run)"
else:
    supply = {"mode": "rsd3305p", "channel": 1, "current_limit_ma": 60, "max_mv": 2500, "initial_mv": args.v,
              "tolerance_mv": 50, "step_timeout_s": 10, "off_on_exit": True, "eol": "LF", "address": args.address}
    supply_txt = f"{args.v / 1000:.2f} V by the RSD3305P ({args.address})"

desc = (f"Maximum-pressure point tracking, blocked outlet, from {', '.join(str(s) for s in starts)} Hz "
        f"({args.fmin}-{args.fmax} Hz, step {args.step0:g} -> {args.step_min:g} Hz), {supply_txt}, PPK2 on. "
        f"Per start: gentle release, the syringe holds its position (closed line), the tracker climbs to the "
        f"frequency that holds the most pressure (stop at {args.stop_above:g} mbar or 5 mbar below the controller "
        f"limit, whichever is lower: maximum >= that), holds it {args.hold_s:g} s, then {args.leak_s:g} s at 0 Hz. "
        f"Results: maxp_track.csv, maxp_track_results.json. ORDER: 1) prime with the outlet vented; 2) syringe "
        f"through the stopcock, EMPTY, stopcock closed to waste; 3) this config. Type the Pump ID.")
cfg = {
    "port": "COM3", "baud": 115200, "read_timeout_s": 0.2, "start_time": None, "end_time": None,
    "break_seconds_between_tests": 0.0, "send_zero_on_exit": True, "require_pump_id": True,
    "meta": {"name": "Max-pressure tracking", "description": desc, "operator": "", "pump_id": args.pump,
             "starts_hz": starts, "syringe_ml": 1.0, "syringe_on": "outlet", "pressure_channel": 1},
    "serial": {"protocol": "freq_only"}, "protocol": "freq_only",
    "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
    "pressure": {"enabled": True, "backend": "arduino", "port": "COM6", "baud": 115200, "firmware": "auto",
                 "sensor": {"backend": "fluigent", "channel": 1}, "limit_mbar": args.limit, "min_mbar": -60.0,
                 "rate_mbar_s": 40.0, "tolerance_mbar": 2.0, "settle_s": 3.0, "microsteps": 16, "pos_min": 0,
                 "pos_max": 2470, "pos_margin": 30, "clear_reserve_steps": 150, "ul_per_step": 0.4049},
    "supply": supply,
    "power": {"enabled": True, "backend": "ppk2", "port": "auto", "bin_ms": 10.0, "watch_ma": 57,
              "watch_hold_ms": 300, "raw_every_min": 10, "raw_length_s": 2.0, "post_run_tail_s": 10.0},
    "tests": tests,
}
out = args.out or f"config_maxp_track_{'_'.join(str(s) for s in starts)}Hz_{args.supply}.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
n_track = len(starts)
print(f"wrote {out}: {len(tests)} blocks, {n_track} tracking(s); at most "
      f"{n_track * (args.max_s + args.hold_s + args.leak_s + 40) / 60:.0f} min (usually 3-5 min per start)")
