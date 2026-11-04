"""Maximum-pressure frequency scan on the dead-head mechanics, any pump (NM, 3 Nov 2026).

Why: the 2 Nov dead-head steps on 261030-1 were coarse (600 / 800 / 1000 / 1300 / 1600 Hz -> 159 / unstable <=129 /
111 still rising / 125-129 / 103 mbar, while the morning PQ reached 181 at 1300 Hz): peaks narrower than 200-300 Hz
are missed, and the best frequency moves with the pump state, so scan fast within one run. Most rises level off
within 3-5 s, so 12 s drives.

Each frequency: gentle release (pull only, stop at the floor), hold_position (motor energised, no regulation; a
syringe clear first if less than 30 % of the syringe is left to pull back), drive 12 s with stop_above_mbar 270
(below the 280 mbar limit, logged as stop_above), 0 Hz 3 s (leak decay, logged), gentle release, 3 s at 0 Hz.
Release: 30 mbar/s towards 5 mbar, stops as soon as it is reached, never pushes (15 mbar/s as in the dead-head
config would take 10-18 s from 150-270 mbar). --release-rate / --release-floor change it.

Syringe through the stopcock, connected, EMPTY at the start (the releases only pull; each hold needs pull room).
RSD3305P (COM7), PPK2 on, Pump ID required (typed in; no default). Prime first (outlet vented), at the pump's own
peak if known (config_scan_quick.json + make_config_prime_at.py). Open the stopcock to waste before any hand push.

usage:
  python make_config_maxp_scan.py --coarse [--fmin 300 --fmax 2000 --step 100] [--v 2500] [--pump <id>]
      -> config_maxp_coarse.json (the first frequency repeated at the end)
  python make_config_maxp_scan.py --fine --center 1300[,600] [--span 200] [--step 20] [--v 2500] [--pump <id>]
      -> config_maxp_fine_<center>Hz.json (centres joined with _; each centre repeated at the end)
  common: [--address COM7 | host:5025 | ""] [--release-rate 30] [--release-floor 5]
"""
import argparse
import json

ap = argparse.ArgumentParser(description="Maximum-pressure frequency scan (dead-head steps), any pump")
mode = ap.add_mutually_exclusive_group(required=True)
mode.add_argument("--coarse", action="store_true")
mode.add_argument("--fine", action="store_true")
ap.add_argument("--fmin", type=int, default=300)
ap.add_argument("--fmax", type=int, default=2000)
ap.add_argument("--step", type=int, default=None, help="Hz (default 100 coarse, 20 fine)")
ap.add_argument("--center", default=None, help="fine: centre frequency or list, e.g. 600,1300")
ap.add_argument("--span", type=int, default=200, help="fine: full span around each centre (Hz)")
ap.add_argument("--v", type=int, default=2500, help="supply voltage in mV (0 < v <= 2500)")
ap.add_argument("--pump", default="", help="pump ID written into the config (still asked for at the start)")
ap.add_argument("--address", default="COM7", help="RSD3305P address (default COM7, the lab PC); '' = Power panel")
ap.add_argument("--release-rate", type=float, default=30.0, help="release ramp, mbar/s")
ap.add_argument("--release-floor", type=float, default=5.0, help="release stops at this pressure, mbar")
args = ap.parse_args()

F_MIN, F_MAX = 1, 3000
if not 0 < args.v <= 2500:
    raise SystemExit("--v must be between 1 and 2500 mV (the pump's limit)")
if not (0.0 < args.release_floor <= 20.0 and 5.0 <= args.release_rate <= 40.0):
    raise SystemExit("--release-floor must be in (0, 20] mbar and --release-rate in [5, 40] mbar/s")


def check_f(fs):
    for f in fs:
        if not F_MIN <= f <= F_MAX:
            raise SystemExit(f"{f} Hz is outside {F_MIN}..{F_MAX} Hz")
    return fs


if args.coarse:
    step = args.step or 100
    if step <= 0 or args.fmax < args.fmin:
        raise SystemExit("--step must be > 0 and --fmax >= --fmin")
    sweep = check_f(list(range(args.fmin, args.fmax + 1, step)))
    repeats = [sweep[0]]
    name = "config_maxp_coarse.json"
    what = f"coarse scan {args.fmin}-{args.fmax} Hz in {step} Hz steps, {sweep[0]} Hz repeated at the end"
else:
    if not args.center:
        raise SystemExit("--fine needs --center <Hz>[,<Hz>...]")
    step = args.step or 20
    centres = [int(float(x)) for x in args.center.replace(";", ",").split(",") if x.strip()]
    if step <= 0 or args.span < 0:
        raise SystemExit("--step must be > 0 and --span >= 0")
    half = args.span // 2
    sweep = []
    for c in centres:
        n = half // step
        sweep += [c + k * step for k in range(-n, n + 1)]
    check_f(sweep)
    repeats = centres
    name = "config_maxp_fine_" + "_".join(map(str, centres)) + "Hz.json"
    what = (f"fine scan +-{half} Hz around {', '.join(map(str, centres))} Hz in {step} Hz steps, "
            f"centre{'s' if len(centres) > 1 else ''} repeated at the end")
FREQS = sweep + repeats

DRIVE_S, STOP_MBAR, OFF_S, REST_S, MIN_PULL_ROOM = 12.0, 270.0, 3.0, 3.0, 0.3
PRESSURE = {
    "enabled": True, "backend": "arduino", "port": "COM6", "baud": 115200, "firmware": "auto",
    "sensor": {"backend": "fluigent", "channel": 1},
    "limit_mbar": 280.0, "min_mbar": -60.0, "rate_mbar_s": 40.0,
    "tolerance_mbar": 2.0, "settle_s": 3.0, "microsteps": 16,
    "pos_min": 0, "pos_max": 2470, "pos_margin": 30, "clear_reserve_steps": 150,
    "ul_per_step": round(1000.0 / 2470, 4),
}


def zero(s):
    return {"type": "constant", "frequency_hz": 0, "duration_s": float(s), "repeat": 1, "settle_ms": 0}


def release():
    return {"type": "pressure", "target_mbar": "release", "timeout_s": 30.0,
            "release_rate_mbar_s": args.release_rate, "release_floor_mbar": args.release_floor}


tests = [{"type": "voltage", "voltage_mv": args.v, "while_waiting": "zero", "settle_s": 5,
          "note": f"{args.v / 1000:.2f} V by the supply"}]
for k, f in enumerate(FREQS, 1):
    repeat = k > len(sweep)
    tests += [release(),
              {"type": "pressure", "target_mbar": "hold_position", "min_pull_room": MIN_PULL_ROOM},
              {"type": "constant", "frequency_hz": int(f), "duration_s": DRIVE_S, "repeat": 1, "settle_ms": 0,
               "section": "maxp_scan", "level_mv": args.v, "step": k, "check": repeat, "stop_above_mbar": STOP_MBAR},
              zero(OFF_S), release(), zero(REST_S)]

supply = {"mode": "rsd3305p", "channel": 1, "current_limit_ma": 60, "max_mv": 2500, "initial_mv": args.v,
          "tolerance_mv": 50, "step_timeout_s": 10, "off_on_exit": True, "eol": "LF"}
if args.address:
    supply["address"] = args.address
desc = (f"Maximum-pressure frequency scan (blocked outlet): {what}, at {args.v / 1000:.2f} V. Per frequency: gentle "
        f"release (pull only, {args.release_rate:g} mbar/s, stop at {args.release_floor:g} mbar), the syringe holds the "
        f"plunger still with the motor energised, {DRIVE_S:g} s drive (ended early at {STOP_MBAR:g} mbar = maximum >= "
        f"{STOP_MBAR:g}), {OFF_S:g} s at 0 Hz (leak decay), release, {REST_S:g} s at 0 Hz. Voltage by the RSD3305P, PPK2 "
        f"on. ORDER: 1) prime with the outlet vented, at the pump's own peak if known (config_scan_quick.json, "
        f"make_config_prime_at.py); 2) connect the syringe through the stopcock, syringe EMPTY, stopcock closed to "
        f"waste; 3) this config. A syringe clear is asked for when less than 30 % is left to pull back. Type the Pump "
        f"ID. Open the stopcock to waste before ANY manual syringe push.")
cfg = {
    "port": "COM3", "baud": 115200, "read_timeout_s": 0.2,
    "start_time": None, "end_time": None,
    "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
    "require_pump_id": True,
    "tests": tests,
    "meta": {"name": "Max-pressure scan " + ("coarse" if args.coarse else "fine"), "description": desc,
             "operator": "", "pump_id": args.pump, "syringe_ml": 1.0, "syringe_on": "outlet", "pressure_channel": 1,
             "freqs_hz": FREQS, "repeats_hz": repeats},
    "serial": {"protocol": "freq_only"},
    "protocol": "freq_only",
    "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
    "pressure": PRESSURE,
    "supply": supply,
    "power": {"enabled": True, "backend": "ppk2", "port": "auto", "bin_ms": 10.0, "watch_ma": 57,
              "watch_hold_ms": 300, "raw_every_min": 10, "raw_length_s": 2.0, "post_run_tail_s": 10.0},
}
with open(name, "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, indent=2)

# run length: release from a ~150 mbar plateau at the ramp, ~0.3 s for the no-op release before each hold
rel_s = 150.0 / args.release_rate
per = DRIVE_S + OFF_S + rel_s + REST_S + 0.3
t = len(FREQS) * per + 6.0 + 10.0
print(f"{name}: {len(sweep)} frequencies + {len(repeats)} repeat(s) = {len(FREQS)} steps, ~{per:.0f} s each, "
      f"~{t / 60:.1f} min (+ ~1.5 min per syringe clear if asked); {FREQS[0]}..{max(FREQS)} Hz")
