"""Blocked-outlet (dead-head) maximum pressure for any pump, at its own frequencies (NM, 2 Nov 2026).

The same steps as config_pq_261030-1_deadhead.json (make_config_pq.py): the syringe holds the plunger still with
the motor energised ("hold_position", a clear first if < 30 % is left to pull back), the pump pressurises the fixed
volume for 60 s (ended early at 270 mbar, below the 280 mbar limit, logged as stop_above), 5 s at 0 Hz, gentle
release to 3 mbar (15 mbar/s, never below), 15 s at 0 Hz. Test each pump at its own peak (config_scan_quick.json)
plus 600 Hz, e.g. --f-list 450,600 for 261030-2.

Order: strong priming / prime_at with the outlet vented -> connect the syringe through the stopcock, syringe EMPTY,
stopcock closed to waste -> this config. Open the stopcock to waste before ANY manual syringe push.

usage: python make_config_deadhead.py [--f-list 600,800,1000,1300,1600,2000] [--repeat-f <Hz>] [--low-f-list 600]
                                      [--low-v 1000] [--pump <id>] [--address COM7 | host:5025 | ""]
  --repeat-f: repeated at 2.5 V after the list as a drift check (default: the first of --f-list; 0 = none)
  --low-f-list: frequencies at --low-v after that ("" = none)
  -> config_deadhead.json, or config_deadhead_<pump>.json with --pump
"""
import argparse
import json

ap = argparse.ArgumentParser(description="Dead-head maximum pressure, any pump")
ap.add_argument("--f-list", default="600,800,1000,1300,1600,2000", help="frequencies at 2.5 V (Hz, comma separated)")
ap.add_argument("--repeat-f", type=int, default=None, help="drift check at 2.5 V after the list (default: first; 0 = none)")
ap.add_argument("--low-f-list", default="600", help="frequencies at --low-v ('' = none)")
ap.add_argument("--low-v", type=int, default=1000, help="low voltage in mV (0 < v <= 2500)")
ap.add_argument("--pump", default="", help="pump ID written into the config (still asked for at the start)")
ap.add_argument("--address", default="COM7", help="RSD3305P address (default COM7, the lab PC); '' = Power panel")
args = ap.parse_args()


def freqs(text):
    out = [int(float(x)) for x in str(text).replace(";", ",").split(",") if x.strip()]
    for f in out:
        if not 1 <= f <= 3000:
            raise SystemExit(f"{f} Hz is outside 1..3000 Hz")
    return out


F_HIGH = freqs(args.f_list)
if not F_HIGH:
    raise SystemExit("--f-list is empty")
REPEAT = F_HIGH[0] if args.repeat_f is None else args.repeat_f
F_LOW = freqs(args.low_f_list)
if not 0 < args.low_v <= 2500:
    raise SystemExit("--low-v must be between 1 and 2500 mV")
LEVELS = [(2500, F_HIGH + ([REPEAT] if REPEAT else []))] + ([(args.low_v, F_LOW)] if F_LOW else [])

DRIVE_S, STOP_MBAR, OFF_S, REST_S = 60.0, 270.0, 5.0, 15.0
RELEASE = {"type": "pressure", "target_mbar": "release", "timeout_s": 45.0,
           "release_rate_mbar_s": 15.0, "release_floor_mbar": 3.0}
STEP_SETTLE_S, MIN_PULL_ROOM, RELEASE_GUESS_S = 5, 0.3, 15.0
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


def voltage(mv):
    return {"type": "voltage", "voltage_mv": mv, "while_waiting": "zero", "settle_s": STEP_SETTLE_S,
            "note": f"{mv / 1000:.1f} V by the supply"}


tests = [voltage(LEVELS[0][0]), dict(RELEASE)]
k = 0
for i, (mv, fs) in enumerate(LEVELS):
    if i:
        tests.append(voltage(mv))
    for j, f in enumerate(fs):
        k += 1
        tests += [{"type": "pressure", "target_mbar": "hold_position", "min_pull_room": MIN_PULL_ROOM},
                  {"type": "constant", "frequency_hz": int(f), "duration_s": DRIVE_S, "repeat": 1, "settle_ms": 0,
                   "section": "deadhead", "level_mv": mv, "step": k, "check": j > 0 and f in fs[:j],
                   "stop_above_mbar": STOP_MBAR},
                  zero(OFF_S), dict(RELEASE), zero(REST_S)]

supply = {"mode": "rsd3305p", "channel": 1, "current_limit_ma": 60, "max_mv": 2500, "initial_mv": 2500,
          "tolerance_mv": 50, "step_timeout_s": 10, "off_on_exit": True, "eol": "LF"}
if args.address:
    supply["address"] = args.address
steps = "; ".join(f"{mv / 1000:g} V: {', '.join(map(str, fs))} Hz" for mv, fs in LEVELS)
desc = (f"Blocked-outlet (dead-head) maximum pressure vs frequency{' for ' + args.pump if args.pump else ''}. The syringe "
        f"holds the plunger still (motor energised, no regulation) and the pump pressurises the fixed volume: "
        f"{DRIVE_S:g} s per step (ended early at {STOP_MBAR:g} mbar, logged as stop_above = maximum >= {STOP_MBAR:g}), "
        f"{OFF_S:g} s at 0 Hz, gentle release to ~3 mbar (15 mbar/s, never below), {REST_S:g} s at 0 Hz. {steps}. "
        f"Voltages by the RSD3305P, PPK2 on. ORDER: 1) prime with the outlet vented (strong priming or prime_at); "
        f"2) connect the syringe through the stopcock, syringe EMPTY, stopcock closed to waste; 3) this config. Type "
        f"the Pump ID. Open the stopcock to waste before ANY manual syringe push (hand pushes against the closed line "
        f"read 530-550 mbar on 2 Nov).")
cfg = {
    "port": "COM3", "baud": 115200, "read_timeout_s": 0.2,
    "start_time": None, "end_time": None,
    "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
    "require_pump_id": True,
    "tests": tests,
    "meta": {"name": "Dead-head maximum pressure" + (f" {args.pump}" if args.pump else ""), "description": desc,
             "operator": "", "pump_id": args.pump, "syringe_ml": 1.0, "syringe_on": "outlet", "pressure_channel": 1},
    "serial": {"protocol": "freq_only"},
    "protocol": "freq_only",
    "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
    "pressure": PRESSURE,
    "supply": supply,
    "power": {"enabled": True, "backend": "ppk2", "port": "auto", "bin_ms": 10.0, "watch_ma": 57,
              "watch_hold_ms": 300, "raw_every_min": 10, "raw_length_s": 2.0, "post_run_tail_s": 10.0},
}
name = f"config_deadhead_{args.pump}.json" if args.pump else "config_deadhead.json"
with open(name, "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, indent=2)
n = sum(len(fs) for _, fs in LEVELS)
t = n * (DRIVE_S + OFF_S + RELEASE_GUESS_S + REST_S) + len(LEVELS) * (1.0 + STEP_SETTLE_S) + RELEASE_GUESS_S + 10.0
print(f"{name}: {n} steps ({steps}), ~{t / 60:.1f} min (shorter when a step stops at {STOP_MBAR:g} mbar)")
