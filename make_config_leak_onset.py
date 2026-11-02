"""Valve back-leak and onset voltage, any pump (NM, 2 Nov 2026).

Why: do the weaker pumps (261102-1, 261030-2) differ from 261030-1 in valve leakage or in the voltage where they
start pumping? In the 261030-1 dead-head run the 5 s pump-off before each release decayed 1.1-2 mbar/s at
100-160 mbar; on 1-2 Nov 261030-1 drew no current below ~0.55 V.

Syringe connected through the stopcock (closed to waste), syringe EMPTY at the start, RSD3305P (COM7), PPK2 on,
Pump ID required (typed in; no default).
- Fill: the syringe holds 5 mbar while the pump runs f at 2.5 V for FILL_S, so the pump's own flow puts liquid in
  the syringe for the pushes in part A (no suction, nothing done by hand).
- Part A, leak holds, pump OFF the whole time: the syringe pushes to 50 mbar (settle), then holds its position
  with the motor energised (no regulation) for 30 s while the pressure decays back through the pump's valves
  (pressure/pressure.csv, pos_steps constant), then a gentle release (15 mbar/s, stop at 3 mbar). Again at 100.
- Part B, onset voltage: f (default 600 Hz) at 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 V, then 2.5 V as reference, 20 s each
  with 15 s zeros, near free flow: the syringe holds ~5 mbar and absorbs the small flows (<= ~100 uL/min, < 0.3 mL
  in all). Blocks below 0.8 V carry "ppk2_below_range" (cross-check with supply/supply.csv). 0.5 V may draw no
  current at all; that is expected.

usage: python make_config_leak_onset.py [--f 600] [--pump <id>] [--fill-s 90] [--address COM7 | host:5025 | ""]
  -> config_leak_onset.json (config_leak_onset_<f>Hz.json when --f is not 600)
"""
import argparse
import json

ap = argparse.ArgumentParser(description="Valve back-leak holds + onset voltage (any pump)")
ap.add_argument("--f", type=int, default=600, help="drive frequency for the fill and part B (Hz)")
ap.add_argument("--pump", default="", help="pump ID written into the config (still asked for at the start)")
ap.add_argument("--fill-s", type=float, default=90.0, help="seconds of pump flow into the syringe before part A")
ap.add_argument("--address", default="COM7", help="RSD3305P address (default COM7, the lab PC); '' = Power panel")
args = ap.parse_args()
if not 1 <= args.f <= 3000:
    raise SystemExit("--f must be within 1..3000 Hz")

LEAK_TARGETS = [50.0, 100.0]
LEAK_HOLD_S, SETTLE_TIMEOUT_S, AFTER_S = 30.0, 60.0, 10.0
ONSET_MV = [500, 600, 700, 800, 900, 1000, 2500]
ONSET_DRIVE_S, ONSET_ZERO_S, STEP_SETTLE_S = 20.0, 15.0, 5
NEAR_FREE_MBAR = 5.0
PPK2_MIN_MV = 800
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


def drive(hz, s, **meta):
    return {"type": "constant", "frequency_hz": int(hz), "duration_s": float(s), "repeat": 1, "settle_ms": 0, **meta}


def voltage(mv):
    b = {"type": "voltage", "voltage_mv": mv, "while_waiting": "zero", "settle_s": STEP_SETTLE_S,
         "note": f"{mv / 1000:.2f} V by the supply"}
    if mv < PPK2_MIN_MV:
        b["ppk2_below_range"] = True
    return b


def gentle_release():
    return {"type": "pressure", "target_mbar": "release", "timeout_s": 45.0,
            "release_rate_mbar_s": 15.0, "release_floor_mbar": 3.0}


tests = [zero(10.0), voltage(2500),
         # fill: hold ~5 mbar while the pump runs, the syringe takes up the flow
         {"type": "pressure", "target_mbar": NEAR_FREE_MBAR, "wait_settled": False},
         drive(args.f, args.fill_s, section="fill", level_mv=2500), zero(5.0)]
for p in LEAK_TARGETS:                                  # part A: pump off throughout
    tests += [{"type": "pressure", "target_mbar": p, "wait_settled": True, "timeout_s": SETTLE_TIMEOUT_S},
              {"type": "pressure", "target_mbar": "hold_position", "section": "leak_hold", "p_start_mbar": p},
              zero(LEAK_HOLD_S), gentle_release(), zero(AFTER_S)]
tests.append({"type": "pressure", "target_mbar": NEAR_FREE_MBAR, "wait_settled": False})   # part B
for mv in ONSET_MV:
    meta = {"section": "onset", "level_mv": mv, "reference": mv == 2500}
    if mv < PPK2_MIN_MV:
        meta["ppk2_below_range"] = True
    tests += [voltage(mv), drive(args.f, ONSET_DRIVE_S, **meta), zero(ONSET_ZERO_S)]
tests += [gentle_release(), {"type": "pressure", "target_mbar": "idle"}]

supply = {"mode": "rsd3305p", "channel": 1, "current_limit_ma": 60, "max_mv": 2500, "initial_mv": 2500,
          "tolerance_mv": 50, "step_timeout_s": 10, "off_on_exit": True, "eol": "LF"}
if args.address:
    supply["address"] = args.address
desc = (f"Valve back-leak and onset voltage. Syringe through the stopcock (closed to waste), syringe EMPTY at the start, "
        f"Control pressure ticked. Fill: the syringe holds {NEAR_FREE_MBAR:g} mbar while {args.f} Hz at 2.5 V runs "
        f"{args.fill_s:g} s. Part A (pump OFF): push to {', '.join(f'{p:g}' for p in LEAK_TARGETS)} mbar, hold the "
        f"position with the motor energised for {LEAK_HOLD_S:g} s (back-leak decay through the valves), gentle release. "
        f"Part B: {args.f} Hz at {', '.join(f'{v / 1000:g}' for v in ONSET_MV)} V, {ONSET_DRIVE_S:g} s each with "
        f"{ONSET_ZERO_S:g} s zeros, syringe holding ~{NEAR_FREE_MBAR:g} mbar (near free flow); 0.5 V may draw no "
        f"current (expected); below 0.8 V cross-check the PPK2 with supply/supply.csv. Type the Pump ID. Open the "
        f"stopcock to waste before ANY manual syringe push.")
cfg = {
    "port": "COM3", "baud": 115200, "read_timeout_s": 0.2,
    "start_time": None, "end_time": None,
    "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
    "require_pump_id": True,
    "tests": tests,
    "meta": {"name": "Valve leak and onset voltage", "description": desc, "operator": "", "pump_id": args.pump,
             "syringe_ml": 1.0, "syringe_on": "outlet", "pressure_channel": 1, "drive_f_hz": args.f,
             "leak_targets_mbar": LEAK_TARGETS, "onset_mv": ONSET_MV},
    "serial": {"protocol": "freq_only"},
    "protocol": "freq_only",
    "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
    "pressure": PRESSURE,
    "supply": supply,
    "power": {"enabled": True, "backend": "ppk2", "port": "auto", "bin_ms": 10.0, "watch_ma": 57,
              "watch_hold_ms": 300, "raw_every_min": 10, "raw_length_s": 2.0, "post_run_tail_s": 10.0},
}
name = "config_leak_onset.json" if args.f == 600 else f"config_leak_onset_{args.f}Hz.json"
with open(name, "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, indent=2)
nom = sum(t.get("duration_s", 0) for t in tests if t["type"] == "constant") + \
    sum(t.get("settle_s", 0) for t in tests if t["type"] == "voltage")
est = nom + len(LEAK_TARGETS) * (15.0 + 8.0) + 8.0 + len(ONSET_MV) * 1.0 + 10.0   # settles, releases, steps, tail
print(f"{name}: fill {args.fill_s:g} s at {args.f} Hz, leak holds at {LEAK_TARGETS} mbar, onset {args.f} Hz at "
      f"{[v / 1000 for v in ONSET_MV]} V; {nom / 60:.1f} min of blocks, ~{est / 60:.1f} min with settling and releases")
