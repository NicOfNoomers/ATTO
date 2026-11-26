"""G test: chamber compliance at 0 to -300 mbar with the syringe (renata_review brief, 26 Nov 2026).

Question: is the chamber compliance that moves the model's flow and pressure removable gas or structure?
Gas compliance grows as 1/p_abs^2 (2x at -300 mbar gauge), structure doesn't. Syringe volume steps at a
falling static pressure give the compliance from 0 to -300 mbar; scripts/analyze_pv_steps.py fits
C(p) = C_struct + V_g0 p_atm / p_abs^2 after subtracting the line.

Why suction from the OUTLET: suction on the outlet opens the outlet check valve (forward flow), so with the
INLET CLAMPED the chamber follows the syringe down. Pushing closes the outlet valve and holds the chamber, so:
  pull branch  = outlet line + chamber (+ the inlet stub)   -> one-way staircase 0 -> -300 mbar
  push branch  = outlet line only (chamber held at its low pressure) -> an in-run check of the line
  reset        = open the inlet clamp a few seconds (the chamber refills from the reservoir), clamp again.
Small up-and-back steps at each level are not possible through the valves; the staircase covers the levels.

Rig (needs Nic, hand changes):
  reservoir -> inlet stub + CLAMP at the inlet port -> pump -> flow sensor -> pressure sensor -> 3-way stopcock
    -> (a) reservoir: priming and flow checks   (b) the 1 mL syringe on the motor: the G test
  The syringe must have room to PULL: the run pulls back at most --room-steps full steps (0.023 mm and
  0.405 uL per full step) from where the plunger is when the first step block starts. Syringe mode restores
  its position from syringe_state.json, so the pinch rig's absolute back limit (count 27133) does not carry
  over: set the plunger with that much travel behind it.
  Line from the outlet to the syringe: short and stiff (no silicone flex tube); it is 70-90 % of the measured
  compliance (29 Oct: 10-25 mbar per full step = 0.016-0.04 uL/mbar, against ~0.004 expected for the chamber).

Sequence: [voltage] -> prompt A (stopcock to reservoir, inlet open) -> zero the sensor -> [priming rounds +
flow checks, NM 23 Nov rule] -> prompt B (stopcock to syringe, clamp the inlet) -> reps x [pull staircase to
--until, push staircase back to ~0, prompt: unclamp/reclamp] -> prompt A again -> [post checks].
--baseline: the same staircases with the pump replaced by a stiff union (line only; no priming/checks).

Each move: hold --hold-s (motor energised, no regulation, pump 0 Hz), pressure averaged over the last
--measure-s -> pv_steps.csv (run folder): counts, uL, pressure before/after, drift, flow-sensor volume.
Moves are sized for ~--dp mbar from the last measured gain; the slack after each direction change (~350
counts at x16) is taken up first. Ends at --until, at --room-steps, or before crossing --min-mbar + 10.

Safety: pressure below --min-mbar (default -330; the code default is -30 because suction degasses the line:
here that is the point of the test, Nic's OK needed) or above --limit -> fault: motor stops, pump 0 Hz,
run stops. Sensor Pressure_S: +-344.74 mbar (the controller also stops at 95 % of it, -327.5).
If a run stops early the chamber may be left at suction: open the inlet clamp, stopcock to the reservoir.

usage:
  python make_config_g_test.py [--pump <id>] [--baseline] [--reps 3] [--until -300] [--dp 20] [--hold-s 8]
      [--measure-s 4] [--room-steps 150] [--prime-f 600 --prime-rounds 4 --checks 300,600 --check-s 20]
      [--min-mbar -330] [--limit 100] [--supply manual|rsd3305p] [--v 2500] [--dummy] [--out <file>]
"""
import argparse
import json

ap = argparse.ArgumentParser(description="G test (pressure-volume steps) with the syringe")
ap.add_argument("--pump", default="", help="pump ID (empty: run_headless --pump-id must give it; confirm with Nic)")
ap.add_argument("--baseline", action="store_true", help="line only: pump replaced by a stiff union")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--until", type=float, default=-300.0, help="pull staircase ends once the pressure is below this")
ap.add_argument("--dp", type=float, default=20.0, help="mbar per move (sizes the moves from the measured gain)")
ap.add_argument("--hold-s", type=float, default=8.0, help="hold after each move (isothermal, creep visible)")
ap.add_argument("--measure-s", type=float, default=4.0, help="last part of the hold that is averaged")
ap.add_argument("--first-counts", type=int, default=4, help="first move of a staircase (motor counts; 4 = 0.1 uL)")
ap.add_argument("--max-counts", type=int, default=160, help="largest move (counts; 160 = 4 uL)")
ap.add_argument("--fine-mbar", type=float, default=40.0,
                help="pull staircases start with fixed small moves over this many mbar: they resolve where the "
                     "outlet valve opens (its cracking pressure, ~10-20 mbar expected); 0 = off")
ap.add_argument("--fine-counts", type=int, default=6, help="size of those moves (counts; 6 = 0.15 uL, ~5 mbar)")
ap.add_argument("--room-steps", type=float, default=150.0, help="most full steps the plunger may move from its start")
ap.add_argument("--min-mbar", type=float, default=-330.0)
ap.add_argument("--limit", type=float, default=250.0,
                help="upper limit (mbar): clamping the inlet squeezes liquid through the pump into the closed line "
                     "(+38 mbar per clamping on 26 Nov, the line sat at 125 mbar and faulted a 100 mbar limit)")
ap.add_argument("--prime-f", type=int, default=0, help="priming frequency before the test (0 = none)")
ap.add_argument("--prime-rounds", type=int, default=4)
ap.add_argument("--checks", default="", help="flow-check frequencies after each priming round and after the test")
ap.add_argument("--check-s", type=float, default=20.0)
ap.add_argument("--supply", choices=("manual", "rsd3305p"), default="manual")
ap.add_argument("--address", default="COM7")
ap.add_argument("--v", type=int, default=2500)
ap.add_argument("--firmware", choices=("v2", "v1", "auto"), default="v2",
                help="COM6 runs syringe_v2; auto-detect misreads it as v1 after the reset (23 Nov)")
ap.add_argument("--no-stopcock", action="store_true",
                help="the syringe sits directly on the outlet line (no 3-way stopcock): no priming, checks or "
                     "sensor zero (the cracking-pressure step makes the result independent of the zero)")
ap.add_argument("--dummy", action="store_true", help="simulated line, syringe and sensors (no rig)")
ap.add_argument("--out", default=None)
a = ap.parse_args()

if not 0 < a.v <= 2500:
    raise SystemExit("--v must be between 1 and 2500 mV")
if not -340.0 <= a.min_mbar <= -30.0 or a.until <= a.min_mbar + 15.0:
    raise SystemExit("need -340 <= --min-mbar <= -30 and --until at least 15 mbar above --min-mbar")
if not 20.0 <= a.limit <= 300.0:
    raise SystemExit("--limit: 20..300 mbar")
if a.measure_s >= a.hold_s or a.hold_s < 2.0:
    raise SystemExit("need --hold-s >= 2 and --measure-s < --hold-s")
checks = [int(x) for x in a.checks.split(",") if x.strip()]
if a.baseline and (a.prime_f or checks):
    raise SystemExit("--baseline has no pump: no priming or checks")
if a.no_stopcock and (a.prime_f or checks):
    raise SystemExit("--no-stopcock: the pump would pump into the 1 mL syringe (a 20 s check is ~80 uL, more than "
                     "the plunger room): no priming or checks in this run")
if a.prime_f and not (1 <= round(0.8 * a.prime_f) and round(1.2 * a.prime_f) <= 3000):
    raise SystemExit("--prime-f: 0.8 F .. 1.2 F must lie within 1..3000 Hz")
if any(not 1 <= f <= 3000 for f in checks) or (checks and a.check_s < 10):
    raise SystemExit("--checks: 1..3000 Hz, --check-s >= 10")


def zero(sec):
    return {"type": "constant", "frequency_hz": 0, "duration_s": float(sec), "repeat": 1, "settle_ms": 0}


def drive(hz, sec, **meta):
    return {"type": "constant", "frequency_hz": int(hz), "duration_s": float(sec), "repeat": 1, "settle_ms": 0,
            "level_mv": a.v, **meta}


def check_blocks(section, rnd):
    out = []
    for f in checks:
        out += [drive(f, a.check_s, section=section, rep=rnd, check_hz=f), zero(10.0)]
    return out


def prompt(title, text, plan_s=60):
    return {"type": "pressure", "target_mbar": "prompt", "title": title, "text": text, "plan_s": plan_s,
            "room_steps": a.room_steps}


def pv(direction, until, rep, section, plan_moves):
    fine = a.fine_mbar if direction == "pull" else 0.0
    return {"type": "pressure", "target_mbar": "pv_steps", "direction": direction, "until_mbar": until,
            "target_dp_mbar": a.dp, "first_counts": a.first_counts, "min_counts": 2, "max_counts": a.max_counts,
            "max_moves": 40, "hold_s": a.hold_s, "measure_s": a.measure_s, "takeup_mbar": 2.0, "takeup_counts": 4,
            "takeup_wait_s": 0.3, "takeup_max_counts": 800, "room_steps": a.room_steps,
            "max_c_ul_mbar": 0.15, "check_counts": 30,
            "fine_mbar": fine, "fine_counts": a.fine_counts, "section": section, "rep": rep,
            "plan_moves": plan_moves + (int(fine / 5.0) if fine else 0)}


nmoves = int(round(abs(a.until) / a.dp)) + 2
room_mm = a.room_steps * 0.023
if a.baseline:
    setup_a = (f"BASELINE (no pump): the pump is replaced by a stiff union between the inlet stub and the outlet "
               f"line (same tubing, sensors and stopcock as with the pump). Stopcock to the RESERVOIR, inlet "
               f"stub OPEN, everything liquid-filled without bubbles. Syringe on the motor with at least "
               f"{a.room_steps:g} full steps (~{room_mm:.1f} mm) of travel BEHIND the plunger.")
    setup_b = ("Stopcock to the SYRINGE (closed to the reservoir). Clamp the inlet stub at the same point as "
               "with the pump.")
else:
    setup_a = (f"Stopcock to the RESERVOIR (pump -> flow -> pressure -> stopcock -> reservoir), inlet line OPEN, "
               f"pump at 0 Hz. Syringe on the motor with at least {a.room_steps:g} full steps (~{room_mm:.1f} mm) "
               f"of travel BEHIND the plunger; line from the outlet to the syringe short and stiff, no bubbles.")
    setup_b = ("Stopcock to the SYRINGE (closed to the reservoir). CLAMP the inlet right at the pump's inlet "
               "port. Pump stays at 0 Hz.")
reset = ("Open the inlet clamp for ~5 s (the chamber refills from the reservoir, back to ~0 mbar), then clamp it "
         "again at the same point.")
if a.no_stopcock:
    room_txt = (f"The plunger needs at least {a.room_steps:g} full steps (~{room_mm:.1f} mm) of travel BEHIND it "
                f"(this test pulls; the run never goes further than that from where it starts). If it has less, "
                f"press Stop now.")
    if a.baseline:
        setup_a = ("BASELINE (no pump): the pump is replaced by a stiff union between the inlet stub and the outlet "
                   "line; the syringe sits directly on the outlet line; everything liquid-filled, no bubbles. "
                   + room_txt)
        setup_b = "CLAMP the inlet stub at the same point as with the pump."
    else:
        setup_a = ("Syringe directly on the outlet line (pump -> flow -> pressure -> syringe), liquid-filled, no "
                   "bubbles; pump at 0 Hz; inlet line OPEN to the reservoir. " + room_txt)
        setup_b = ("Open the inlet clamp ~5 s if it was closed (chamber at the reservoir's pressure), then CLAMP the "
                   "inlet right at the pump's inlet port. Pump stays at 0 Hz.")

tests = []
if a.supply == "rsd3305p" and not a.baseline:
    tests.append({"type": "voltage", "voltage_mv": a.v, "while_waiting": "zero", "settle_s": 5,
                  "note": f"{a.v / 1000:.2f} V by the supply"})
tests.append(prompt("Setup A: open loop" if not a.no_stopcock else "Setup: syringe on the outlet", setup_a, 120))
if not a.no_stopcock:
    tests.append({"type": "pressure", "target_mbar": "zero", "wait_s": 3, "seconds": 3})
if a.prime_f:
    F = a.prime_f
    for r in range(1, a.prime_rounds + 1):
        for f, sec in ((F, 30.0), (0, 5.0), (round(0.8 * F), 10.0), (0, 5.0), (round(1.2 * F), 10.0), (0, 10.0)):
            tests.append(zero(sec) if f == 0 else drive(f, sec, section="prime_at", rep=r))
        tests += check_blocks("prime_check", r)
elif checks:
    tests += check_blocks("prime_check", 0)
tests.append(prompt("Setup B: closed for the G test", setup_b, 60))
for rep in range(1, a.reps + 1):
    tests.append(pv("pull", a.until, rep, "g_pull", nmoves))
    tests.append(pv("push", -10.0, rep, "g_push", nmoves))
    if rep < a.reps and not a.baseline:
        tests.append(prompt(f"Reset after rep {rep}", reset, 30))
if a.no_stopcock:
    end_text = "Open the inlet clamp (the chamber back to the reservoir's pressure). The run then ends."
else:
    end_text = ("Open the inlet clamp, stopcock back to the RESERVOIR (Setup A)." if not a.baseline
                else "Stopcock back to the RESERVOIR, unclamp the inlet stub.")
tests.append(prompt("Back to the open loop" if not a.no_stopcock else "Open the inlet", end_text, 60))
tests += check_blocks("post_check", 0)

if a.supply == "manual":
    supply = {"mode": "manual", "current_limit_ma": 60, "max_mv": 2500, "initial_mv": a.v, "lead_min": 2}
    stxt = f"MANUAL bench supply set by hand to {a.v / 1000:.2f} V"
else:
    supply = {"mode": "rsd3305p", "channel": 1, "current_limit_ma": 60, "max_mv": 2500, "initial_mv": a.v,
              "tolerance_mv": 50, "step_timeout_s": 10, "off_on_exit": True, "eol": "LF", "address": a.address}
    stxt = f"{a.v / 1000:.2f} V by the RSD3305P ({a.address})"
pressure = {"enabled": True, "backend": "dummy" if a.dummy else "arduino", "port": "COM6", "baud": 115200,
            "firmware": a.firmware, "sensor": {"backend": "fluigent", "channel": 1},
            "limit_mbar": a.limit, "min_mbar": a.min_mbar, "microsteps": 16, "pos_min": 0, "pos_max": 2470,
            "pos_margin": 30, "clear_reserve_steps": 150, "ul_per_step": 0.4049, "release_on_stop": True}
if a.dummy:   # dry-run line: 0.02 uL/mbar line + 0.0015 chamber, 2.5 uL gas, no leak (no valves in this sim)
    pressure["sim"] = {"compliance_ul_mbar": 0.0215, "gas_ul": 2.5, "r_leak": 1e12, "p_lose": 0.0,
                       "backlash_full_steps": 22, "noise_mbar": 0.3, "sensor_range": [-344.74, 344.74]}
what = "BASELINE, line only (pump replaced by a union)" if a.baseline else "pump undriven"
desc = (f"G test ({what}): {a.reps} x [syringe pull staircase 0 -> {a.until:g} mbar (~{a.dp:g} mbar per move), "
        f"push staircase back to ~0]; hold {a.hold_s:g} s per move, mean of the last {a.measure_s:g} s; "
        f"pv_steps.csv. Inlet clamped, suction from the outlet; prompts for the hand changes (--on-prompt wait, "
        f"someone at the rig). Limits {a.min_mbar:g} / +{a.limit:g} mbar; plunger room {a.room_steps:g} full steps. "
        f"{stxt}; PPK2 on."
        + (f" Before: {a.prime_rounds} priming rounds at {a.prime_f} Hz with flow checks at {checks} Hz." if a.prime_f
           else "")
        + (f" After: flow checks at {checks} Hz (post_check)." if checks else "")
        + " Analysis: scripts/analyze_pv_steps.py.")
cfg = {
    "port": "COM3", "baud": 115200, "read_timeout_s": 0.2, "start_time": None, "end_time": None,
    "break_seconds_between_tests": 0.0, "send_zero_on_exit": True, "require_pump_id": True,
    "meta": {"name": "G test baseline" if a.baseline else "G test", "description": desc, "operator": "",
             "pump_id": a.pump or ("line_baseline" if a.baseline else "")},
    "serial": {"protocol": "freq_only"}, "protocol": "freq_only",
    "flow_sensor": {"backend": "dummy" if a.dummy else "fluigent", "channel": 0, "sample_rate_hz": 25.0},
    "pressure": pressure, "supply": supply,
    "power": {"enabled": not a.dummy, "backend": "ppk2", "port": "auto", "bin_ms": 10.0, "watch_ma": 57,
              "watch_hold_ms": 300, "raw_every_min": 10, "raw_length_s": 2.0, "post_run_tail_s": 10.0},
    "tests": tests,
}
per_stair = nmoves * (a.hold_s + 0.3) + a.hold_s + 20
est = a.reps * 2 * per_stair + a.reps * int(a.fine_mbar / 5.0) * (a.hold_s + 0.3) + (a.reps - 1) * 30 + 3 * 60 + 6
est += sum(t["duration_s"] for t in tests if t.get("type") == "constant")
out = a.out or ("config_g_test_baseline" if a.baseline else "config_g_test") + ("_dummy" if a.dummy else "") + ".json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
print(f"wrote {out}: {a.reps} reps, ~{nmoves} moves per staircase; about {est / 60:.0f} min including ~4 min of "
      f"hand changes")
