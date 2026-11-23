"""PQ through the pinch-valve loop (NM, 23 Nov 2026): the syringe motor squeezes a flex tube.

Loop: reservoir (outlet low) -> pump -> flow sensor -> pressure sensor -> flex tube (ID 1/16 in) -> reservoir
(inlet high). The same stepper, driver and syringe_v2 firmware on COM6; the plunger presses on the tube.
No mechanical limits: the pressure rise shows contact ("touch"). See pinch_control.py.

Per frequency (block 'pinch_pq'):
  run up (spin_up_s) -> find the touch (once per run: squeeze until the pressure rises touch_mbar)
  -> open point -> position points (full steps past touch) -> closing sweep to max_stop_mbar or a plateau
  (= the blocked-outlet pressure, and the pressure-squeeze map) -> open -> pressure targets (moved to the mapped
  squeeze, held within 2 mbar, then measured with the motor still) -> open.
Each point: settle_s, then flow and pressure averaged over measure_s -> pinch_pq.csv (run folder);
pinch/pressure.csv and pinch/events.json log every sample and event.

Priming first (NM 23 Nov: "make sure the pump is fully primed"): --prime-f F --prime-rounds N --checks 520,780,1090
  N x [F 30 s, 0 Hz 5 s, 0.8 F 10 s, 0 Hz 5 s, 1.2 F 10 s, 0 Hz 10 s (as make_config_prime_at.py), then each check
  frequency 20 s + 0 Hz 10 s] before the PQ, plunger open (at its start, in front of the tube); the same checks
  again after the PQ ('post_check'). Primed = the last two rounds' checks agree within ~3 %; the post checks show
  whether the closures changed the pump. Sections prime_at / prime_check (rep = round) / post_check.

Safety: pressure above min(--limit, 95 % of the sensor range; Pressure_S: 327.5 mbar) -> the motor retracts and
the run stops (pump 0 Hz). Squeeze never goes past --max-squeeze full steps beyond touch.
Plunger travel: about 0.023 mm per full step if the lead screw matches the 1 mL syringe calibration
(0.4049 uL per full step, BD 1 mL bore 4.78 mm); a 1.59 mm bore then closes within ~70 full steps past touch.

Setup: the plunger just off the tube at the start. That start position is a HARD back limit (NM 23 Nov: further
back the plunger falls off its stage): open, the safety retract and the search never go behind it;
pump at 0 Hz when the run starts (the sensor zero is taken then); outlet loop open to the reservoir.

usage:
  python make_config_pinch_pq.py --pump <id> --freqs 1150[,520] [--positions 0,10,20,30,40,50,60]
      [--targets 10,25,50,75,100,150,200,250,300] [--no-close] [--settle-s 5 --measure-s 20]
      [--hold-timeout-s 60] [--limit 330] [--max-stop 320] [--max-squeeze 150] [--start-retract 40]
      [--supply manual|rsd3305p] [--v 2500] [--dummy] [--out <file>]
"""
import argparse
import json

ap = argparse.ArgumentParser(description="PQ through the pinch-valve loop")
ap.add_argument("--pump", required=True)
ap.add_argument("--freqs", required=True, help="e.g. 1150 or 1150,520")
ap.add_argument("--positions", default="0,2,4,6,8,9,10,11,12,13,14,15,16",
                help="full steps past touch ('' = none); 23 Nov: 81 mbar at +10, > 320 at +15, so fine near closure")
ap.add_argument("--targets", default="10,25,50,75,100,150,200,250,300", help="mbar ('' = none)")
ap.add_argument("--no-close", action="store_true", help="skip the fully closed point (targets still map)")
ap.add_argument("--settle-s", type=float, default=5.0)
ap.add_argument("--measure-s", type=float, default=20.0)
ap.add_argument("--hold-timeout-s", type=float, default=60.0)
ap.add_argument("--spin-up-s", type=float, default=30.0,
                help="s at each new frequency before its first point (23 Nov: 1090 Hz after 520 Hz climbed 209 -> 420 uL/min over ~1 min)")
ap.add_argument("--limit", type=float, default=330.0, help="mbar; also capped at 95 %% of the sensor range")
ap.add_argument("--max-stop", type=float, default=300.0,
                help="closing sweep and position points stop here, minus a 10 mbar guard (mbar); on the steep closure "
                     "the last step can overshoot (23 Nov sim: 323-326 with a 320 stop): keep >= 25 below the 327.5 limit")
ap.add_argument("--max-squeeze", type=float, default=50.0, help="full steps past touch, hard cap (23 Nov: closed at 12-37)")
ap.add_argument("--start-retract", type=float, default=0.0, help="full steps retracted before the touch search (clamped by --back-limit)")
ap.add_argument("--back-limit", type=float, default=0.0, help="HARD floor in full steps from the start position (0 = never behind the start)")
ap.add_argument("--back-limit-pos", type=int, default=None,
                help="ABSOLUTE motor count of the hard back limit (23 Nov rig: 27133); a start behind it is refused")
ap.add_argument("--approach", type=float, default=0.0,
                help="full steps from the back limit moved at once before the fine search (23 Nov rig: 480; touch ~530)")
ap.add_argument("--max-search", type=float, default=800.0, help="full steps of touch search before giving up (~18 mm); then back to the start")
ap.add_argument("--touch-mbar", type=float, default=3.0)
ap.add_argument("--supply", choices=("manual", "rsd3305p"), default="manual")
ap.add_argument("--address", default="COM7")
ap.add_argument("--v", type=int, default=2500)
ap.add_argument("--firmware", choices=("v2", "v1", "auto"), default="v2",
                help="syringe firmware (23 Nov: COM6 runs syringe_v2; auto-detect misread it as v1 after the reset)")
ap.add_argument("--prime-f", type=int, default=0, help="priming frequency before the PQ (0 = no priming)")
ap.add_argument("--prime-rounds", type=int, default=4, help="priming rounds, each followed by the flow checks")
ap.add_argument("--checks", default="", help="flow-check frequencies after each priming round and after the PQ, e.g. 520,780,1090")
ap.add_argument("--check-s", type=float, default=20.0, help="s per check frequency (>= 10; 1 s steps are unusable)")
ap.add_argument("--dummy", action="store_true", help="simulated loop, motor and sensor (test without the rig)")
ap.add_argument("--out", default=None)
a = ap.parse_args()

if not 0 < a.v <= 2500:
    raise SystemExit("--v must be between 1 and 2500 mV")
freqs = [int(x) for x in a.freqs.split(",") if x.strip()]
positions = [float(x) for x in a.positions.split(",") if x.strip()]
targets = [float(x) for x in a.targets.split(",") if x.strip()]
if not freqs or any(not 1 <= f <= 5000 for f in freqs):
    raise SystemExit("--freqs: 1..5000 Hz")
if any(p < -20 or p > a.max_squeeze for p in positions):
    raise SystemExit("--positions must lie within -20..--max-squeeze full steps")
if not 20 <= a.limit <= 400 or a.max_stop >= a.limit:
    raise SystemExit("need 20 <= --limit <= 400 and --max-stop < --limit")

checks = [int(x) for x in a.checks.split(",") if x.strip()]
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


prime = []
if a.prime_f:
    F = a.prime_f
    for r in range(1, a.prime_rounds + 1):
        for f, sec in ((F, 30.0), (0, 5.0), (round(0.8 * F), 10.0), (0, 5.0), (round(1.2 * F), 10.0), (0, 10.0)):
            prime.append(zero(sec) if f == 0 else drive(f, sec, section="prime_at", rep=r))
        prime += check_blocks("prime_check", r)
elif checks:
    prime += check_blocks("prime_check", 0)

tests = []
if a.supply == "rsd3305p":
    tests.append({"type": "voltage", "voltage_mv": a.v, "while_waiting": "zero", "settle_s": 5,
                  "note": f"{a.v / 1000:.2f} V by the supply"})
tests += prime
tests += [{"type": "constant", "frequency_hz": 0, "duration_s": 10.0, "repeat": 1, "settle_ms": 0},
          {"type": "pinch_pq", "frequencies_hz": freqs, "positions_steps": positions, "targets_mbar": targets,
           "close_max": not a.no_close, "max_stop_mbar": a.max_stop, "settle_s": a.settle_s, "measure_s": a.measure_s,
           "hold_timeout_s": a.hold_timeout_s, "spin_up_s": a.spin_up_s, "section": "pinch_pq", "level_mv": a.v},
          {"type": "constant", "frequency_hz": 0, "duration_s": 10.0, "repeat": 1, "settle_ms": 0}]
tests += check_blocks("post_check", 0)
if a.supply == "manual":
    supply = {"mode": "manual", "current_limit_ma": 60, "max_mv": 2500, "initial_mv": a.v, "lead_min": 2}
    stxt = f"MANUAL bench supply set by hand to {a.v / 1000:.2f} V (switch it off by hand after the run)"
else:
    supply = {"mode": "rsd3305p", "channel": 1, "current_limit_ma": 60, "max_mv": 2500, "initial_mv": a.v,
              "tolerance_mv": 50, "step_timeout_s": 10, "off_on_exit": True, "eol": "LF", "address": a.address}
    stxt = f"{a.v / 1000:.2f} V by the RSD3305P ({a.address})"
pressure = {"enabled": True, "mode": "pinch", "backend": "dummy" if a.dummy else "arduino", "port": "COM6",
            "baud": 115200, "firmware": a.firmware, "sensor": {"backend": "fluigent", "channel": 1},
            "limit_mbar": a.limit, "min_mbar": -60.0, "microsteps": 16, "pos_min": 0, "pos_max": 2470,
            "pos_margin": 30, "clear_reserve_steps": 150, "ul_per_step": 0.4049,
            "pinch": {"touch_mbar": a.touch_mbar, "start_retract_steps": a.start_retract,
                      "back_limit_steps": a.back_limit, "back_limit_pos": a.back_limit_pos,
                      "approach_steps": a.approach, "max_search_steps": a.max_search,
                      "max_squeeze_steps": a.max_squeeze}}
desc = (f"PQ through the pinch-valve loop (motor squeezes the flex tube) at {', '.join(str(f) for f in freqs)} Hz: "
        f"touch search, open point, positions {positions or '-'} steps past touch, closing sweep to "
        f"{a.max_stop:g} mbar or plateau (blocked outlet){'' if not a.no_close else ' (not measured)'}, targets "
        f"{targets or '-'} mbar; settle {a.settle_s:g} s, measure {a.measure_s:g} s per point. {stxt}; PPK2 on; "
        f"limit {a.limit:g} mbar (and 95 % of the sensor range). Loop: pump -> flow -> pressure -> flex tube -> "
        f"reservoir. Plunger off the tube and pump at 0 Hz at the start. Results: pinch_pq.csv, pinch/."
        + (f" Before the PQ: {a.prime_rounds} priming rounds at {a.prime_f} Hz, each followed by flow checks at "
           f"{checks} Hz ({a.check_s:g} s each); primed = the last two rounds agree within ~3 %." if a.prime_f else "")
        + (f" After the PQ: flow checks at {checks} Hz (post_check)." if checks else ""))
cfg = {
    "port": "COM3", "baud": 115200, "read_timeout_s": 0.2, "start_time": None, "end_time": None,
    "break_seconds_between_tests": 0.0, "send_zero_on_exit": True, "require_pump_id": True,
    "meta": {"name": "Pinch PQ", "description": desc, "operator": "", "pump_id": a.pump, "freqs_hz": freqs},
    "serial": {"protocol": "freq_only"}, "protocol": "freq_only",
    "flow_sensor": {"backend": "dummy" if a.dummy else "fluigent", "channel": 0, "sample_rate_hz": 25.0},
    "pressure": pressure, "supply": supply,
    "power": {"enabled": not a.dummy, "backend": "ppk2", "port": "auto", "bin_ms": 10.0, "watch_ma": 57,
              "watch_hold_ms": 300, "raw_every_min": 10, "raw_length_s": 2.0, "post_run_tail_s": 10.0},
    "tests": tests,
}
per = a.settle_s + a.measure_s
npts = 1 + len(positions) + len(targets) + (0 if a.no_close else 1)
est = len(freqs) * (a.spin_up_s + 30 + npts * per + 20 * len(targets) + 30) + 20
est += sum(t["duration_s"] for t in prime) + len(checks) * (a.check_s + 10)
out = a.out or f"config_pinch_pq_{'_'.join(str(f) for f in freqs)}Hz_{a.supply}{'_dummy' if a.dummy else ''}.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
print(f"wrote {out}: {len(freqs)} frequency(ies), {npts} points each; about {est / 60:.0f} min")
