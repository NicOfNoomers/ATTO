"""PQ curves: pump flow vs outlet pressure, the pressure held by the stepper syringe (NM, 29 Oct 2026;
curves for pump 261030-1 with the programmable supply, 2 Nov 2026).

Rig: 1 mL liquid-filled syringe on the pump OUTLET, the pressure sensor (a Fluigent channel) between the
outlet and the syringe, stepper on an Arduino Uno (firmware/syringe_v2, 1/16 microstepping). The flow sensor
stays where it is. For each frequency the run holds each pressure in turn; once the 1 s mean has been
within 2 mbar for 3 s (or after settle_timeout_s, on_timeout "measure") it averages flow and pressure over
measure_s -> one row in pq_points.csv, one point on the "PQ curve" tab (pq_curve.png at the end).

The pump flow goes into the syringe, so the syringe fills up. Each PQ block starts with a clear if the
syringe is more than 20 % used; before a point that would not fit, and whenever fewer than 150 steps are
left, the run sets the pump to 0 Hz, lets the pressure down and waits: open the line to waste, "Push syringe
empty" (motor) in the banner, close the line, press "Done - continue". The point is then repeated. A pressure
the pump cannot reach (pull-only while pumping, the pressure has stopped rising below the target) is logged
as "above_shutoff" and the higher pressures of that frequency are skipped.

Pump 261030-1 (why: Paper/analysis_2026-11-02_lowf_voltage_261030-1/README.md, the 1-2 Nov low-f runs):
travel-limited stroke ~200-270 nL per cycle at 1-2 Hz, V0 tanh(1/(4 f tau)) with tau ~26-40 ms; V0 flat
above ~1.5 V and ~15-17 nL/mA below; ~89 ohm. PQ shows how stroke and flow hold up against back-pressure in
the low-f regime, at the usual operating frequencies, and at low voltage (best energy per uL at 0.8-1.0 V,
10-20 Hz).
- 2.5 V: 1, 5, 20, 100, 600, 1300 Hz; 1.0 V: 1, 20, 600 Hz; then 600 Hz at 2.5 V again (drift / state).
  Low frequency first within each voltage. --lean drops 5 and 1300 Hz (fewer clears, shorter run).
  At <= 5 Hz the shut-off plateau window is 15 s (shutoff_plateau_s), so a slow rise is not taken for it.
- 10, 25, 50, 75, 100, 150, 200, 250 mbar (limit 280); measure 30 s at <= 5 Hz (>= 30 strokes; pulsatile
  flow, so those points usually measure on the 30 s timeout), 20 s above; settle timeout 30 s.
- RSD3305P sets the voltages (supply.mode "rsd3305p", COM7 on the lab PC); PPK2 on, 10 ms bins; Pump ID
  required. No priming in the PQ config (the syringe would fill): the free-flow companion primes.
- Free-flow anchors (config_pq_261030-1_freeflow.json): outlet vented, NO syringe; same voltages and
  frequencies, 20 s each (30 s at <= 5 Hz), 20 s zeros, 3 x 600 Hz 60 s priming at the start. Run it right
  before connecting the syringe and right after removing it.

- SHORT variant (Nic, 2 Nov: a complete PQ in one day, short set-ups): 600 Hz only (best flow) at 2.5, 1.0 and
  0.8 V (0.8 V: the most efficient setting, 3.8 J/mL free-flow), same pressures, 20 s per point, plus its own
  free-flow anchors (priming, then 600 Hz 20 s at each voltage).

- DEAD-HEAD maximum pressure (Nic, 2 Nov; the PQ of 20261102_044118 / 051025 gave shut-off plateaus that still
  rise with frequency at 2.5 V: 70 mbar at 100 Hz, 137-159 at 600, 181 at 1300; 77 at 600 Hz, 1.0 V): the
  syringe controller holds the plunger still with the motor energised ("hold_position"), so the pump
  pressurises a fixed volume. Each step: 60 s drive (ended early at 270 mbar, below the 280 mbar limit, and
  logged as stop_above in pressure/events.json), 5 s at 0 Hz (leak-back at the maximum), release to ~0 mbar
  with the syringe, 15 s at 0 Hz. 2.5 V: 600, 800, 1000, 1300, 1600, 2000 Hz, then 1300 again (drift); 1.0 V:
  600 and 1300 Hz. Before each hold the run asks for a syringe clear if less than 30 % is left to pull back.
  The releases are gentle (15 mbar/s, stop at 3 mbar, never pull below it): after the PQ the -62 mbar
  undershoots had pulled gas out (free flow -25 % at 600 Hz, -40 % at 1300 Hz, 2 Nov).
- STRONG PRIMING (outlet vented to waste, no syringe pressure): 3 x [1300 Hz 10 s, 2000 Hz 10 s, 600 Hz 60 s]
  with 10 s zeros (high-frequency steps primed the other pumps in the purge runs; 600 Hz flushes), then a
  600 Hz 20 s check: about 200 uL/min or more at 2.5 V passes; lower, run it again. The three standard
  600 Hz 60 s primes did not recover the 2 Nov state shift. Order: strong priming (vented) -> connect the
  syringe through the stopcock -> blocked-outlet config.
- PRIME PLUS (Nic, 2 Nov: the strong prime 20261102_063050 raised 1300 Hz 147 -> 182 uL/min but 600 Hz drifted
  193 -> 179 and 2000 Hz barely pumps now, 41 -> 12; fresh state this morning 600 Hz 218-228, 1300 Hz 281):
  pump tilted OUTLET-UP, vented. 3 x [2 Hz 30 s (full-travel ~200-280 nL strokes, large displacement and peak
  velocity to move gas), 0 Hz 5 s, 1000 / 1300 / 1600 Hz 10 s each, 0 Hz 5 s, 600 Hz 45 s flush, 0 Hz 10 s],
  then checks 600 Hz 20 s and 1300 Hz 20 s with 15 s zeros. Pass: ~200 uL/min or more at 600 Hz and ~250 or
  more at 1300 Hz; else run it again. Keep the tilt for the following test.

Writes:
  config_pq_smoke.json                     one frequency, 3 pressures: checks the chain
  config_pq_261030-1.json                  the curves for 261030-1 (config_pq_261030-1_lean.json with --lean)
  config_pq_261030-1_freeflow.json         the 0 mbar anchors
  config_pq_261030-1_short.json            600 Hz at 2.5 / 1.0 / 0.8 V
  config_pq_261030-1_short_freeflow.json   its 0 mbar anchors
  config_pq_261030-1_deadhead.json         blocked-outlet maximum pressure vs frequency
  config_prime_strong_261030-1.json        strong priming + 600 Hz check (vented, no syringe)
  config_prime_plus_261030-1.json          2 Hz strokes + 1000-1600 Hz + flush, 600 / 1300 Hz checks (tilted)
The estimate printed at the end simulates the syringe (1 mL, 2470 full steps) with the free-outlet flows
below, for two cases: no shut-off below 250 mbar (every point measured: the upper bound) and a linear PQ
line to SHUTOFF_GUESS_MBAR (like 260930-2: ~54 mbar at 400 Hz on 29 Oct).
Before the first run: Pressure panel -> port of the syringe Uno, "Find" for the pressure channel, then
Connect Manual and run Calibrate... (zero, sweep) once; the calibration is kept on the PC.

usage: python make_config_pq.py [--lean] [--address COM7 | host:5025 | ""] [--pump-id 261030-1]
"""
import argparse
import json
import math

ap = argparse.ArgumentParser(description="PQ configs (syringe on the outlet)")
ap.add_argument("--lean", action="store_true", help="drop 5 Hz and 1300 Hz (fewer syringe clears)")
ap.add_argument("--address", default="COM7", help="RSD3305P address (default COM7, the lab PC); '' = Power panel")
ap.add_argument("--pump-id", default="261030-1")
args = ap.parse_args()

PUMP_PORT = "COM3"
SYRINGE_PORT = "COM6"          # the syringe Uno on the lab PC (29 Oct)
PRESSURE_CHANNEL = 1           # Fluigent index of the pressure sensor ("Find" in the Pressure panel)
FLOW_CHANNEL = 0
MICROSTEPS = 16                # A4988 with MS1-3 high (29 Oct); at full steps one step was 10-25 mbar
# From 10 mbar: holding 0 means pulling the whole pump flow at no back-pressure, where the syringe
# sticks and slips worst (lab 29 Oct). The 0 mbar (free-flow) point: outlet vented, no syringe.
PRESSURES_MBAR = [10, 25, 50, 75, 100, 150, 200, 250]    # the sensor reads to 345 mbar; the limit is 280
# The syringe stopper sticks and slips (+-30 mbar, lab 29 Oct), so points rarely meet the +-2 mbar
# settle rule; 30 s, then measure anyway (the row keeps the mean +- SD).
SETTLE_TIMEOUT_S = 30.0
MEASURE_LOW_S, MEASURE_S, LOW_F_MAX = 30.0, 20.0, 5     # 30 s at <= 5 Hz, 20 s above
# above_shutoff = below the target with the pressure rising < 2 mbar in this window (dashboard default 5 s). At
# <= 5 Hz the pump gives only ~10-17 uL/min, so a soft line may rise < 0.4 mbar/s: give it 15 s there
PLATEAU_LOW_S = 15.0

PRESSURE = {
    "enabled": True, "backend": "arduino", "port": SYRINGE_PORT, "baud": 115200, "firmware": "auto",
    "sensor": {"backend": "fluigent", "channel": PRESSURE_CHANNEL},
    # Min -60: the 10 mbar points dipped to -65/-78 mbar on 29 Oct and faulted on -30 (Nic then used -100)
    "limit_mbar": 280.0, "min_mbar": -60.0, "rate_mbar_s": 40.0,
    "tolerance_mbar": 2.0, "settle_s": 3.0, "microsteps": MICROSTEPS,
    # in FULL steps (multiplied by microsteps in the software)
    "pos_min": 0, "pos_max": 2470, "pos_margin": 30, "clear_reserve_steps": 150,
    # 1 mL over the old tool's 0..2470 range, per full step: only for the syringe-flow cross-check column
    "ul_per_step": round(1000.0 / 2470, 4),
}

# ---- pump 261030-1 ----
LEVELS = [(2500, [1, 5, 20, 100, 600, 1300]), (1000, [1, 20, 600]), (2500, [600])]
LEAN_DROP = {5, 1300}
# free-outlet flow, uL/min (top of the 1-2 Nov ranges; for the syringe estimate only)
Q_FREE = {(2500, 1): 17, (2500, 5): 66, (2500, 20): 143, (2500, 100): 126, (2500, 600): 230, (2500, 1300): 210,
          (1000, 1): 10, (1000, 20): 100, (1000, 600): 140}
SHORT_LEVELS = [(2500, [600]), (1000, [600]), (800, [600])]
Q_FREE_SHORT = {(2500, 600): 210, (1000, 600): 135, (800, 600): 93}     # uL/min free outlet (2 Nov estimate)
DEADHEAD_LEVELS = [(2500, [600, 800, 1000, 1300, 1600, 2000, 1300]), (1000, [600, 1300])]
DH_DRIVE_S, DH_STOP_MBAR, DH_OFF_S, DH_REST_S = 60.0, 270.0, 5.0, 15.0
DH_RELEASE_TIMEOUT_S, DH_MIN_PULL_ROOM = 45.0, 0.3
DH_RELEASE_RATE, DH_RELEASE_FLOOR = 15.0, 3.0     # gentle: 15 mbar/s ramp, stop at 3 mbar (no undershoot)
DH_RELEASE_GUESS_S = 15.0      # ~70-270 -> 3 mbar at 15 mbar/s, for the run-length estimate
PRIME_STRONG = [(1300, 10.0), (2000, 10.0), (600, 60.0)]
PRIME_STRONG_REPS, PRIME_ZERO_S, PRIME_CHECK = 3, 10.0, (600, 20.0)
PRIME_PASS_UL_MIN = 200
PRIME_PLUS_ROUND = [(2, 30.0), (0, 5.0), (1000, 10.0), (1300, 10.0), (1600, 10.0), (0, 5.0), (600, 45.0), (0, 10.0)]
PRIME_PLUS_REPS = 3
PRIME_PLUS_CHECKS = [(600, 20.0), (0, 15.0), (1300, 20.0), (0, 15.0)]
PRIME_PLUS_PASS = {600: 200, 1300: 250}
LIMIT_MA, TRIP_MA, HOLD_MS = 60, 57, 300
STEP_SETTLE_S = 5
FF_DRIVE_S, FF_DRIVE_LOW_S, FF_ZERO_S = 20.0, 30.0, 20.0
PRIME = (600, 60.0, 3)
SHUTOFF_GUESS_MBAR = 60.0
CLEAR_S = 90.0                 # operator time per syringe clear (let down, open to waste, push empty, close)
SETTLE_GUESS_S = 20.0          # the dashboard's own planning guess for a settling pressure (> 5 Hz)


def pq(freqs, pressures, measure_s, plateau_s=None):
    b = {"type": "pq", "frequency_hz": freqs if len(freqs) > 1 else freqs[0], "pressures_mbar": pressures,
         "settle_timeout_s": SETTLE_TIMEOUT_S, "measure_s": measure_s, "on_timeout": "measure",
         "release_after": True}
    if plateau_s:
        b["shutoff_plateau_s"] = plateau_s
    return b


def voltage(mv, note):
    return {"type": "voltage", "voltage_mv": mv, "while_waiting": "zero", "settle_s": STEP_SETTLE_S, "note": note}


def zero(s):
    return {"type": "constant", "frequency_hz": 0, "duration_s": float(s), "repeat": 1, "settle_ms": 0}


def drive(hz, s, **meta):
    return {"type": "constant", "frequency_hz": int(hz), "duration_s": float(s), "repeat": 1, "settle_ms": 0, **meta}


def config(name, desc, tests, pump_id="", pressure=True, supply=None, power=None):
    cfg = {
        "port": PUMP_PORT, "baud": 115200, "read_timeout_s": 0.2,
        "start_time": None, "end_time": None,
        "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
        "require_pump_id": True,
        "tests": tests,
        "meta": {"name": name, "description": desc, "operator": "", "pump_id": pump_id,
                 "syringe_ml": 1.0 if pressure else None, "syringe_on": "outlet" if pressure else None,
                 "pressure_channel": PRESSURE_CHANNEL},
        "serial": {"protocol": "freq_only"},
        "protocol": "freq_only",
    }
    if pressure:
        cfg["pressure"] = PRESSURE
    if supply is not None:
        cfg["flow_sensor"] = {"backend": "fluigent", "channel": FLOW_CHANNEL, "sample_rate_hz": 25.0}
        cfg["supply"] = supply
        cfg["power"] = power
    return cfg


def supply_section():
    s = {"mode": "rsd3305p", "channel": 1, "current_limit_ma": LIMIT_MA, "max_mv": 2500, "initial_mv": 2500,
         "tolerance_mv": 50, "step_timeout_s": 10, "off_on_exit": True, "eol": "LF"}
    if args.address:
        s["address"] = args.address
    return s


def power_section():
    return {"enabled": True, "backend": "ppk2", "port": "auto", "bin_ms": 10.0, "watch_ma": TRIP_MA,
            "watch_hold_ms": HOLD_MS, "raw_every_min": 10, "raw_length_s": 2.0, "post_run_tail_s": 10.0}


def levels():
    return [(mv, [f for f in fs if not (args.lean and f in LEAN_DROP)]) for mv, fs in LEVELS]


def build_pq(lv, name, ff_file, drift_index=2):
    tests = []
    for i, (mv, fs) in enumerate(lv):
        tests.append(voltage(mv, f"{mv / 1000:.1f} V by the supply" +
                             (" (drift / state check)" if i == drift_index else "")))
        low = [f for f in fs if f <= LOW_F_MAX]
        high = [f for f in fs if f > LOW_F_MAX]
        if low:
            tests.append(pq(low, PRESSURES_MBAR, MEASURE_LOW_S, PLATEAU_LOW_S))
        if high:
            tests.append(pq(high, PRESSURES_MBAR, MEASURE_S))
    for t in tests:
        if t["type"] == "pq":
            t["level_mv"] = next(x["voltage_mv"] for x in reversed(tests[:tests.index(t)]) if x["type"] == "voltage")
    curves = ", ".join(f"{mv / 1000:g} V: {'/'.join(map(str, fs))} Hz" for mv, fs in lv)
    desc = (f"PQ curves for pump {args.pump_id} with the syringe on the outlet: {curves}; "
            f"{', '.join(map(str, PRESSURES_MBAR))} mbar (limit 280), {MEASURE_LOW_S:g} s per point at <= "
            f"{LOW_F_MAX} Hz, {MEASURE_S:g} s above, settle timeout {SETTLE_TIMEOUT_S:g} s, above_shutoff skips the "
            f"rest of a curve. Voltages by the RSD3305P, PPK2 on. NO priming: run "
            f"{ff_file} (outlet vented, no syringe) before connecting the syringe and after "
            f"removing it. Start with an empty syringe; clear it when the banner asks (line open to waste, Push "
            f"syringe empty, close the line, Done).")
    return config(name, desc, tests, args.pump_id, True, supply_section(), power_section())


def build_freeflow(lv, name, check_index=2):
    tests = [zero(FF_ZERO_S)]
    first = True
    for i, (mv, fs) in enumerate(lv):
        tests.append(voltage(mv, f"{mv / 1000:.1f} V by the supply"))
        tests.append(zero(FF_ZERO_S))
        if first:
            for r in range(PRIME[2]):
                tests += [drive(PRIME[0], PRIME[1], section="priming", level_mv=mv, rep=r + 1), zero(FF_ZERO_S)]
            first = False
        for f in fs:
            tests += [drive(f, FF_DRIVE_LOW_S if f <= LOW_F_MAX else FF_DRIVE_S, section="free_flow", level_mv=mv,
                            check=(i == check_index)),
                      zero(FF_ZERO_S)]
    desc = (f"Free-flow (0 mbar) anchors for the PQ curves of pump {args.pump_id}: outlet VENTED, NO syringe. "
            f"3 x 600 Hz 60 s priming, then the PQ voltages and frequencies, {FF_DRIVE_S:g} s each ({FF_DRIVE_LOW_S:g} "
            f"s at <= {LOW_F_MAX} Hz), {FF_ZERO_S:g} s zeros. Run right before connecting the syringe and right after "
            f"removing it. Voltages by the RSD3305P, PPK2 on.")
    return config(name, desc, tests, args.pump_id, False, supply_section(), power_section())


def gentle_release():
    return {"type": "pressure", "target_mbar": "release", "timeout_s": DH_RELEASE_TIMEOUT_S,
            "release_rate_mbar_s": DH_RELEASE_RATE, "release_floor_mbar": DH_RELEASE_FLOOR}


def build_deadhead():
    tests = [voltage(DEADHEAD_LEVELS[0][0], f"{DEADHEAD_LEVELS[0][0] / 1000:.1f} V by the supply"), gentle_release()]
    k = 0
    for i, (mv, fs) in enumerate(DEADHEAD_LEVELS):
        if i:
            tests.append(voltage(mv, f"{mv / 1000:.1f} V by the supply"))
        for j, f in enumerate(fs):
            k += 1
            check = j > 0 and f in fs[:j]                   # a repeated frequency = drift check
            tests += [{"type": "pressure", "target_mbar": "hold_position", "min_pull_room": DH_MIN_PULL_ROOM},
                      drive(f, DH_DRIVE_S, section="deadhead", level_mv=mv, step=k, check=check,
                            stop_above_mbar=DH_STOP_MBAR),
                      zero(DH_OFF_S),
                      gentle_release(),
                      zero(DH_REST_S)]
    steps = "; ".join(f"{mv / 1000:g} V: {', '.join(map(str, fs))} Hz" for mv, fs in DEADHEAD_LEVELS)
    desc = (f"Blocked-outlet (dead-head) maximum pressure vs frequency, pump {args.pump_id}. The syringe controller "
            f"holds the plunger still (motor energised, no regulation) and the pump pressurises the fixed volume: "
            f"{DH_DRIVE_S:g} s per step (ended early at {DH_STOP_MBAR:g} mbar, logged as stop_above = maximum >= "
            f"{DH_STOP_MBAR:g}), {DH_OFF_S:g} s at 0 Hz, gentle release to ~{DH_RELEASE_FLOOR:g} mbar "
            f"({DH_RELEASE_RATE:g} mbar/s, never below), {DH_REST_S:g} s at 0 Hz. {steps}. Voltages by the RSD3305P, "
            f"PPK2 on. ORDER: 1) config_prime_strong_261030-1.json with the outlet vented (check block >= "
            f"~{PRIME_PASS_UL_MIN} uL/min at 600 Hz, else run it again); 2) connect the syringe through the stopcock, "
            f"syringe EMPTY (room to pull back for the releases), stopcock closed to waste; 3) this config. Open the "
            f"stopcock to waste before ANY manual syringe push: hand pushes against the closed line read 530-550 mbar "
            f"on 2 Nov, beyond the sensor range.")
    return config(f"Dead-head maximum pressure {args.pump_id}", desc, tests, args.pump_id, True, supply_section(),
                  power_section())


def build_prime_strong():
    mv = 2500
    tests = [zero(PRIME_ZERO_S), voltage(mv, f"{mv / 1000:.1f} V by the supply"), zero(PRIME_ZERO_S)]
    for r in range(PRIME_STRONG_REPS):
        for f, dur in PRIME_STRONG:
            tests += [drive(f, dur, section="prime_strong", level_mv=mv, rep=r + 1), zero(PRIME_ZERO_S)]
    tests += [drive(PRIME_CHECK[0], PRIME_CHECK[1], section="prime_check", level_mv=mv), zero(PRIME_ZERO_S)]
    seq = ", ".join(f"{f} Hz {d:g} s" for f, d in PRIME_STRONG)
    desc = (f"STRONG PRIMING for pump {args.pump_id}: outlet VENTED to waste, no syringe pressure (pressure control "
            f"off). {PRIME_STRONG_REPS} x [{seq}] with {PRIME_ZERO_S:g} s zeros at 2.5 V, then a {PRIME_CHECK[0]} Hz "
            f"{PRIME_CHECK[1]:g} s CHECK. Pass: about {PRIME_PASS_UL_MIN} uL/min or more in the check block; if lower, "
            f"run this config again before the blocked-outlet test. Voltages by the RSD3305P, PPK2 on. Then connect "
            f"the syringe through the stopcock and run config_pq_261030-1_deadhead.json.")
    return config(f"Strong priming {args.pump_id}", desc, tests, args.pump_id, False, supply_section(), power_section())


def build_prime_plus():
    mv = 2500
    tests = [zero(PRIME_ZERO_S), voltage(mv, f"{mv / 1000:.1f} V by the supply"), zero(PRIME_ZERO_S)]
    for r in range(PRIME_PLUS_REPS):
        for f, dur in PRIME_PLUS_ROUND:
            tests.append(zero(dur) if f == 0 else drive(f, dur, section="prime_plus", level_mv=mv, rep=r + 1))
    for f, dur in PRIME_PLUS_CHECKS:
        tests.append(zero(dur) if f == 0 else drive(f, dur, section="prime_check", level_mv=mv))
    rnd = ", ".join(f"{f} Hz {d:g} s" for f, d in PRIME_PLUS_ROUND)
    desc = (f"PRIME PLUS for pump {args.pump_id}: pump TILTED OUTLET-UP, outlet vented to waste, no syringe pressure "
            f"(pressure control off); keep the tilt for the following test. {PRIME_PLUS_REPS} x [{rnd}] at 2.5 V: the "
            f"2 Hz strokes are full travel (~200-280 nL) to move gas, 1000-1600 Hz prime, 600 Hz flushes (2000 Hz is "
            f"left out: it barely pumps in the current state). Then CHECKS: 600 Hz 20 s and 1300 Hz 20 s. Pass: about "
            f"{PRIME_PLUS_PASS[600]} uL/min or more at 600 Hz and about {PRIME_PLUS_PASS[1300]} or more at 1300 Hz; "
            f"else run it again. Voltages by the RSD3305P, PPK2 on.")
    return config(f"Prime plus {args.pump_id}", desc, tests, args.pump_id, False, supply_section(), power_section())


def estimate(cfg, shutoff, q_free):
    """Simulated run: (minutes incl. clears, clears per PQ block, measured points)."""
    usable_ul = (PRESSURE["pos_max"] - PRESSURE["pos_min"] - 2 * PRESSURE["pos_margin"]) * PRESSURE["ul_per_step"]
    reserve_ul = PRESSURE["clear_reserve_steps"] * PRESSURE["ul_per_step"]
    used, t, clears, per_block, points = 0.0, 0.0, 0, [], 0
    mv = 2500
    for blk in cfg["tests"]:
        if blk["type"] == "voltage":
            mv = blk["voltage_mv"]
            t += 1.0 + blk["settle_s"]
            continue
        c0 = clears
        if used > 0.2 * usable_ul:                      # each PQ block starts (almost) empty
            clears += 1
            used = 0.0
        fs = blk["frequency_hz"] if isinstance(blk["frequency_hz"], list) else [blk["frequency_hz"]]
        for f in fs:
            q0 = q_free[mv, f]
            settle = SETTLE_TIMEOUT_S if f <= LOW_F_MAX else SETTLE_GUESS_S
            for p in blk["pressures_mbar"]:
                q = q0 * max(0.0, 1.0 - p / shutoff) if shutoff else q0
                if shutoff and q <= 0.0:
                    t += SETTLE_TIMEOUT_S               # above_shutoff found within the settle timeout
                    break
                need = q / 60.0 * (blk["measure_s"] + min(SETTLE_TIMEOUT_S, SETTLE_GUESS_S)) * 1.3
                if usable_ul - used < reserve_ul + need:
                    clears += 1
                    used = 0.0
                used += q / 60.0 * (settle + blk["measure_s"])
                t += settle + blk["measure_s"]
                points += 1
        t += 10.0                                       # release_after
        per_block.append((mv, fs, clears - c0))
    t += clears * CLEAR_S + 10.0
    return t / 60.0, clears, per_block, points


smoke = config("PQ smoke test", "One PQ curve at 400 Hz, 10/25/50 mbar, 15 s each: checks syringe, pressure sensor, "
                                "flow and the clearing prompt before the full run.",
               [pq([400], [10, 25, 50], 15.0)])
tag = "_lean" if args.lean else ""
full = build_pq(levels(), f"PQ curves {args.pump_id}" + (" (lean)" if args.lean else ""),
                "config_pq_261030-1_freeflow.json")
ff = build_freeflow(LEVELS, f"PQ free-flow anchors {args.pump_id}")     # every frequency: anchors for full and lean
short = build_pq(SHORT_LEVELS, f"PQ curves {args.pump_id} (short: 600 Hz at 2.5 / 1.0 / 0.8 V)",
                 "config_pq_261030-1_short_freeflow.json", drift_index=None)
short_ff = build_freeflow(SHORT_LEVELS, f"PQ free-flow anchors {args.pump_id} (short)", check_index=None)
deadhead = build_deadhead()
prime_strong = build_prime_strong()
prime_plus = build_prime_plus()
for fname, cfg in (("config_pq_smoke.json", smoke), (f"config_pq_261030-1{tag}.json", full),
                   ("config_pq_261030-1_freeflow.json", ff), ("config_pq_261030-1_short.json", short),
                   ("config_pq_261030-1_short_freeflow.json", short_ff),
                   ("config_pq_261030-1_deadhead.json", deadhead),
                   ("config_prime_strong_261030-1.json", prime_strong),
                   ("config_prime_plus_261030-1.json", prime_plus)):
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    n = sum(len(t["pressures_mbar"]) * (len(t["frequency_hz"]) if isinstance(t["frequency_hz"], list) else 1)
            for t in cfg["tests"] if t["type"] == "pq")
    nom = sum(t.get("duration_s", 0) for t in cfg["tests"] if t["type"] == "constant") + \
        sum(t.get("settle_s", 0) for t in cfg["tests"] if t["type"] == "voltage")
    print(f"{fname}: {n} PQ points" + (f", {nom / 60:.1f} min" if not n else ""))

for fname, cfg, qf in ((f"config_pq_261030-1{tag}.json", full, Q_FREE),
                       ("config_pq_261030-1_short.json", short, Q_FREE_SHORT)):
    print(f"\nEstimate for {fname} (syringe 1 mL, clear when >20 % used at a block start or a point would not fit; "
          f"{CLEAR_S:g} s per clear; settle {SETTLE_TIMEOUT_S:g} s at <= {LOW_F_MAX} Hz, {SETTLE_GUESS_S:g} s above):")
    for label, shut in (("no shut-off below 250 mbar (upper bound)", None),
                        (f"linear PQ to a {SHUTOFF_GUESS_MBAR:g} mbar shut-off", SHUTOFF_GUESS_MBAR)):
        mins, clears, per_block, points = estimate(cfg, shut, qf)
        blocks = "; ".join(f"{mv / 1000:g} V {'/'.join(map(str, fs))} Hz: {c}" for mv, fs, c in per_block)
        print(f"  {label}: {points} points, {clears} clears, {mins:.0f} min incl. clears  [clears per block: {blocks}]")

n_dh = sum(len(fs) for _, fs in DEADHEAD_LEVELS)
t_dh = (n_dh * (DH_DRIVE_S + DH_OFF_S + DH_RELEASE_GUESS_S + DH_REST_S) + len(DEADHEAD_LEVELS) * (1.0 + STEP_SETTLE_S)
        + DH_RELEASE_GUESS_S + 10.0)
print(f"\nconfig_pq_261030-1_deadhead.json: {n_dh} steps, ~{t_dh / 60:.1f} min ({DH_DRIVE_S:g} s drive + {DH_OFF_S:g} s off + "
      f"~{DH_RELEASE_GUESS_S:g} s release + {DH_REST_S:g} s rest per step; shorter when a step stops at "
      f"{DH_STOP_MBAR:g} mbar); a syringe clear only if less than {DH_MIN_PULL_ROOM:.0%} is left to pull back")
