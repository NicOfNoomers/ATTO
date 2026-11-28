"""Priming-state calibration on a see-through pump, filmed (NM, 27 Nov 2026).

Question: what does the gas inside the pump look like, and how does the full frequency response change with it?
The November full scans show the same pump switching its high band (>= 1000 Hz) on and off and moving its peak
(260930-2 700-800 <-> 1300 Hz; 261123-1 1300 Hz 213 -> 17 -> 4 -> 146 uL/min) at an unchanged current
(Paper/analysis_2026-11-27_priming_state). This run films the chamber while the state is pushed in known ways and
takes a FULL quick scan (12 frequencies, not 2-3 check frequencies) after each push, so a visible bubble can be
matched to the curve. The scans are the candidate "gas gauge" (high-band share, peak frequency).

Sequence (default ~36 min), video the whole time (run with --video --camera <index of the camera on the chamber>):
  scan A (as found)
  push 1: high-frequency drive (--hf, 60 s; on 261123-1 this killed the high band)          -> scan B
  push 2: pumping-frequency drive (--pf, 3 x 30 s; on 261123-1 this restored it)              -> scan C
  push 3: pressure pulse through the pinch tube at --pf: hold --p-hold mbar for --hold-s, open
          (on 261123-2 a gas lock during a PQ released and the flow jumped 45 -> 365)        -> scan D
  push 4: rest at 0 Hz (--rest-s; does a bubble move, shrink or redissolve?)                 -> scan E
Each scan: 100 ... 1600 Hz, 10 s each with 5 s zeros (make_config_scan_quick.py), section scan_A ... scan_E.
Loop: reservoir -> pump -> flow -> pressure -> flex tube (pinch) -> reservoir (the 23 Nov pinch loop), plunger
just off the tube at the start (that position is the HARD back limit), pump at 0 Hz at the start (sensor zero).
--no-pinch: skip push 3 (no tube/motor), outlet vented.

Safety: pressure above min(--limit, 95 % of the sensor range) -> motor retracts, run stops (pump 0 Hz); squeeze
never past --max-squeeze full steps beyond touch; RSD3305P at --v (<= 2.5 V), PPK2 trip 57 mA / 300 ms.

usage: python make_config_state_cal.py --pump <id> [--pf 600] [--hf 1300] [--p-hold 200] [--hold-s 30]
        [--rest-s 300] [--no-pinch] [--v 2500] [--address COM7] [--out <file>]
"""
import argparse
import json

ap = argparse.ArgumentParser(description="Filmed priming-state calibration (see-through pump)")
ap.add_argument("--pump", required=True)
ap.add_argument("--pf", type=int, default=600, help="the pump's own pumping frequency (Hz): push 2 and the pressure pulse")
ap.add_argument("--hf", type=int, default=1300, help="high-frequency push (Hz)")
ap.add_argument("--hf-s", type=float, default=60.0)
ap.add_argument("--p-hold", type=float, default=200.0, help="pressure pulse (mbar), held with the pinch")
ap.add_argument("--hold-s", type=float, default=30.0)
ap.add_argument("--rest-s", type=float, default=300.0)
ap.add_argument("--no-pinch", action="store_true", help="no pressure pulse (outlet vented, no tube/motor)")
ap.add_argument("--limit", type=float, default=300.0)
ap.add_argument("--max-squeeze", type=float, default=100.0)
ap.add_argument("--max-search", type=float, default=800.0)
ap.add_argument("--v", type=int, default=2500)
ap.add_argument("--address", default="COM7")
ap.add_argument("--ilim", type=int, default=60, help="supply current limit and PPK2 trip (mA); NM 27 Nov: 110 for the "
                "see-through pump 260930_04 (90-95 mA for 1.4-3 s at each drive start in October)")
ap.add_argument("--out", default=None)
a = ap.parse_args()
if not 20 <= a.ilim <= 120:
    raise SystemExit("--ilim: 20..120 mA")
if not 0 < a.v <= 2500:
    raise SystemExit("--v must be between 1 and 2500 mV")
if not (1 <= a.pf <= 3000 and 1 <= a.hf <= 3000):
    raise SystemExit("--pf / --hf: 1..3000 Hz")
if not 20 <= a.p_hold <= a.limit - 40:
    raise SystemExit("--p-hold must lie between 20 and --limit - 40 mbar")

FREQS = [100, 150, 200, 300, 400, 500, 600, 700, 800, 1000, 1300, 1600]


def zero(s):
    return {"type": "constant", "frequency_hz": 0, "duration_s": float(s), "repeat": 1, "settle_ms": 0}


def drive(hz, s, **meta):
    return {"type": "constant", "frequency_hz": int(hz), "duration_s": float(s), "repeat": 1, "settle_ms": 0,
            "level_mv": a.v, **meta}


def scan(name):
    out = [zero(5.0)]
    for f in FREQS:
        out += [drive(f, 10.0, section=name), zero(5.0)]
    return out


tests = [zero(10.0), {"type": "voltage", "voltage_mv": a.v, "while_waiting": "zero", "settle_s": 5,
                      "note": f"{a.v / 1000:.2f} V by the supply"}]
tests += scan("scan_A")
tests += [drive(a.hf, a.hf_s, section="push1_hf"), zero(10.0)] + scan("scan_B")
for r in range(3):
    tests += [drive(a.pf, 30.0, section="push2_pf", rep=r + 1), zero(5.0)]
tests += [zero(5.0)] + scan("scan_C")
if not a.no_pinch:
    tests += [{"type": "pinch", "action": "find_touch", "frequency_hz": a.pf, "spin_up_s": 5, "section": "push3_pressure",
               "level_mv": a.v},
              {"type": "pinch", "action": "hold", "target_mbar": a.p_hold, "timeout_s": 60, "section": "push3_pressure",
               "level_mv": a.v},
              drive(a.pf, a.hold_s, section="push3_pressure"),
              {"type": "pinch", "action": "open", "section": "push3_pressure"},
              drive(a.pf, 10.0, section="push3_pressure"), zero(10.0)]
    tests += scan("scan_D")
tests += [zero(a.rest_s)] + scan("scan_E")

supply = {"mode": "rsd3305p", "channel": 1, "current_limit_ma": a.ilim, "max_mv": 2500, "initial_mv": a.v,
          "tolerance_mv": 50, "step_timeout_s": 10, "off_on_exit": True, "eol": "LF", "address": a.address}
cfg = {
    "port": "COM3", "baud": 115200, "read_timeout_s": 0.2, "start_time": None, "end_time": None,
    "break_seconds_between_tests": 0.0, "send_zero_on_exit": True, "require_pump_id": True,
    "meta": {"name": "Priming-state calibration", "operator": "", "pump_id": a.pump, "freqs_hz": FREQS,
             "description": (f"Filmed priming-state calibration on a see-through pump: full quick scans "
                             f"({', '.join(map(str, FREQS))} Hz, 10 s each) as found (A), after {a.hf_s:g} s at "
                             f"{a.hf} Hz (B), after 3 x 30 s at {a.pf} Hz (C)"
                             + ("" if a.no_pinch else f", after a {a.p_hold:g} mbar pinch hold for {a.hold_s:g} s at "
                                                      f"{a.pf} Hz (D)")
                             + f", after {a.rest_s:g} s rest (E). Run with --video --camera <chamber camera>. "
                               f"{a.v / 1000:.2f} V by the RSD3305P, PPK2 on.")},
    "serial": {"protocol": "freq_only"}, "protocol": "freq_only",
    "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
    "supply": supply,
    "power": {"enabled": True, "backend": "ppk2", "port": "auto", "bin_ms": 10.0, "watch_ma": a.ilim - 3 if a.ilim <= 60 else a.ilim,
              "watch_hold_ms": 300, "raw_every_min": 10, "raw_length_s": 2.0, "post_run_tail_s": 10.0},
    "tests": tests,
}
if not a.no_pinch:
    cfg["pressure"] = {"enabled": True, "mode": "pinch", "backend": "arduino", "port": "COM6", "baud": 115200,
                       "firmware": "v2", "sensor": {"backend": "fluigent", "channel": 1},
                       "limit_mbar": a.limit, "min_mbar": -60.0, "microsteps": 16, "pos_min": 0, "pos_max": 2470,
                       "pos_margin": 30, "clear_reserve_steps": 150, "ul_per_step": 0.4049,
                       "pinch": {"touch_mbar": 3.0, "start_retract_steps": 0.0, "back_limit_steps": 0.0,
                                 "back_limit_pos": None, "approach_steps": 0.0, "max_search_steps": a.max_search,
                                 "max_squeeze_steps": a.max_squeeze}}
dur = sum(t.get("duration_s", 0) for t in tests if t["type"] == "constant") + (0 if a.no_pinch else 60 + 60)
out = a.out or f"config_state_cal_{a.pump}{'_nopinch' if a.no_pinch else ''}.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
print(f"wrote {out}: {len(tests)} blocks, about {dur / 60:.0f} min")
