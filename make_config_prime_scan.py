"""Priming with full-scan checks, open loop, any pump (NM, 27 Nov 2026).

The checks are FULL quick scans (12 frequencies, 100-1600 Hz), not 2-3 frequencies: within one pump the peak moves
(400 -> 1300 Hz) and the high band switches on and off with the priming state (Paper/analysis_2026-11-27_priming_state).
Sequence: scan_A (as found) -> ROUNDS x [prime round at F (F 30 s, 0.8 F 10 s, 1.2 F 10 s, as make_config_prime_at.py),
scan_<round>]. Primed = two consecutive scans agree within ~3 % at every frequency that pumps.
Loop open to the reservoir (no pressure control, no syringe/pinch), RSD3305P at --v, PPK2 on, Pump ID required.

usage: python make_config_prime_scan.py --pump <id> [--f 600] [--rounds 4] [--v 2500] [--address COM7] [--ilim 60]
       [--out <file>]
"""
import argparse
import json

ap = argparse.ArgumentParser(description="Priming rounds with full-scan checks (open loop)")
ap.add_argument("--pump", required=True)
ap.add_argument("--f", type=int, default=600, help="priming frequency (Hz): the pump's own best")
ap.add_argument("--rounds", type=int, default=4)
ap.add_argument("--v", type=int, default=2500)
ap.add_argument("--address", default="COM7")
ap.add_argument("--ilim", type=int, default=60, help="supply limit and PPK2 trip (mA); 57 mA trip at the default 60")
ap.add_argument("--out", default=None)
a = ap.parse_args()
if not 0 < a.v <= 2500:
    raise SystemExit("--v must be between 1 and 2500 mV")
if not (1 <= round(0.8 * a.f) and round(1.2 * a.f) <= 3000):
    raise SystemExit("--f: 0.8 F .. 1.2 F must lie within 1..3000 Hz")
if not 20 <= a.ilim <= 120 or not 1 <= a.rounds <= 8:
    raise SystemExit("--ilim 20..120 mA, --rounds 1..8")

FREQS = [100, 150, 200, 300, 400, 500, 600, 700, 800, 1000, 1300, 1600]


def zero(s):
    return {"type": "constant", "frequency_hz": 0, "duration_s": float(s), "repeat": 1, "settle_ms": 0}


def drive(hz, s, **meta):
    return {"type": "constant", "frequency_hz": int(hz), "duration_s": float(s), "repeat": 1, "settle_ms": 0,
            "level_mv": a.v, **meta}


def scan(name, rnd):
    out = [zero(5.0)]
    for fr in FREQS:
        out += [drive(fr, 10.0, section=name, rep=rnd, check_hz=fr), zero(5.0)]
    return out


F = a.f
tests = [zero(10.0), {"type": "voltage", "voltage_mv": a.v, "while_waiting": "zero", "settle_s": 5,
                      "note": f"{a.v / 1000:.2f} V by the supply"}]
tests += scan("scan_A", 0)
for r in range(1, a.rounds + 1):
    for fr, sec in ((F, 30.0), (0, 5.0), (round(0.8 * F), 10.0), (0, 5.0), (round(1.2 * F), 10.0), (0, 10.0)):
        tests.append(zero(sec) if fr == 0 else drive(fr, sec, section="prime_at", rep=r))
    tests += scan("scan_round", r)
supply = {"mode": "rsd3305p", "channel": 1, "current_limit_ma": a.ilim, "max_mv": 2500, "initial_mv": a.v,
          "tolerance_mv": 50, "step_timeout_s": 10, "off_on_exit": True, "eol": "LF", "address": a.address}
cfg = {
    "port": "COM3", "baud": 115200, "read_timeout_s": 0.2, "start_time": None, "end_time": None,
    "break_seconds_between_tests": 0.0, "send_zero_on_exit": True, "require_pump_id": True,
    "meta": {"name": "Prime with scan checks", "operator": "", "pump_id": a.pump, "freqs_hz": FREQS,
             "description": (f"Open loop (no pressure control): full quick scan as found (scan_A), then {a.rounds} priming "
                             f"rounds at {F} Hz (30 s, 0.8 F 10 s, 1.2 F 10 s), each followed by a full quick scan "
                             f"({', '.join(map(str, FREQS))} Hz, 10 s each). Primed = two consecutive scans within ~3 %. "
                             f"{a.v / 1000:.2f} V by the RSD3305P, PPK2 on.")},
    "serial": {"protocol": "freq_only"}, "protocol": "freq_only",
    "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
    "supply": supply,
    "power": {"enabled": True, "backend": "ppk2", "port": "auto", "bin_ms": 10.0,
              "watch_ma": a.ilim - 3 if a.ilim <= 60 else a.ilim, "watch_hold_ms": 300, "raw_every_min": 10,
              "raw_length_s": 2.0, "post_run_tail_s": 10.0},
    "tests": tests,
}
dur = sum(t.get("duration_s", 0) for t in tests if t["type"] == "constant") + 5
out = a.out or f"config_prime_scan_{a.pump}_{F}Hz.json"
with open(out, "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, indent=2)
print(f"wrote {out}: scan + {a.rounds} x [prime at {F} Hz + scan], about {dur / 60:.0f} min")
