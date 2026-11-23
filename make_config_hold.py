"""Continuous hold at one frequency (free flow), for a stability check before a PQ (NM, 18 Nov 2026).

Layout: 20 s at 0 Hz, --f for --minutes continuously, 20 s at 0 Hz. Read the flow drift from merged.csv
(e.g. 1-min means): within +-10 % over the hold = steady enough for a PQ at that frequency.
usage: python make_config_hold.py --pump <id> --f 1150 [--minutes 10] [--supply manual|rsd3305p] [--v 2500] [--out <file>]
"""
import argparse
import json

ap = argparse.ArgumentParser(description="Continuous hold at one frequency (stability check)")
ap.add_argument("--pump", required=True)
ap.add_argument("--f", type=int, required=True)
ap.add_argument("--minutes", type=float, default=10.0)
ap.add_argument("--supply", choices=("manual", "rsd3305p"), default="manual")
ap.add_argument("--address", default="COM7")
ap.add_argument("--v", type=int, default=2500)
ap.add_argument("--out", default=None)
a = ap.parse_args()
if not (0 < a.v <= 2500 and 1 <= a.f <= 5000 and 0 < a.minutes <= 600):
    raise SystemExit("need 0 < v <= 2500 mV, 1 <= f <= 5000 Hz, 0 < minutes <= 600")
tests = []
if a.supply == "rsd3305p":
    tests.append({"type": "voltage", "voltage_mv": a.v, "while_waiting": "zero", "settle_s": 5,
                  "note": f"{a.v / 1000:.2f} V by the supply"})
tests += [{"type": "constant", "frequency_hz": 0, "duration_s": 20.0, "repeat": 1, "settle_ms": 0},
          {"type": "constant", "frequency_hz": a.f, "duration_s": a.minutes * 60.0, "repeat": 1, "settle_ms": 0,
           "section": "hold", "level_mv": a.v},
          {"type": "constant", "frequency_hz": 0, "duration_s": 20.0, "repeat": 1, "settle_ms": 0}]
if a.supply == "manual":
    supply = {"mode": "manual", "current_limit_ma": 60, "max_mv": 2500, "initial_mv": a.v, "lead_min": 2}
else:
    supply = {"mode": "rsd3305p", "channel": 1, "current_limit_ma": 60, "max_mv": 2500, "initial_mv": a.v,
              "tolerance_mv": 50, "step_timeout_s": 10, "off_on_exit": True, "eol": "LF", "address": a.address}
cfg = {"port": "COM3", "baud": 115200, "read_timeout_s": 0.2, "start_time": None, "end_time": None,
       "break_seconds_between_tests": 0.0, "send_zero_on_exit": True, "require_pump_id": True,
       "meta": {"name": f"Hold {a.f} Hz", "operator": "", "pump_id": a.pump,
                "description": f"Continuous {a.f} Hz for {a.minutes:g} min, free flow, outlet vented, "
                               f"{a.v / 1000:.2f} V ({a.supply} supply), PPK2 on: stability check (+-10 %)."},
       "serial": {"protocol": "freq_only"}, "protocol": "freq_only",
       "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0}, "supply": supply,
       "power": {"enabled": True, "backend": "ppk2", "port": "auto", "bin_ms": 10.0, "watch_ma": 57,
                 "watch_hold_ms": 300, "raw_every_min": 10, "raw_length_s": 2.0, "post_run_tail_s": 10.0},
       "tests": tests}
out = a.out or f"config_hold_{a.f}Hz_{a.minutes:g}min_{a.supply}.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
print(f"wrote {out}: {a.f} Hz for {a.minutes:g} min")
