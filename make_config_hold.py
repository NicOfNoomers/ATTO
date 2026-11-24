"""Continuous hold at one frequency (free flow), for a stability check before a PQ (NM, 18 Nov 2026).

Layout: 20 s at 0 Hz, --f for --minutes continuously, 20 s at 0 Hz. Read the flow drift from merged.csv
(e.g. 1-min means): within +-10 % over the hold = steady enough for a PQ at that frequency.
usage: python make_config_hold.py --pump <id> --f 1150 [--minutes 10] [--supply manual|rsd3305p] [--v 2500] [--out <file>]

With checks (NM 24 Nov: "run a 30-minute hold for 450 hertz and see what's going on"):
  --prime-f 1000 [--prime-rounds 2]: rounds of [F 30 s, 0 Hz 5 s, 0.8 F 10 s, 0 Hz 5 s, 1.2 F 10 s, 0 Hz 10 s] first;
  --checks 300,450,600,800,1000 --check-every 3 [--check-s 10]: the hold is split into --check-every minute
  segments, each followed by the check frequencies (check_s each, back to back) - a check round before the
  first segment too. Sections: prime_at / hold (rep = segment) / hold_check (rep = round).
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
ap.add_argument("--prime-f", type=int, default=0, help="prime around this frequency first (0 = no priming)")
ap.add_argument("--prime-rounds", type=int, default=2)
ap.add_argument("--checks", default="", help="check frequencies between hold segments, e.g. 300,450,600,800,1000")
ap.add_argument("--check-every", type=float, default=3.0, help="minutes of hold between check rounds")
ap.add_argument("--check-s", type=float, default=10.0, help="s per check frequency (>= 5; 1 s steps are unusable)")
ap.add_argument("--out", default=None)
a = ap.parse_args()
if not (0 < a.v <= 2500 and 1 <= a.f <= 5000 and 0 < a.minutes <= 600):
    raise SystemExit("need 0 < v <= 2500 mV, 1 <= f <= 5000 Hz, 0 < minutes <= 600")
tests = []
if a.supply == "rsd3305p":
    tests.append({"type": "voltage", "voltage_mv": a.v, "while_waiting": "zero", "settle_s": 5,
                  "note": f"{a.v / 1000:.2f} V by the supply"})
checks = [int(x) for x in a.checks.split(",") if x.strip()]
if checks and (a.check_s < 5 or a.check_every <= 0 or any(not 1 <= f <= 5000 for f in checks)):
    raise SystemExit("--checks: 1..5000 Hz, --check-s >= 5, --check-every > 0")


def blk(hz, sec, **meta):
    b = {"type": "constant", "frequency_hz": int(hz), "duration_s": float(sec), "repeat": 1, "settle_ms": 0}
    if hz:
        b.update({"level_mv": a.v, **meta})
    return b


tests.append(blk(0, 20.0))
if a.prime_f:
    F = a.prime_f
    for r in range(1, a.prime_rounds + 1):
        for f, sec in ((F, 30.0), (0, 5.0), (round(0.8 * F), 10.0), (0, 5.0), (round(1.2 * F), 10.0), (0, 10.0)):
            tests.append(blk(f, sec, section="prime_at", rep=r))
if checks:
    nseg = max(1, int(round(a.minutes / a.check_every)))
    seg_s = a.minutes * 60.0 / nseg
    tests += [blk(f, a.check_s, section="hold_check", rep=0, check_hz=f) for f in checks]
    for k in range(1, nseg + 1):
        tests.append(blk(a.f, seg_s, section="hold", rep=k))
        tests += [blk(f, a.check_s, section="hold_check", rep=k, check_hz=f) for f in checks]
else:
    tests.append(blk(a.f, a.minutes * 60.0, section="hold"))
tests.append(blk(0, 20.0))
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
