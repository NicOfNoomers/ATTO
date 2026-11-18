"""Fine free-flow frequency sweep with drift references (NM, 18 Nov 2026: "a full freq sweep for flow before and
after priming, 5 Hz resolution").

Layout:
  1. flow check: 30 s at 0 Hz, the reference frequency 10 s, the low reference 10 s, 20 s at 0 Hz;
  2. ascending step sweep fmin..fmax in --step Hz steps, --dwell s each, no gaps (sweep blocks, one per
     --chunk Hz);
  3. after each chunk, the reference frequency for --ref-s (the pump state drifts at the high band: 260930-2,
     18 Nov, 1250 Hz 229 -> 74-88 uL/min within a minute), so drift can be read off;
  4. 20 s at 0 Hz at the end.
Run it before and after a priming config. Analysis: each step's flow = mean of its last 60 % (the leading 40 %
is settling; 1 s steps are unusable, so the dwell defaults to 3 s).

Default: 100-2000 Hz in 5 Hz steps (381 steps x 3 s = 19 min) with a 1150 Hz reference every 100 Hz
(20 x 5 s); about 22 min per sweep.

usage:
  python make_config_flow_sweep_fine.py --pump <id> [--fmin 100 --fmax 2000 --step 5 --dwell 3]
      [--chunk 100] [--ref 1150 --ref-s 5] [--ref-low 520] [--supply manual|rsd3305p] [--v 2500] [--out <file>]
"""
import argparse
import json

ap = argparse.ArgumentParser(description="Fine free-flow frequency sweep with drift references")
ap.add_argument("--pump", required=True)
ap.add_argument("--fmin", type=int, default=100)
ap.add_argument("--fmax", type=int, default=2000)
ap.add_argument("--step", type=int, default=5)
ap.add_argument("--dwell", type=float, default=3.0, help="s per frequency (>= 2; 1 s steps are unusable)")
ap.add_argument("--chunk", type=int, default=100, help="Hz per sweep block between references")
ap.add_argument("--ref", type=int, default=1150, help="drift reference frequency (0 = none)")
ap.add_argument("--ref-s", type=float, default=5.0)
ap.add_argument("--ref-low", type=int, default=520, help="second check frequency in the flow check (0 = none)")
ap.add_argument("--supply", choices=("manual", "rsd3305p"), default="manual")
ap.add_argument("--address", default="COM7")
ap.add_argument("--v", type=int, default=2500)
ap.add_argument("--descending", action="store_true", help="sweep fmax -> fmin (with --ref 0: one continuous sweep)")
ap.add_argument("--out", default=None)
a = ap.parse_args()

if not 0 < a.v <= 2500:
    raise SystemExit("--v must be between 1 and 2500 mV")
if a.dwell < 2.0:
    raise SystemExit("--dwell must be >= 2 s (1 s steps are unusable: sensor lag)")
if not (1 <= a.fmin < a.fmax <= 5000 and a.step >= 1 and a.chunk >= a.step):
    raise SystemExit("need 1 <= fmin < fmax <= 5000, step >= 1, chunk >= step")

tests = []
if a.supply == "rsd3305p":
    tests.append({"type": "voltage", "voltage_mv": a.v, "while_waiting": "zero", "settle_s": 5,
                  "note": f"{a.v / 1000:.2f} V by the supply"})
tests.append({"type": "constant", "frequency_hz": 0, "duration_s": 30.0, "repeat": 1, "settle_ms": 0,
              "section": "flow_check"})
for f in (a.ref, a.ref_low):
    if f > 0:
        tests.append({"type": "constant", "frequency_hz": f, "duration_s": 10.0, "repeat": 1, "settle_ms": 0,
                      "section": "flow_check", "level_mv": a.v})
tests.append({"type": "constant", "frequency_hz": 0, "duration_s": 20.0, "repeat": 1, "settle_ms": 0})
n_steps = 0
chunks = []
lo = a.fmin
while lo <= a.fmax:
    hi = min(a.fmax, lo + a.chunk - a.step)
    n = (hi - lo) // a.step + 1
    hi = lo + (n - 1) * a.step
    chunks.append((lo, hi, n))
    lo = hi + a.step
if a.descending:                      # 18 Nov: check the band above 1450 Hz without the ascending sweep's history
    chunks = [(hi, lo, n) for lo, hi, n in reversed(chunks)]
for c0, c1, n in chunks:
    tests.append({"type": "sweep", "start_hz": c0, "end_hz": c1, "steps": n, "step_duration_s": a.dwell,
                  "repeat": 1, "settle_ms": 0, "section": "sweep", "level_mv": a.v})
    n_steps += n
    if a.ref > 0:
        tests.append({"type": "constant", "frequency_hz": a.ref, "duration_s": a.ref_s, "repeat": 1,
                      "settle_ms": 0, "section": "ref", "level_mv": a.v})
tests.append({"type": "constant", "frequency_hz": 0, "duration_s": 20.0, "repeat": 1, "settle_ms": 0})

if a.supply == "manual":
    supply = {"mode": "manual", "current_limit_ma": 60, "max_mv": 2500, "initial_mv": a.v, "lead_min": 2}
    stxt = f"MANUAL bench supply set by hand to {a.v / 1000:.2f} V (switch it off by hand after the run)"
else:
    supply = {"mode": "rsd3305p", "channel": 1, "current_limit_ma": 60, "max_mv": 2500, "initial_mv": a.v,
              "tolerance_mv": 50, "step_timeout_s": 10, "off_on_exit": True, "eol": "LF", "address": a.address}
    stxt = f"{a.v / 1000:.2f} V by the RSD3305P ({a.address})"
dur = sum((t.get("duration_s") or 0) + (t.get("steps", 0) * t.get("step_duration_s", 0)) for t in tests)
desc = (f"Fine free-flow sweep {a.fmax if a.descending else a.fmin}-{a.fmin if a.descending else a.fmax} Hz ({'descending' if a.descending else 'ascending'}) in {a.step} Hz steps ({n_steps} steps x {a.dwell:g} s, no gaps), "
        f"{a.ref} Hz reference {a.ref_s:g} s after every {a.chunk} Hz, flow check first; {stxt}; PPK2 on. "
        f"Outlet VENTED, no syringe. Analysis: last 60 % of each step. Run before and after priming.")
cfg = {
    "port": "COM3", "baud": 115200, "read_timeout_s": 0.2, "start_time": None, "end_time": None,
    "break_seconds_between_tests": 0.0, "send_zero_on_exit": True, "require_pump_id": True,
    "meta": {"name": "Fine flow sweep", "description": desc, "operator": "", "pump_id": a.pump,
             "fmin": a.fmin, "fmax": a.fmax, "step_hz": a.step, "dwell_s": a.dwell, "ref_hz": a.ref},
    "serial": {"protocol": "freq_only"}, "protocol": "freq_only",
    "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
    "supply": supply,
    "power": {"enabled": True, "backend": "ppk2", "port": "auto", "bin_ms": 10.0, "watch_ma": 57,
              "watch_hold_ms": 300, "raw_every_min": 10, "raw_length_s": 2.0, "post_run_tail_s": 10.0},
    "tests": tests,
}
out = a.out or f"config_flow_sweep_fine_{a.fmin}-{a.fmax}_{a.step}Hz_{a.supply}.json"
with open(out, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
print(f"wrote {out}: {len(tests)} blocks, {n_steps} sweep steps, about {dur / 60:.1f} min")
