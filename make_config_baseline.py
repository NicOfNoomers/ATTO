"""Baseline state check for a pump coming back on the rig: does it match its own history? (NM via atto_test_configs,
5 Nov 2026). No priming, no low-frequency full strokes (prime_plus's 2 Hz strokes hurt 261030-1's high frequencies
on 2 Nov and 260930-2's on 5 Nov).

Blocks: the config_scan_quick.json block list as is (2.5 V step, 100-1600 Hz, 10 s each, 5 s zeros), then the
reference blocks 20 Hz 20 s, 100 Hz 20 s, 600 Hz 60 s, 1300 Hz 20 s with 5 s zeros between. Compare with the pump's
own history (HISTORY below, written into the description). RSD3305P COM7 at 2.5 V, PPK2 on (trip 57 mA / 300 ms),
outlet VENTED, no syringe, Pump ID required.

usage: python make_config_baseline.py --pump 261030-1 [--address COM7]   -> config_baseline_<pump>.json
Runner (outlet video): py -3.10 scripts\\run_headless.py config_baseline_261030-1.json --name baseline_261030-1
                           --pump-id 261030-1 --video --camera 1 --show
"""
import argparse
import copy
import json

HISTORY = {
    "261030-1": ("261030-1 at 2.5 V, free flow: 600 Hz 218-228 uL/min fresh (2 Nov 04:25), 177-197 after strong "
                 "priming, 182-185 at its last run (2 Nov 07:12 prime_plus); 1300 Hz 281 fresh, 77-79 last; 1000 Hz "
                 "48 last; 20 Hz 112-143; 100 Hz 112-126; current 27-29 mA."),
    "260930-2": ("260930-2 at 2.5 V, free flow: 600-700 Hz is its dip (68-85 uL/min, 30 Oct); peak band 1000 Hz "
                 "282-352, 1100 Hz 356-421, 1300 Hz 351-425, 1600 Hz 214-281 (30 Oct energy map, metal adapter)."),
}
REF = [(20, 20.0), (100, 20.0), (600, 60.0), (1300, 20.0)]
REF_ZERO_S = 5.0

ap = argparse.ArgumentParser(description="Baseline state check (scan + reference blocks), any pump")
ap.add_argument("--pump", required=True, help="pump ID on the rig (ask Nic; never from a config name)")
ap.add_argument("--address", default="COM7", help="RSD3305P address (default COM7, the lab PC); '' = Power panel")
args = ap.parse_args()

scan = json.load(open("config_scan_quick.json", encoding="utf-8"))
tests = copy.deepcopy(scan["tests"])
level = next((t["voltage_mv"] for t in tests if t["type"] == "voltage"), 2500)
for f, s in REF:
    tests += [{"type": "constant", "frequency_hz": f, "duration_s": s, "repeat": 1, "settle_ms": 0,
               "section": "reference", "level_mv": level},
              {"type": "constant", "frequency_hz": 0, "duration_s": REF_ZERO_S, "repeat": 1, "settle_ms": 0}]

cfg = copy.deepcopy(scan)
cfg["tests"] = tests
if args.address:
    cfg["supply"]["address"] = args.address
else:
    cfg["supply"].pop("address", None)
hist = HISTORY.get(args.pump, "No stored history for this pump: compare with its last runs.")
cfg["meta"] = {"name": f"Baseline {args.pump}", "operator": "", "pump_id": args.pump,
               "freqs_hz": scan["meta"].get("freqs_hz"), "reference_blocks": [f"{f} Hz {s:g} s" for f, s in REF],
               "history": hist,
               "description": (f"BASELINE for pump {args.pump}: outlet VENTED, no syringe, no priming. "
                               f"Quick scan 100-1600 Hz (10 s each, 5 s zeros), then 20 Hz 20 s, 100 Hz 20 s, "
                               f"600 Hz 60 s, 1300 Hz 20 s with 5 s zeros, at {level / 1000:.2f} V (RSD3305P), PPK2 on. "
                               f"No 2 Hz / low-f full strokes. Compare with its history: {hist}")}
name = f"config_baseline_{args.pump}.json"
with open(name, "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, indent=2)
total = sum(t.get("duration_s", 0) for t in tests if t["type"] == "constant") + 5
print(f"{name}: {len(tests)} blocks, {total / 60:.1f} min + 10 s tail; pump {args.pump}; supply {args.address or 'Power panel'}")
