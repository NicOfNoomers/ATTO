"""Low frequency against supply voltage, power and flow, for pump 261030-1 (NM, 1 Nov 2026).

Why (Paper/analysis_2026-11-01_pump_261030-1, analysis_2026-10-30_energy_map_260930-2):
- On the new pumps (260930-2, 261030-1) power is flat with frequency at 2.5 V (70-79 mW, 100-3000 Hz, no start-up
  surge), so a further 2.5 V energy map adds little: energy per uL is power / flow.
- Never measured on any new pump: anything below 100 Hz, and anything below 2.5 V. The see-through pump at 1 Hz
  drew 28-30 mA at 0.8 V and 35-38 mA at 1.0 V (~27 ohm, it never left its surge state), so current does not
  scale from the 2.5 V figures. The capsule puts 0.4-1 V on the coil.
- config_cell_supply_v6 measures current at 0.4-1.5 V at 1, 100 and 2000 Hz (15 s drives, no flow focus). This
  config measures FLOW and current at 1-100 Hz from 0.8 to 2.5 V, with 600 and 1300 Hz state references.

Run layout
- 60 s zero; conditioning at 2.5 V: 3 x [600 Hz 60 s, 20 s zero] (the 30 Oct purge run reached its plateau
  after the 600 Hz 60 s flushes).
- LEVELS 2.5, 2.0, 1.5, 1.0, 0.8 V, then 2.5 V again (drift over the run). Each level: a voltage step, 20 s zero,
  600 Hz reference, the low frequencies, 1300 Hz, 600 Hz reference again.
- Voltage steps are made by the RS PRO RSD3305P programmable supply (supply.mode "rsd3305p", supply_rsd.py): pump at
  0 Hz, the supply is set and the run goes on once its output reads the new voltage (within 50 mV, at most 10 s),
  then 5 s settle. No operator, no pause. The measured voltage is the supply's own reading: voltage_changes.csv
  (confirmed_mv, set_by "supply"), every second in supply/supply.csv, and it is the voltage the PPK2 bins use for
  power. It is read at the supply terminals, upstream of the PPK2 ammeter and the leads, so VM at the driver is
  lower by I x (PPK2 shunt + leads): check it once with the multimeter at 0.8 V.
- Low frequencies 1, 2, 5 Hz for 60 s (a 15 nL stroke at 1 Hz is ~1 uL/min, near the flow sensor's offset, so
  long dwells), 10, 20, 50, 100 Hz for 20 s; 20 s zero after every drive (offset reference before and after).
  The order alternates up / down between levels so frequency and time into the run are not confounded.
- Lowest level 0.8 V: the PPK2 ampere meter is rated for 0.8-5.0 V (config_cell_supply_v6 covers 0.4-0.6 V).
- After the last block the pump is at 0 Hz and everything records a 10 s tail, then the PPK2, supply (output
  off), flow and video stop by themselves.

Supply and power: RSD3305P CH1 -> PPK2 VIN, PPK2 VOUT -> driver VM, grounds together; nothing else on CH2/CH3
(the supply's output command switches all channels). Limit 60 mA (sent to the supply); the supply is read every
second and two readings in current limiting stop the run; PPK2 software trip at 57 mA held 300 ms (261030-1 draws
~29 mA at 2.5 V, no surge). Power on, 10 ms bins, raw snapshots every 10 min.

Supply address: the COM port of the supply on the PC that runs the dashboard (USB 0416:5011, "USB Serial Device"),
or <pc>:5025 when the supply hangs on another PC running supply_bridge.py. Without --address the config carries
none and the dashboard uses the Power panel's Address field (it refuses to start while that is empty).

usage: python make_config_lowf_voltage.py [pump_id] [--address COMx | host:5025] [--channel 1]
                                          [--out-on CMD --out-off CMD]    (what supply_rsd.py --probe-output found)
           -> config_lowf_voltage.json
       python make_config_lowf_voltage.py [pump_id] --manual
           -> config_lowf_voltage_manual.json (bench supply set by hand, the operator types the measured VM)
"""
import argparse
import json

ap = argparse.ArgumentParser(description="Low frequency vs voltage config (pump 261030-1)")
ap.add_argument("pump_id", nargs="?", default="261030-1")
ap.add_argument("--address", default=None, help="supply address: COMx, or host:5025 (supply_bridge.py)")
ap.add_argument("--channel", type=int, default=1, help="supply channel (1 = CH1)")
ap.add_argument("--out-on", default=None, help="output-on command if not OUT1 (supply_rsd.py --probe-output)")
ap.add_argument("--out-off", default=None, help="output-off command if not OUT0")
ap.add_argument("--manual", action="store_true", help="manual bench supply variant (operator voltage steps)")
args = ap.parse_args()

PUMP_ID = args.pump_id
PORT = "COM3"
LEVELS_MV = [2500, 2000, 1500, 1000, 800, 2500]
LOW_F = [1, 2, 5, 10, 20, 50, 100]
LONG_S, SHORT_S, LONG_BELOW_HZ = 60.0, 20.0, 5        # dwell: 60 s at <= 5 Hz, 20 s above
ZERO_S = 20.0
REF_HZ, REF2_HZ, REF_S = 600, 1300, 20.0
COND = (600, 60.0, 3)                                 # conditioning: f, s, repeats (at the first level)
LIMIT_MA, TRIP_MA, HOLD_MS = 60, 57, 300
STEP_SETTLE_S = 5
TAIL_S = 10.0
# rough flows at 2.5 V for the liquid estimate only (30 Oct purge run, 261030-1): nL per cycle below 600 Hz
Q_HZ = {600: 165.0, 1300: 150.0}
NL_PER_CYCLE = 15.0


def zero(s):
    return {"type": "constant", "frequency_hz": 0, "duration_s": float(s), "repeat": 1, "settle_ms": 0}


def drive(hz, s, **meta):
    return {"type": "constant", "frequency_hz": int(hz), "duration_s": float(s), "repeat": 1, "settle_ms": 0, **meta}


def voltage_step(lvl, mv):
    if args.manual:
        note = (f"Level {lvl + 1}/{len(LEVELS_MV)}: set the bench supply to {mv / 1000:.2f} V "
                f"(limit {LIMIT_MA} mA). Measure VM at the driver with the multimeter and type the "
                f"MEASURED value, then confirm.")
    else:
        note = f"Level {lvl + 1}/{len(LEVELS_MV)}: {mv / 1000:.2f} V, set by the programmable supply"
    return {"type": "voltage", "voltage_mv": mv, "while_waiting": "zero", "settle_s": STEP_SETTLE_S, "note": note}


tests = [zero(60.0)]
for lvl, mv in enumerate(LEVELS_MV):
    tests.append(voltage_step(lvl, mv))
    tests.append(zero(ZERO_S))
    if lvl == 0:
        for r in range(COND[2]):
            tests += [drive(COND[0], COND[1], section="conditioning", level_mv=mv, rep=r + 1), zero(ZERO_S)]
    tests += [drive(REF_HZ, REF_S, section="ref_start", level_mv=mv), zero(ZERO_S)]
    order = LOW_F if lvl % 2 == 0 else LOW_F[::-1]
    for f in order:
        tests += [drive(f, LONG_S if f <= LONG_BELOW_HZ else SHORT_S, section="low_f", level_mv=mv), zero(ZERO_S)]
    tests += [drive(REF2_HZ, REF_S, section="ref_1300", level_mv=mv), zero(ZERO_S)]
    tests += [drive(REF_HZ, REF_S, section="ref_end", level_mv=mv), zero(ZERO_S)]
tests.append(zero(60.0))

if args.manual:
    supply = {"mode": "manual", "current_limit_ma": LIMIT_MA, "max_mv": max(LEVELS_MV),
              "initial_mv": LEVELS_MV[0], "lead_min": 2}
    steps_txt = "Type the MEASURED VM at every voltage step."
    out_name = "config_lowf_voltage_manual.json"
else:
    supply = {"mode": "rsd3305p", "channel": args.channel, "current_limit_ma": LIMIT_MA,
              "max_mv": max(LEVELS_MV), "initial_mv": LEVELS_MV[0],
              "tolerance_mv": 50, "step_timeout_s": 10, "off_on_exit": True, "eol": "LF"}
    if args.address:
        supply["address"] = args.address
    if args.out_on:
        supply["output_on"] = args.out_on
    if args.out_off:
        supply["output_off"] = args.out_off
    steps_txt = ("Voltage steps by the RSD3305P programmable supply (no operator); measured voltage = the "
                 "supply's reading (supply/supply.csv, voltage_changes.csv).")
    out_name = "config_lowf_voltage.json"

cfg = {
    "port": PORT, "baud": 115200, "read_timeout_s": 0.2,
    "start_time": None, "end_time": None,
    "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
    "require_pump_id": True,
    "tests": tests,
    "meta": {
        "name": "Low frequency vs voltage" + (" (manual supply)" if args.manual else ""),
        "description": ("MEASURE POWER ON. Flow and supply current at 1, 2, 5 Hz (60 s) and 10, 20, 50, 100 Hz "
                        "(20 s) at 2.5, 2.0, 1.5, 1.0, 0.8 V and 2.5 V again, 600 and 1300 Hz references per "
                        "level, 3 x 600 Hz 60 s conditioning at the start. " + steps_txt +
                        f" Supply limit {LIMIT_MA} mA."),
        "operator": "", "pump_id": PUMP_ID,
        "levels_mv": LEVELS_MV, "low_f_hz": LOW_F,
        "analysis": "Paper/analysis_2026-11-01_pump_261030-1 (q1-style extraction, zero-referenced dwell means)",
    },
    "serial": {"protocol": "freq_only"},
    "protocol": "freq_only",
    "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
    "supply": supply,
    "power": {"enabled": True, "backend": "ppk2", "port": "auto", "bin_ms": 10.0,
              "watch_ma": TRIP_MA, "watch_hold_ms": HOLD_MS, "raw_every_min": 10, "raw_length_s": 2.0,
              "post_run_tail_s": TAIL_S},
    "video": {"codec": "mjpg", "quality": 95, "overlay": True, "chunk_s": 60.0, "crop": None},
}
with open(out_name, "w") as fh:
    json.dump(cfg, fh, indent=2)

dur = sum(x.get("duration_s", 0) + x.get("settle_s", 0) for x in tests if x["type"] in ("constant", "voltage"))
vol = 0.0
for x in tests:
    if x["type"] == "constant" and x["frequency_hz"] > 0:
        f = x["frequency_hz"]
        q = Q_HZ.get(f, NL_PER_CYCLE * f * 60 / 1000)      # uL/min
        vol += q * x["duration_s"] / 60
n_drive = sum(1 for x in tests if x["type"] == "constant" and x["frequency_hz"] > 0)
if args.manual:
    extra = f" + {len(LEVELS_MV)} voltage confirmations"
else:
    extra = (f" + {TAIL_S:g} s tail + {len(LEVELS_MV)} automatic steps (~1-2 s each); supply address "
             + (args.address if args.address else "NOT SET (Power panel Address field)"))
print(f"{out_name}: pump {PUMP_ID}, {len(tests)} blocks ({n_drive} drives), "
      f"{dur / 60:.1f} min{extra}; liquid <= {vol / 1000:.1f} mL (2.5 V flows)")
