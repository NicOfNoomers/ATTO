"""Can 2 x SR516 coin cells run the pump? Series-resistance supply test, one config per variant.

Brief: PhD/Pump/capsule_driver/03_bench_brief_cell_supply_2026-10-29.md (pump/capsule-driver
session, requested by NM, 29 Oct 2026).

The capsule runs on 2 x SR516 in series: 3.1 V open circuit, ~120 ohm stack, 12.5 mAh, low drain.
The pump draws a 90-95 mA start-up surge (1.4 s at 100 Hz, 4.4 s at 2 kHz) and then 37-64 mA at
2.5 V straight from a bench supply. The battery node must stay >= 1.8 V (DRV8837 UVLO, ANNA-B112),
and a >= 300 mV drop within 300 ms may reset the nRF.

Bench (NM):
    bench supply V_s (current limit on) -- R_series -- PPK2 (ampere meter) -- driver VM -- coil
    battery node = V_s - I x R_series (computed afterwards from the PPK2 current)

Each reading: rest REST_S at 0 Hz -> drive DRIVE_S at f -> recover RECOVER_S at 0 Hz, REPEATS per
frequency. Each config starts with an operator voltage step: set V_s, check the hardware for that
variant, type the measured V_s, confirm.

Safety: the software trip (PPK2) is 40 mA with the 120 ohm in place. Normal current is <= ~17 mA
(3.1 V / (120 + 65) ohm), so 40 mA also stops the run if the resistor was left shorted at 3.1 V.
The control (resistor shorted, 2.5 V) uses 150 mA held 300 ms, above the 90-95 mA surge.
max_mv in the R variants is V_s (3.1 / 2.6 V): that is the supply BEFORE the 120 ohm; the coil
sees at most ~1.1 V while driving.

Analysis: python scripts/analyze_cell_supply.py <run folder> [<run folder> ...]
Drive firmware: the XIAO nRF52840 drive (tone(), fixed 50 % square wave) has no duty control,
so the capsule-style duty sweep is not part of this test.
"""
import json
import re

FREQS_HZ = [1, 100, 2000]
REST_S, DRIVE_S, RECOVER_S, REPEATS = 20.0, 15.0, 30.0, 3
PORT = "COM3"
CELL_MAH = 12.5
NODE_MIN_V = 1.8

# key: (label, V_s mV, series ohm, extra part, frequencies, bench limit mA, trip mA, trip hold ms)
VARIANTS = {
    "v1": ("Cells, fresh", 3100, 120, "none", FREQS_HZ, 50, 40, 200),
    "v2": ("Cells, fresh, bulk cap", 3100, 120, "22 uF at VM (board C23)", FREQS_HZ, 50, 40, 200),
    "v3": ("Cells, half used", 2600, 120, "none", FREQS_HZ, 50, 40, 200),
    "v4": ("Cells + BLE load", 3100, 120, "470 ohm VM to GND (~5 mA, BLE stand-in)", [100], 50, 40, 200),
    "v5": ("Control, resistor shorted", 2500, 0, "none", [100, 2000], 150, 150, 300),
}


def zero(s):
    return {"type": "constant", "frequency_hz": 0, "duration_s": float(s), "repeat": 1, "settle_ms": 0}


def make(key, label, vs_mv, r_ohm, extra, freqs, limit_ma, trip_ma, hold_ms):
    setup = (f"R_series {r_ohm} ohm" if r_ohm else "120 ohm SHORTED") + f", extra part: {extra}"
    tests = [{"type": "voltage", "voltage_mv": vs_mv, "while_waiting": "zero", "settle_s": 5,
              "note": f"Variant {key[1:]} ({label}): bench supply {vs_mv / 1000:.2f} V, limit {limit_ma} mA; "
                      f"{setup}. Type the MEASURED V_s, then confirm."}]
    n = 0
    for f in freqs:
        for rep in range(1, REPEATS + 1):
            n += 1
            tests += [zero(REST_S),
                      {"type": "constant", "frequency_hz": int(f), "duration_s": float(DRIVE_S), "repeat": 1,
                       "settle_ms": 0, "reading": n, "rep": rep, "variant": key},
                      zero(RECOVER_S)]
    cfg = {
        "port": PORT, "baud": 115200, "read_timeout_s": 0.2,
        "start_time": None, "end_time": None,
        "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
        "require_pump_id": True,
        "tests": tests,
        "meta": {
            "name": f"Cell supply {key} {label}",
            "description": (f"2 x SR516 supply test, variant {key[1:]}: {label}. V_s {vs_mv / 1000:.2f} V, "
                            f"{setup}. Readings: {REST_S:.0f} s rest, {DRIVE_S:.0f} s drive, {RECOVER_S:.0f} s "
                            f"recover, x{REPEATS} at {', '.join(str(f) for f in freqs)} Hz. PPK2 ampere meter at "
                            f"1 ms bins after the series resistor; battery node = V_s - I x R_series."),
            "operator": "", "pump_id": "",
            "variant": key, "variant_label": label,
            "v_supply_mv": vs_mv, "series_ohm": r_ohm, "extra_part": extra,
            "cell_capacity_mah": CELL_MAH, "node_min_v": NODE_MIN_V,
            "brief": "PhD/Pump/capsule_driver/03_bench_brief_cell_supply_2026-10-29.md",
        },
        "serial": {"protocol": "freq_only"},
        "protocol": "freq_only",
        "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
        "supply": {"mode": "manual", "current_limit_ma": limit_ma, "max_mv": vs_mv, "initial_mv": vs_mv,
                   "lead_min": 2},
        "power": {"enabled": True, "backend": "ppk2", "port": "auto", "bin_ms": 1,
                  "watch_ma": trip_ma, "watch_hold_ms": hold_ms, "raw_every_min": 2, "raw_length_s": 2.0},
    }
    dur = 5 + len(freqs) * REPEATS * (REST_S + DRIVE_S + RECOVER_S)
    return cfg, dur


total = 0.0
print(f"{'config':<42} {'V_s':>5} {'R':>5}  {'freqs':<14} {'trip':>11} {'min':>5}")
for key, v in VARIANTS.items():
    cfg, dur = make(key, *v)
    slug = re.sub(r"[^a-z0-9]+", "_", v[0].lower()).strip("_")
    name = f"config_cell_supply_{key}_{slug}.json"
    with open(name, "w") as f:
        json.dump(cfg, f, indent=2)
    total += dur
    print(f"{name:<42} {v[1] / 1000:>4.2f}V {v[2]:>4}R  {','.join(map(str, v[4])):<14} "
          f"{v[6]:>4} mA/{v[7]} ms {dur / 60:>5.1f}")
print(f"total {total / 60:.0f} min + one voltage confirmation per variant")
