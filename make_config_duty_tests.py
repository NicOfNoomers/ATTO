"""Carrier-PWM (duty-cycle) drive tests for the capsule (capsule-driver note 12, NM 28 Nov 2026).

Needs the duty firmware on the pump driver (firmware/pump_driver_duty_nrf52) and protocol 'freq_amp': blocks carry
carrier_duty (coil amplitude +-D x VM) and carrier_hz; the runner refuses carrier blocks under any other protocol.

  --test A   equivalence: reference arm D = 1 at VM 0.6/0.8/1.0/1.2/1.5 V (RSD3305P voltage steps) against the duty
             arm VM = --v (2.5 V) at D = 0.24/0.32/0.40/0.48/0.60 (same D x VM), carrier 50 kHz, plus the 1.0 V
             point at 125 kHz; at 20 Hz, 600 Hz and --fbest; 30 s drive / 30 s rest per block.
             --no-reference: duty arm only (fixed supply).
  --test B   capsule dose pattern: 10 x [10 s on / 60 s off] at --fbest, D x VM = 1.0 V then 0.8 V, ramp --ramp-s.
  --test C   lowest working drive: VM = --v, D 0.40 -> 0.16 in 0.04 steps at --fbest, 30 s on / 30 s off.
Analysis: scripts/duty_blocks.py <run> (per block: VM, f, D, carrier, flow, current, power, uL/J, uL per dose).

usage: python make_config_duty_tests.py --test A|B|C --pump <id> [--fbest 800] [--v 2500]
       [--supply rsd3305p|manual] [--address COM7] [--ramp-s 1.5] [--no-reference] [--out <file>]
"""
import argparse
import json

ap = argparse.ArgumentParser(description="Carrier-PWM drive tests")
ap.add_argument("--test", required=True, choices=("A", "B", "C"))
ap.add_argument("--pump", required=True)
ap.add_argument("--fbest", type=int, default=800, help="the pump's best frequency from a recent scan")
ap.add_argument("--v", type=int, default=2500, help="VM for the duty arm (mV, <= 2500)")
ap.add_argument("--supply", choices=("rsd3305p", "manual"), default="rsd3305p")
ap.add_argument("--address", default="COM7")
ap.add_argument("--ramp-s", type=float, default=1.5, help="driver amplitude ramp at start/stop (test B)")
ap.add_argument("--no-reference", action="store_true", help="test A without the D = 1 voltage arm")
ap.add_argument("--ilim", type=int, default=60)
ap.add_argument("--out", default=None)
a = ap.parse_args()
if not 0 < a.v <= 2500:
    raise SystemExit("--v: 1..2500 mV")
if a.test == "A" and not a.no_reference and a.supply != "rsd3305p":
    raise SystemExit("test A's reference arm steps VM: needs --supply rsd3305p (or --no-reference)")


def zero(s):
    return {"type": "constant", "frequency_hz": 0, "duration_s": float(s), "repeat": 1, "settle_ms": 0}


def drive(hz, s, mv, d=1.0, carrier=50000, **meta):
    return {"type": "constant", "frequency_hz": int(hz), "duration_s": float(s), "repeat": 1, "settle_ms": 0,
            "level_mv": mv, "carrier_duty": round(d, 4), "carrier_hz": int(carrier), **meta}


def volt(mv):
    return {"type": "voltage", "voltage_mv": int(mv), "while_waiting": "zero", "settle_s": 5,
            "note": f"VM {mv / 1000:.2f} V by the supply"}


tests = [zero(10.0)]
if a.supply == "rsd3305p":
    tests.append(volt(a.v))
VEFF = [600, 800, 1000, 1200, 1500]
if a.test == "A":
    for f in (20, 600, a.fbest):
        if not a.no_reference:
            for mv in VEFF:
                tests += [volt(mv), drive(f, 30, mv, 1.0, section="A_reference", f_group=f, v_eff_mv=mv), zero(30)]
            tests.append(volt(a.v))
        for mv in VEFF:
            d = mv / a.v
            tests += [drive(f, 30, a.v, d, section="A_duty", f_group=f, v_eff_mv=mv), zero(30)]
        tests += [drive(f, 30, a.v, 1000 / a.v, 125000, section="A_duty_125k", f_group=f, v_eff_mv=1000), zero(30)]
    name = "Duty test A (equivalence)"
elif a.test == "B":
    for mv in (1000, 800):
        for k in range(1, 11):
            tests += [drive(a.fbest, 10, a.v, mv / a.v, section="B_dose", rep=k, v_eff_mv=mv), zero(60)]
    name = "Duty test B (capsule dose pattern)"
else:
    d = 0.40
    while d >= 0.159:
        tests += [drive(a.fbest, 30, a.v, d, section="C_lowest", v_eff_mv=round(d * a.v)), zero(30)]
        d = round(d - 0.04, 4)
    name = "Duty test C (lowest working duty)"
if a.supply == "manual":
    supply = {"mode": "manual", "current_limit_ma": a.ilim, "max_mv": 2500, "initial_mv": a.v, "lead_min": 2}
    tests = [t for t in tests if t["type"] != "voltage"]
else:
    supply = {"mode": "rsd3305p", "channel": 1, "current_limit_ma": a.ilim, "max_mv": 2500, "initial_mv": a.v,
              "tolerance_mv": 50, "step_timeout_s": 10, "off_on_exit": True, "eol": "LF", "address": a.address}
cfg = {
    "port": "COM3", "baud": 115200, "read_timeout_s": 0.2, "start_time": None, "end_time": None,
    "break_seconds_between_tests": 0.0, "send_zero_on_exit": True, "require_pump_id": True,
    "meta": {"name": name, "operator": "", "pump_id": a.pump,
             "description": (f"{name}: carrier-PWM coil amplitude (firmware pump_driver_duty, protocol freq_amp); VM "
                             f"{a.v / 1000:.2f} V for the duty blocks, best frequency {a.fbest} Hz; supply {a.supply}; "
                             f"PPK2 on (battery-side current ~ D x coil current). Capsule-driver note 12.")},
    "serial": {"protocol": "freq_amp"}, "protocol": "freq_amp",
    "driver": {"carrier_duty": 1.0, "carrier_hz": 50000, "ramp_s": a.ramp_s if a.test == "B" else 0.0},
    "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
    "supply": supply,
    "power": {"enabled": True, "backend": "ppk2", "port": "auto", "bin_ms": 10.0,
              "watch_ma": a.ilim - 3 if a.ilim <= 60 else a.ilim, "watch_hold_ms": 300, "raw_every_min": 10,
              "raw_length_s": 2.0, "post_run_tail_s": 10.0},
    "tests": tests,
}
dur = sum(t.get("duration_s", 0) for t in tests if t["type"] == "constant") + 10 * sum(1 for t in tests if t["type"] == "voltage")
out = a.out or f"config_duty_{a.test}_{a.pump}{'_manual' if a.supply == 'manual' else ''}.json"
with open(out, "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, indent=2)
print(f"wrote {out}: {len(tests)} blocks, about {dur / 60:.0f} min")
