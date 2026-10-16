"""Purge attempt, before/after: full performance panel, removal attempt, same panel again.

Pump tilted (outlet up) for the whole run, freshly primed. Built on the tilt run of 16 Oct
(Paper/analysis_2026-10-16_tiltpage):
  * 600 Hz for 10 s merged the foam at the back end into one pocket
  * 100 Hz pulses (0.1 s every 2 s) then cleared the back end within 1-5 min
  * every 2000 Hz block, the sweep and the 1-1000 Hz burst pulled hidden gas back into view
    (22-27 % of the back-end region): the gas had moved, not left
The PANEL measures performance; it runs identically before and after the removal attempt.
The removal attempt uses purge cycles, with 2000 Hz CHALLENGE blocks that show how much
hidden gas is left.

PERFORMANCE PANEL (16.4 min, low-disturbance measures first, high frequencies last)
  still 30 s
  delivery   100 Hz pulses 0.1 s / 2 s x 150 (5 min): volume, pulse-to-pulse CV, drift
  pulses     600 Hz pulses 0.1 s / 2 s x 30
  steady     100 Hz 20 s | 600 Hz 60 s (decline within a block)
  response   10 s on / 10 s off, ascending: 100 200 300 400 600 800 1000 1300 1600 1800 2000
             2200 2600 3000 Hz (flow 3-10 s, power, start-up phase per frequency)
  pulses     2000 Hz pulses 0.1 s / 2 s x 30
  re-check   600 Hz 10 s (state after the high frequencies)
  burst      1-1000 Hz, the state reference used in every run
  still 30 s
REMOVAL ATTEMPT (57 min)
  purge cycle (4.8 min): 600 Hz 10 s (merge) | zero 10 | 100 Hz pulses 3 min (clear the back
    end; tilted, gas should drift to the high end) | zero 10 | 600 Hz 60 s (carry it out) | zero 20
  challenge (4.2 min): reference (600 Hz 10 s, 10 x 100 Hz pulses) | still | 2000 Hz 30 s |
    still | 600 Hz 10 s | 100 Hz pulses 2 min | still
  3 x [3 purge cycles + challenge], then 5 min of 100 Hz pulses to settle
RUN: still, sync marker, PANEL (before), REMOVAL ATTEMPT, PANEL (after), sync marker, still

DECIDES IT
  removal    hidden gas pulled into view by challenges 1-3 falls step by step, and the
             high-frequency part and burst of the AFTER panel bring back less than in BEFORE
  better     AFTER vs BEFORE: pulse-to-pulse CV and drift (delivery), pulse volumes,
             frequency response, burst peak
Same panel twice, same tilt, same camera: before and after differ only by the attempt (and
the ~75 min between them; the tilt run drifted -5.5 %/h).

Before running: prime fresh with the outlet up (degassed water if available; note it), the
same tilt as the tilt run, with the angle and which end is up written down. Wide camera on
the whole pump AND the tube, locked; room lights fixed. PPK2 on; the 3000 Hz step draws
~100 mA. Pump ID below.
"""
import json

PUMP_ID = ""                       # fill in
BURST_S = 72.0
SYNC_GAPS = [2.0, 5.0, 3.0, 7.0, 4.0, 2.5]
RESPONSE = [100, 200, 300, 400, 600, 800, 1000, 1300, 1600, 1800, 2000, 2200, 2600, 3000]

zero = lambda s: {"type": "constant", "frequency_hz": 0, "duration_s": float(s),
                  "repeat": 1, "settle_ms": 0}
sp = lambda hz, s: {"type": "constant", "frequency_hz": int(hz), "duration_s": float(s),
                    "repeat": 1, "settle_ms": 0}
burst = {"type": "sweep", "start_hz": 1, "end_hz": 1000, "steps": 18,
         "total_duration_s": 1.0, "repeat": 60, "settle_ms": 0}


def group(hz, on, period, n):
    return {"type": "group", "repeat": n, "blocks": [sp(hz, on), zero(round(period - on, 2))]}


def sync_marker():
    out = []
    for gap in SYNC_GAPS:
        out += [sp(600, 0.2), zero(gap)]
    return out


def panel():
    return ([zero(30.0),
             group(100, 0.1, 2.0, 150), zero(10.0),
             group(600, 0.1, 2.0, 30), zero(10.0),
             sp(100, 20.0), zero(10.0), sp(600, 60.0), zero(10.0)]
            + sum(([sp(f, 10.0), zero(10.0)] for f in RESPONSE), [])
            + [group(2000, 0.1, 2.0, 30), zero(10.0),
               sp(600, 10.0), zero(10.0),
               dict(burst), zero(30.0)])


def ref():
    return [sp(600, 10.0), zero(10.0), group(100, 0.1, 2.0, 10), zero(10.0)]


def purge_cycle():
    return [sp(600, 10.0), zero(10.0), group(100, 0.1, 2.0, 90), zero(10.0), sp(600, 60.0), zero(20.0)]


def challenge():
    return ref() + [zero(10.0), sp(2000, 30.0), zero(10.0), sp(600, 10.0), zero(10.0),
                    group(100, 0.1, 2.0, 60), zero(10.0)]


sections, tests = [], []


def add(name, blocks):
    sections.append((name, blocks))
    tests.extend(blocks)


add("still", [zero(60.0)])
add("sync marker", sync_marker() + [zero(10.0)])
add("PANEL BEFORE", panel())
for b in range(1, 4):
    for c in range(1, 4):
        add(f"purge cycle {3 * (b - 1) + c}", purge_cycle())
    add(f"challenge {b}", challenge())
add("settle: 100 Hz pulses 5 min", [group(100, 0.1, 2.0, 150), zero(10.0)])
add("PANEL AFTER", panel())
add("sync marker", sync_marker() + [zero(10.0)])
add("still", [zero(60.0)])

cfg = {"port": "COM3", "baud": 115200, "read_timeout_s": 0.2,
       "start_time": None, "end_time": None,
       "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
       "tests": tests,
       "meta": {"name": "Purge attempt tilted before-after",
                "description": ("Fresh priming, pump tilted outlet-up throughout. Performance panel (100 Hz "
                                "delivery 5 min, 600 Hz pulses, 100/600 Hz steady, response 100-3000 Hz, "
                                "2000 Hz pulses, 600 Hz re-check, burst), then 3 x [3 purge cycles (600 Hz "
                                "10 s, 100 Hz pulses 3 min, 600 Hz 60 s) + 2000 Hz 30 s challenge], 5 min "
                                "settle, then the same panel again."),
                "operator": "", "pump_number": PUMP_ID},
       "serial": {"protocol": "freq_only"},
       "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
       "protocol": "freq_only"}
json.dump(cfg, open("config_purge_attempt.json", "w"), indent=2)


def dur(x):
    if x["type"] == "constant":
        return x["duration_s"]
    if x["type"] == "sweep":
        return BURST_S
    return x["repeat"] * sum(dur(b) for b in x["blocks"])


t = 0.0
print("timeline (mm:ss):")
for name, blocks in sections:
    d = sum(dur(b) for b in blocks)
    print(f"  {int(t // 60):3d}:{int(t % 60):02d}  {name:30s} {d/60:5.1f} min")
    t += d
print(f"config_purge_attempt.json: {len(tests)} blocks, {t/60:.1f} min")
