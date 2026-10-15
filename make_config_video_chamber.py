"""Video run for a camera zoomed onto the chamber: known flow events to film.

The 14-15 Oct video analysis (Paper/analysis_2026-10-15_bubble-video) found the
chamber image stable to < 0.1 px and < 0.12 % dark-body area while flow drifted
4-7 % -- but the camera was 640x480, front-lit and focused on the receiver. This
run is for a zoomed view of the chamber and gives the video events with known
flow behaviour, so there is something definite to look for.

SECTIONS
  A  still reference   90 s pump off (start and end): reference image for masks
  S  sync marker       6 x 0.2 s pulses at 600 Hz with irregular gaps (2, 5, 3, 7,
                       4, 2.5 s); three of them. Irregular spacing gives the data-
                       driven sync (w1_sync.py) unambiguous anchors.
  B  burst             state reference (start and end)
  L  slow strokes      1 Hz 30 s, 2 Hz 30 s, 5 Hz 20 s. At ~17 frames/s each stroke
                       spans 17 / 8 / 3 frames: whatever moves per stroke is filmed
                       (a trapped bubble would pulsate with the stroke).
  D  positive control  8 x [600 Hz 12 s, 25 s rest], 4 x [100 Hz 12 s, 25 s rest].
                       The ~1.5 s flow drop is now known to be electrical (PPK2, 15 Oct:
                       ~90 mA for the first ~1.45 s of every drive, then ~38 mA; flow
                       falls 1.49x at the same instant). A 33 % flow change with a known
                       cause at a known time: if the zoomed video can see the chamber /
                       diaphragm motion change there, it can resolve stroke changes; if
                       not, a 4-7 % drift is out of its reach.
  F  frequency survey  300, 600, 900, 1100, 1300, 1600, 2000 Hz, 10 s each with
                       15 s rests, up then down: stable band, the unstable 900-1200
                       Hz band (CV 65-95 %) and the 1600 Hz peak (CV 47 %)
  H  slow drift        600 Hz, 0.1 s pulse every 5 s, 3 x 5 min segments with 20 s
                       rests (pulses >= 4 s apart keep the sync to ~0.2 s). With the
                       power meter on, each pulse also has its own charge/energy (10 ms
                       bins), so the run tests whether the 4-7 % flow drift follows the
                       drive energy -- video and power on the same pulses.

Camera: zoom on the chamber, lock exposure and focus if the camera allows it,
do not touch it during the run. Tick "Measure power" in the PPK2 panel (it is a
dashboard setting, not a config field) so the same run logs energy.
"""
import json

PUMP_ID = ""                       # fill in
BURST_S = 72.0
SYNC_GAPS = [2.0, 5.0, 3.0, 7.0, 4.0, 2.5]

zero = lambda s: {"type": "constant", "frequency_hz": 0, "duration_s": float(s),
                  "repeat": 1, "settle_ms": 0}
sp = lambda hz, s: {"type": "constant", "frequency_hz": int(hz), "duration_s": float(s),
                    "repeat": 1, "settle_ms": 0}
burst = {"type": "sweep", "start_hz": 1, "end_hz": 1000, "steps": 18,
         "total_duration_s": 1.0, "repeat": 60, "settle_ms": 0}


def sync_marker():
    out = []
    for gap in SYNC_GAPS:
        out += [sp(600, 0.2), zero(gap)]
    return out


sections = []
tests = []


def add(name, blocks):
    sections.append((name, len(tests), blocks))
    tests.extend(blocks)


add("A  still reference", [zero(90.0)])
add("S1 sync marker", sync_marker() + [zero(10.0)])
add("B1 burst", [dict(burst), zero(30.0)])
add("L  slow strokes", [sp(1, 30.0), zero(20.0), sp(2, 30.0), zero(20.0), sp(5, 20.0), zero(30.0)])
add("D  drop, 600 Hz x8", sum(([sp(600, 12.0), zero(25.0)] for _ in range(8)), []))
add("D  drop, 100 Hz x4", sum(([sp(100, 12.0), zero(25.0)] for _ in range(4)), []))
fs = [300, 600, 900, 1100, 1300, 1600, 2000]
add("F  frequency survey up", sum(([sp(f, 10.0), zero(15.0)] for f in fs), []))
add("F  frequency survey down", sum(([sp(f, 10.0), zero(15.0)] for f in fs[::-1]), []))
add("S2 sync marker", sync_marker() + [zero(20.0)])
seg = [{"type": "group", "repeat": 60, "blocks": [sp(600, 0.1), zero(4.9)]}]
add("H  slow drift, 600 Hz every 5 s", seg + [zero(20.0)] + seg + [zero(20.0)] + seg + [zero(30.0)])
add("B2 burst", [dict(burst), zero(30.0)])
add("S3 sync marker", sync_marker() + [zero(10.0)])
add("A  still reference", [zero(90.0)])

cfg = {"port": "COM3", "baud": 115200, "read_timeout_s": 0.2,
       "start_time": None, "end_time": None,
       "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
       "tests": tests,
       "meta": {"name": "Video chamber",
                "description": ("Camera zoomed on the chamber. Still references, irregular sync "
                                "markers, slow strokes (1/2/5 Hz), repeated 600 and 100 Hz blocks "
                                "for the ~1.75 s drop, frequency survey 300-2000 Hz up and down, "
                                "15 min of 600 Hz pulses every 5 s, bursts at start and end."),
                "operator": "", "pump_number": PUMP_ID},
       "serial": {"protocol": "freq_only"},
       "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
       "protocol": "freq_only"}
json.dump(cfg, open("config_video_chamber.json", "w"), indent=2)


def dur(x):
    if x["type"] == "constant":
        return x["duration_s"]
    if x["type"] == "sweep":
        return BURST_S
    return x["repeat"] * sum(dur(b) for b in x["blocks"])


t = 0.0
print("timeline (approximate; the rig's command timing adds a little):")
for name, _, blocks in sections:
    d = sum(dur(b) for b in blocks)
    print(f"  {t/60:5.1f} min  {name:34s} {d/60:4.1f} min")
    t += d
print(f"config_video_chamber.json: {len(tests)} blocks, {t/60:.1f} min")
