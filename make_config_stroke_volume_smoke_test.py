"""Smoke test for the stroke-volume branch on the test PC, before config A (NM via atto-work, 25 Oct 2026).

One operator stop (2.5 V), then:
    10 s rest -> 5 strokes at 1 Hz -> 10 s rest -> 25-stroke burst at 100 Hz -> 10 s rest
    -> 25-stroke burst at 700 Hz -> 10 s rest -> 10 s continuous at 400 Hz -> 10 s rest
Bursts last (N - 0.25)/f, as in the other stroke-volume configs; the 400 Hz block is 4000 strokes in the same way.
Nominal ~65 s plus the voltage stop; the camera connection and the checks make it the ~3 min session.

Afterwards, in the run folder (details: Paper/analysis_2026-10-22_stroke_volume/NOTES_2026-10-25_frequency_configs.md):
    python scripts\\check_video.py <run folder>     frames.csv vs video, frame rate, drops, gaps
    python scripts\\check_bursts.py <run folder>    commanded burst lengths from t_cmd_s, flags > 0.5 ms

Wiring as for the other stroke-volume configs: bench supply -> driver VM, PPK2 out of the path, power off.
Dashboard from this branch (run_dashboard.bat in this worktree): Pump ID, video on, overlay off, codec mjpg.
"""
import json

PORT = "COM3"
LEVEL_MV = 2500
REST_S = 10.0
DRIVES = [(1, 5), (100, 25), (700, 25), (400, 4000)]     # (Hz, strokes); 400 Hz x 4000 = 10 s continuous


def zero(s):
    return {"type": "constant", "frequency_hz": 0, "duration_s": float(s), "repeat": 1, "settle_ms": 0}


def drive(hz, n, k):
    return {"type": "constant", "frequency_hz": int(hz), "duration_s": (n - 0.25) / hz, "repeat": 1,
            "settle_ms": 0, "strokes": int(n), "edges_commanded": 2 * int(n), "reading": k}


tests = [{"type": "voltage", "voltage_mv": LEVEL_MV, "while_waiting": "zero", "settle_s": 0,
          "note": f"Smoke test: set the bench supply to {LEVEL_MV / 1000:.3f} V, MEASURE VM and type it "
                  f"(at most 2.500 V), meniscus in view, then confirm."},
         zero(REST_S)]
for k, (hz, n) in enumerate(DRIVES, 1):
    tests += [drive(hz, n, k), zero(REST_S)]

cfg = {
    "port": PORT, "baud": 115200, "read_timeout_s": 0.2,
    "start_time": None, "end_time": None,
    "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
    "require_pump_id": True,
    "tests": tests,
    "meta": {"name": "Stroke volume smoke test",
             "description": ("Check of the stroke-volume branch on the test PC: 2.5 V; 5 strokes at 1 Hz, 25-stroke "
                             "bursts at 100 and 700 Hz, 10 s at 400 Hz, 10 s rests. Verify frames.csv, t_cmd_s, "
                             "burst lengths, frame rate and the meniscus in the video."),
             "operator": "", "pump_id": "", "tube_id_mm": 0.5, "tube_od_mm": 1.5875, "line_mm": None},
    "serial": {"protocol": "freq_only"},
    "protocol": "freq_only",
    "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
    "supply": {"mode": "manual", "current_limit_ma": 60, "max_mv": 2500, "initial_mv": LEVEL_MV, "lead_min": 2},
    "power": {"enabled": False},
    "video": {"codec": "mjpg", "quality": 95, "overlay": False, "chunk_s": 60.0, "crop": None},
}

with open("config_stroke_volume_smoke_test.json", "w") as f:
    json.dump(cfg, f, indent=2)
nominal = sum(t["duration_s"] for t in tests if t["type"] == "constant")
print(f"config_stroke_volume_smoke_test.json: {len(DRIVES)} drives, nominal {nominal:.1f} s + 1 voltage stop")
