"""Offline check of the pinch-valve loop (pinch_control.py + the runner's 'pinch_pq' block), real time, no hardware.

A fake runner holds the real PinchController on a simulated loop (pinch_control.PinchPlant) driven through
pressure_control.DummySyringe with 16 microsteps; a thread feeds the flow buffer from the plant. Cases:
  1. pinch_pq at 1150 Hz: touch found, open point, 4 positions, 4 pressure targets, fully closed point;
  2. limit: limit_mbar 100 and a close to 300 -> the sampler faults, retracts and calls on_fault.
usage: python scripts/sim_pinch.py   (about 1-2 min; exit 1 on failure)
"""
import collections
import csv
import os
import sys
import tempfile
import threading
import time
import types

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, W)
import importlib.util as u  # noqa: E402

spec = u.spec_from_file_location("ptd", os.path.join(W, "pump_test_dashboard.py"))
m = u.module_from_spec(spec)
spec.loader.exec_module(m)
import pinch_control  # noqa: E402
import pressure_control as pc  # noqa: E402

FAST = {"touch_settle_s": 0.25, "hold_dt": 0.35, "hold_settle_s": 1.0, "start_retract_steps": 10}   # the retract is clamped by back_limit 0


def make_fake(limit=340.0, pinch=None):
    fake = types.SimpleNamespace()
    fake.freq = 0
    plant = pinch_control.PinchPlant(get_freq=lambda: fake.freq, x_touch=600, tau_s=0.3, q0=200.0, pmax=250.0)
    syr = pc.DummySyringe(plant, 0, 2470 * 16, "v1")
    faults = []
    run_dir = tempfile.mkdtemp()
    ctl = pinch_control.PinchController({"limit_mbar": limit, "min_mbar": -60, "pinch": {**FAST, **(pinch or {})}},
                                        syr, plant.read_sensor, plant.range, run_dir=run_dir,
                                        on_fault=faults.append, microsteps=16)
    ctl.open()
    ctl.zero_sensor(0.5)
    fake.t0 = time.perf_counter()
    ctl.start(fake.t0)
    fake.ctl, fake.plant, fake.faults, fake.run_dir = ctl, plant, faults, run_dir
    fake.lock = threading.Lock()
    fake.buf_t = collections.deque(maxlen=5000)
    fake.buf_flow = collections.deque(maxlen=5000)
    fake.stop_flag = threading.Event()
    fake._emit = lambda k, msg: print("   STATUS:", msg)
    fake._set_freq = lambda hz: setattr(fake, "freq", int(hz))
    fake._wait_until = lambda t_end: time.sleep(max(0.0, t_end - time.perf_counter()))
    fake._need_ctl = lambda: fake.ctl
    fake._need_pinch = types.MethodType(m.PumpTestRunner._need_pinch, fake)
    fake._window_flow_stats = types.MethodType(m.PumpTestRunner._window_flow_stats, fake)

    def feeder():
        while not fake.stop_flag.is_set():
            q = plant.pump_flow_ul_min(plant.pressure())
            with fake.lock:
                fake.buf_t.append(time.perf_counter() - fake.t0)
                fake.buf_flow.append(q)
            time.sleep(0.04)

    threading.Thread(target=feeder, daemon=True).start()
    return fake


ok_all = True
print("case 1: pinch_pq at 1150 Hz")
f = make_fake()
test = {"type": "pinch_pq", "frequencies_hz": [1150], "positions_steps": [0, 20, 40, 60],
        "targets_mbar": [20, 50, 100, 150], "close_max": True, "max_stop_mbar": 300, "settle_s": 1.0,
        "measure_s": 2.0, "hold_timeout_s": 25, "spin_up_s": 1.0}
t0 = time.perf_counter()
m.PumpTestRunner._exec_pinch_pq(f, test, lambda: None)
rows = list(csv.DictReader(open(os.path.join(f.run_dir, "pinch_pq.csv"))))
for r in rows:
    print(f"   {r['kind']:8s} target {r['target_mbar']:>4s}  squeeze {r['past_touch_steps']:>6s}  "
          f"P {r['p_mean_mbar']:>7s} mbar  Q {r['flow_mean']:>8s} uL/min  {r['reason']}")
kinds = [r["kind"] for r in rows]
targets_ok = all(abs(float(r["p_mean_mbar"]) - float(r["target_mbar"])) < 6 for r in rows if r["kind"] == "target")
closed = [float(r["p_mean_mbar"]) for r in rows if r["kind"] == "closed"]
# the first measurable rise comes after geometric contact (a barely squeezed tube adds little resistance)
touch_ok = f.ctl.touch is not None and 600 <= f.ctl.touch <= 600 + 20 * 16
ok = (kinds.count("target") == 4 and kinds.count("position") == 4 and "open" in kinds and closed
      and closed[0] > 200 and targets_ok and touch_ok and not f.faults)
back_ok = f.ctl.min_squeeze_seen >= 0 and min(x for x in [f.plant.x]) >= 0
ok = ok and back_ok
print(f"   lowest squeeze ever {f.ctl.min_squeeze_seen / 16:.2f} full steps (must be >= 0: back limit {back_ok}); "
      f"touch at {f.ctl.touch / 16:.1f} full steps (plant contact at {600 / 16:.1f}); "
      f"closed {closed}; {time.perf_counter() - t0:.0f} s -> {'OK' if ok else 'FAIL'}")
ok_all &= ok
f.stop_flag.set()
f.ctl.close()

print("case 2: pressure limit 100 mbar, close to 300")
f = make_fake(limit=100.0)
f.freq = 1150
time.sleep(1.0)
f.ctl.find_touch()
p, why = f.ctl.close_until(300.0, 1.0)
time.sleep(1.5)
ok = bool(f.faults) and f.ctl.mode == "fault" and f.plant.pressure() < 95.0 and f.ctl.squeeze() >= 0
print(f"   close_until -> {p:.1f} mbar ({why}); faults {f.faults}; pressure after retract {f.plant.pressure():.1f} -> "
      f"{'OK' if ok else 'FAIL'}")
ok_all &= ok
f.stop_flag.set()
f.ctl.close()
print("all OK" if ok_all else "FAILED")
sys.exit(0 if ok_all else 1)
