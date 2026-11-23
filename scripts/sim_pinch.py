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

print("case 2: pressure limit 100 mbar, unguarded jump to 60 steps past touch -> fault + retract")
f = make_fake(limit=100.0)
f.freq = 1150
time.sleep(1.0)
f.ctl.find_touch()
why = 'direct jump'
try:
    f.ctl.set_squeeze(60)                       # unguarded position jump: must trip the limit and retract
except RuntimeError:
    pass
p = f.ctl.p_1s
time.sleep(1.5)
backed_off = (f.plant.x_max - f.plant.x) >= 39 * 16          # the fault retract (40 full steps) happened
ok = bool(f.faults) and f.ctl.mode == "fault" and backed_off and f.ctl.squeeze() >= 0
print(f"   {why} -> {p:.1f} mbar; faults {f.faults}; backed off {(f.plant.x_max - f.plant.x) / 16:.1f} full steps, squeeze {f.ctl.squeeze() / 16:.1f} -> "
      f"{'OK' if ok else 'FAIL'}")
ok_all &= ok
f.stop_flag.set()
f.ctl.close()
print("case 3: no tube in reach -> touch not found -> back to the start")
f = make_fake(pinch={"max_search_steps": 60})
f.plant.x_touch = 10 ** 9                      # the plunger never meets the tube
f.freq = 1150
time.sleep(1.0)
res = f.ctl.find_touch()
ok = res is None and f.ctl.squeeze() == 0 and f.ctl.min_squeeze_seen >= 0 and f.ctl.mode == "idle"
print(f"   find_touch -> {res}; squeeze now {f.ctl.squeeze() / 16:.2f} full steps (start = 0) -> {'OK' if ok else 'FAIL'}")
ok_all &= ok
f.stop_flag.set()
f.ctl.close()
print("case 4: the motor reports its position only when asked (POS?), as on the rig (27133)")


class LateSyringe(pc.DummySyringe):
    """Like the real driver: position 0 until the firmware reports it."""
    def __init__(self, plant, lo, hi):
        super().__init__(plant, lo, hi, "v1")
        self.true_pos = 27133
        self.pos = 0

    def query_position(self):
        with self.lock:
            self.pos = self.true_pos
        return self.position()

    def _do(self, steps):
        self.plant.step(1 if steps > 0 else -1, abs(steps))
        with self.lock:
            self.true_pos -= steps
            self.pos = self.true_pos


fz = types.SimpleNamespace(freq=0)
pl = pinch_control.PinchPlant(get_freq=lambda: fz.freq, x_touch=600, tau_s=0.3)
ls = LateSyringe(pl, 0, 2470 * 16)
c4 = pinch_control.PinchController({"limit_mbar": 340, "pinch": FAST}, ls, pl.read_sensor, pl.range,
                                   run_dir=tempfile.mkdtemp(), microsteps=16)
c4.open()
start_ok = c4.start_pos == 27133
c4.start(time.perf_counter())
moved = c4._move_full(-20)                      # a backward request at the back limit: must move nothing
ok = start_ok and moved == 0 and c4.squeeze() == 0 and ls.true_pos == 27133
print(f"   start_pos {c4.start_pos} (true 27133); backward request moved {moved} counts; position {ls.true_pos} "
      f"-> {'OK' if ok else 'FAIL'}")
ok_all &= ok
c4.close()
print("case 5: the 23 Nov rig: steep closure, the pump can exceed the sensor range; absolute back limit + approach")
f5 = types.SimpleNamespace(freq=0)
pl5 = pinch_control.PinchPlant(get_freq=lambda: f5.freq, x_touch=480, k_mm_per_count=0.005, r_line=0.019,
                               r_tube=0.004, q0=450.0, pmax=450.0, tau_s=0.3)
sy5 = pc.DummySyringe(pl5, 0, 2470 * 16, "v1")
start5 = sy5.position()
refused = False
try:
    bad = pinch_control.PinchController({"limit_mbar": 340, "pinch": {**FAST, "back_limit_pos": start5 - 100}},
                                        sy5, pl5.read_sensor, pl5.range, microsteps=16)
    bad.open()
except RuntimeError as e:
    refused = "BEHIND" in str(e)
fake5 = make_fake()
fake5.stop_flag.set(); fake5.ctl.close()                 # reuse the helper's runner wiring with a new controller
f = make_fake()
f.stop_flag.set(); f.ctl.close()
f.stop_flag = threading.Event()
f.plant = pl5
c5 = pinch_control.PinchController({"limit_mbar": 340, "pinch": {**FAST, "back_limit_pos": start5, "approach_steps": 20}},
                                   sy5, pl5.read_sensor, pl5.range, run_dir=tempfile.mkdtemp(),
                                   on_fault=f.faults.append, microsteps=16)
c5.open(); c5.zero_sensor(0.5); f.t0 = time.perf_counter(); c5.start(f.t0)
f.ctl = c5
f.run_dir = c5.run_dir
f._set_freq = lambda hz: setattr(f5, "freq", int(hz))
def feeder5():
    while not f.stop_flag.is_set():
        q = pl5.pump_flow_ul_min(pl5.pressure())
        with f.lock:
            f.buf_t.append(time.perf_counter() - f.t0); f.buf_flow.append(q)
        time.sleep(0.04)
threading.Thread(target=feeder5, daemon=True).start()
test5 = {"type": "pinch_pq", "frequencies_hz": [600], "positions_steps": [0, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16],
         "targets_mbar": [25, 100, 200, 300], "close_max": True, "max_stop_mbar": 320, "settle_s": 0.5,
         "measure_s": 1.0, "hold_timeout_s": 25, "spin_up_s": 1.0}
m.PumpTestRunner._exec_pinch_pq(f, test5, lambda: None)
rows5 = list(csv.DictReader(open(os.path.join(f.run_dir, "pinch_pq.csv"))))
for r in rows5:
    print(f"   {r['kind']:8s} target {r['target_mbar']:>4s} past {r['past_touch_steps']:>6s}  P {r['p_mean_mbar']:>7s}  "
          f"Q {r['flow_mean']:>8s}  {r['reason']}")
pmax_seen = max(float(r["p_mean_mbar"]) for r in rows5)
ok = refused and not f.faults and pmax_seen < 327.5 and c5.min_squeeze_seen >= c5.back_limit
print(f"   start behind the limit refused: {refused}; faults {f.faults}; highest point {pmax_seen:.1f} mbar "
      f"(limit 327.5) -> {'OK' if ok else 'FAIL'}")
ok_all &= ok
f.stop_flag.set(); c5.close()
print("case 6: the 23 Nov 520 Hz curve: x1.5-3 per step near closure, +1-step positions (the run that faulted at +10)")
f6 = types.SimpleNamespace(freq=0)
pl6 = pinch_control.PinchPlant(get_freq=lambda: f6.freq, x_touch=480, k_mm_per_count=0.0085, r_line=0.015,
                               r_tube=0.003, q0=470.0, pmax=600.0, tau_s=0.3)
sy6 = pc.DummySyringe(pl6, 0, 2470 * 16, "v1")
g6 = make_fake()
g6.stop_flag.set(); g6.ctl.close()
g6.stop_flag = threading.Event()
c6 = pinch_control.PinchController({"limit_mbar": 340, "pinch": {**FAST, "back_limit_pos": sy6.position(),
                                                                  "approach_steps": 20}},
                                   sy6, pl6.read_sensor, pl6.range, run_dir=tempfile.mkdtemp(),
                                   on_fault=g6.faults.append, microsteps=16)
c6.open(); c6.zero_sensor(0.5); g6.t0 = time.perf_counter(); c6.start(g6.t0)
g6.ctl, g6.run_dir = c6, c6.run_dir
g6._set_freq = lambda hz: setattr(f6, "freq", int(hz))
def feeder6():
    while not g6.stop_flag.is_set():
        q = pl6.pump_flow_ul_min(pl6.pressure())
        with g6.lock:
            g6.buf_t.append(time.perf_counter() - g6.t0); g6.buf_flow.append(q)
        time.sleep(0.04)
threading.Thread(target=feeder6, daemon=True).start()
test6 = {"type": "pinch_pq", "frequencies_hz": [520], "positions_steps": list(range(0, 17)),
         "targets_mbar": [100, 250, 300], "close_max": True, "max_stop_mbar": 300, "settle_s": 0.5,
         "measure_s": 1.0, "hold_timeout_s": 25, "spin_up_s": 1.0}
m.PumpTestRunner._exec_pinch_pq(g6, test6, lambda: None)
rows6 = list(csv.DictReader(open(os.path.join(g6.run_dir, "pinch_pq.csv"))))
for r in rows6:
    print(f"   {r['kind']:8s} target {r['target_mbar']:>4s} past {r['past_touch_steps']:>6s}  P {r['p_mean_mbar']:>7s}  {r['reason']}")
pmax6 = max(float(r["p_mean_mbar"]) for r in rows6)
ok = not g6.faults and pmax6 < 327.5
print(f"   faults {g6.faults}; highest point {pmax6:.1f} mbar -> {'OK' if ok else 'FAIL'}")
ok_all &= ok
g6.stop_flag.set(); c6.close()
print("all OK" if ok_all else "FAILED")
sys.exit(0 if ok_all else 1)
