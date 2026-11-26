"""Offline check of the G test ('pressure' block, target_mbar 'pv_steps'), real time, no hardware.

A fake runner holds the real PressureController on a simulated line with check valves:
  outlet line (compliance C_line) -- outlet valve (forward = chamber -> line) -- chamber (C_ch + gas_ul of gas
  at 1 atm, isothermal) -- inlet valve -- inlet clamped.
Pulling at the syringe: the outlet valve opens, the chamber follows (C_line + C_ch + gas term).
Pushing: the outlet valve closes, only the line responds (C_line). 'Unclamping the inlet' refills the chamber.
The syringe is pressure_control.DummySyringe (16 microsteps, 22 full steps of backlash), and a thread feeds
the flow buffer from the true plunger motion (the flow-sensor cross-check).

Cases:
  1. pull to -300 mbar, push back to ~0, unclamp, pull again: rows written, slack taken up, the guard holds,
     no fault; then scripts/analyze_pv_steps.py must recover C_line, C_ch and gas_ul.
  2. room: room_steps 3 -> the block ends on 'room' without crossing it.
usage: python scripts/sim_pv_steps.py [--gas 2.5] [--hold 2]   (about 2-3 min; exit 1 on failure)
"""
import argparse
import collections
import csv
import math
import os
import subprocess
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
import pressure_control as pc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gas", type=float, default=2.5, help="gas in the chamber at 1 atm (uL)")
ap.add_argument("--c-line", type=float, default=0.020, help="outlet line compliance (uL/mbar)")
ap.add_argument("--c-ch", type=float, default=0.0015, help="chamber structural compliance (uL/mbar)")
ap.add_argument("--crack", type=float, default=15.0, help="outlet valve cracking pressure (mbar)")
ap.add_argument("--hold", type=float, default=2.0, help="hold per move (s); the rig config uses 8")
a = ap.parse_args()
MS = 16


def quad_p(c: float, g: float, v: float, pa: float) -> float:
    """p (mbar gauge) with c p + g p / (pa + p) = v."""
    if g <= 0:
        return v / c
    b = c * pa + g - v
    return (-b + math.sqrt(b * b + 4.0 * c * v * pa)) / (2.0 * c)


class ValvedPlant(pc.PressurePlant):
    def __init__(self, c_line, c_ch, gas_ul, crack_mbar=0.0, **kw):
        super().__init__(compliance_ul_mbar=c_line, r_leak=1e12, **kw)
        self.c_line, self.c_ch, self.gas, self.crack = c_line, c_ch, gas_ul, crack_mbar
        self.v_ch = 0.0                       # excess volume in the chamber compartment (uL)

    def pressure(self) -> float:
        with self.lock:
            self._advance()
            p_line = (self.v_in - self.v_ch) / self.c_line
            p_ch = quad_p(self.c_ch, self.gas, self.v_ch, self.p_atm)
            if p_line < p_ch - self.crack:    # outlet valve opens: the chamber follows, crack above the line
                q = quad_p(self.c_line + self.c_ch, self.gas, self.v_in + self.c_line * self.crack, self.p_atm)
                self.v_ch = self.c_ch * q + self.gas * q / (self.p_atm + q)
                p_line = q - self.crack
            return p_line

    inlet_open, crack_in = False, 10.0       # inlet left open: the chamber can't go below -crack_in

    def pin_inlet(self):
        if not self.inlet_open:
            return
        p_floor = -(self.crack_in + self.crack)
        p_line = (self.v_in - self.v_ch) / self.c_line
        if p_line < p_floor:                  # liquid comes in from the reservoir through the pump
            q = -self.crack_in
            self.v_ch = self.c_ch * q + self.gas * q / (self.p_atm + q)
            self.v_in = self.c_line * p_floor + self.v_ch

    def read_sensor(self):
        with self.lock:
            self.pin_inlet()
        return super().read_sensor()

    def unclamp_inlet(self):
        with self.lock:                       # the chamber refills from the reservoir to 0 mbar
            self.v_in -= self.v_ch
            self.v_ch = 0.0


def make_fake(room_steps_note=""):
    fake = types.SimpleNamespace()
    plant = ValvedPlant(a.c_line, a.c_ch, a.gas, a.crack, ul_per_step=0.4049 / MS, backlash=22 * MS, p_lose=0.0,
                        pos_max=2470 * MS, sensor_range=(-344.74, 344.74), offset_mbar=0.8, noise_mbar=0.3)
    syr = pc.DummySyringe(plant, 0, 2470 * MS, "v2")
    faults = []
    run_dir = tempfile.mkdtemp(prefix="sim_pv_")
    cfg = {"limit_mbar": 100.0, "min_mbar": -330.0, "microsteps": MS, "ul_per_step": 0.4049, "firmware": "v2",
           "backend": "dummy"}
    ctl = pc.PressureController(cfg, syr, plant.read_sensor, plant.range, run_dir=run_dir, on_fault=faults.append)
    ctl.open()                                # opens the syringe too
    ctl.zero_sensor(0.5)
    fake.t0 = time.perf_counter()
    ctl.start(fake.t0)
    fake.ctl, fake.plant, fake.faults, fake.run_dir = ctl, plant, faults, run_dir
    fake.lock = threading.Lock()
    fake.buf_t = collections.deque(maxlen=20000)
    fake.buf_flow = collections.deque(maxlen=20000)
    fake.stop_flag = threading.Event()
    fake.freq_set = 0
    fake.statuses = []
    fake._emit = lambda k, msg: (fake.statuses.append(str(msg)), print("   STATUS:", msg))
    fake._set_freq = lambda hz: setattr(fake, "freq_set", int(hz))
    fake._need_ctl = lambda: fake.ctl
    fake._pv_file = fake._pv_writer = fake._pv_origin = None
    fake._pv_n = 0
    for name in ("_window_flow_stats", "_pv_window", "_pv_flow_volume", "_write_pv_row", "_exec_pv_steps"):
        setattr(fake, name, types.MethodType(getattr(m.PumpTestRunner, name), fake))

    def feeder():                             # flow sensor: liquid towards the syringe = forward (+)
        last_pos, last_t = plant.pos, time.perf_counter()
        while not fake.stop_flag.is_set():
            time.sleep(0.04)
            now = time.perf_counter()
            pos = plant.pos
            q = (pos - last_pos) * plant.ul_step / (now - last_t) * 60.0 + 0.5    # + a 0.5 uL/min offset
            last_pos, last_t = pos, now
            with fake.lock:
                fake.buf_t.append(now - fake.t0)
                fake.buf_flow.append(q)

    threading.Thread(target=feeder, daemon=True).start()
    return fake


def block(direction, until, rep, section, **kw):
    b = {"type": "pressure", "target_mbar": "pv_steps", "direction": direction, "until_mbar": until,
         "target_dp_mbar": 20, "first_counts": 4, "min_counts": 2, "max_counts": 160, "max_moves": 40,
         "hold_s": a.hold, "measure_s": a.hold / 2, "takeup_mbar": 2.0, "takeup_counts": 4, "takeup_wait_s": 0.3,
         "takeup_max_counts": 800, "room_steps": 150, "section": section, "rep": rep,
         "fine_mbar": 40.0 if direction == "pull" else 0.0, "fine_counts": 6, "max_c_ul_mbar": 0.15, "check_counts": 30}
    b.update(kw)
    return b


ok_all = True
print(f"case 1: pull to -300, push back, unclamp, pull again (C_line {a.c_line}, C_ch {a.c_ch}, gas {a.gas} uL)")
f = make_fake()
t0 = time.perf_counter()
for rep in (1, 2):
    f._exec_pv_steps(block("pull", -300.0, rep, "g_pull"), lambda: None)
    f._exec_pv_steps(block("push", -10.0, rep, "g_push"), lambda: None)
    f.plant.unclamp_inlet()
    time.sleep(1.0)
rows = list(csv.DictReader(open(os.path.join(f.run_dir, "pv_steps.csv"))))
pulls = [r for r in rows if r["direction"] == "pull" and r["phase"] == "step"]
pushes = [r for r in rows if r["direction"] == "push" and r["phase"] == "step"]
takeups = [int(r["counts"]) for r in rows if r["phase"] == "takeup"]
p_min_seen = min(h[1] for h in f.ctl.hist)
reached = [float(r["p_after_mbar"]) for r in pulls]
ok = (len(pulls) >= 10 and len(pushes) >= 3 and not f.faults and p_min_seen > -327.5 and min(reached) <= -280
      and all(abs(t) >= 22 * MS * 0.8 for t in takeups[1:]))
print(f"   {len(pulls)} pull moves, {len(pushes)} push moves, slack take-ups {takeups} counts "
      f"(backlash {22 * MS}); lowest pressure {p_min_seen:.1f} mbar; faults {f.faults}; "
      f"{time.perf_counter() - t0:.0f} s -> {'OK' if ok else 'FAIL'}")
ok_all &= ok
f.stop_flag.set()
f.ctl.close()
csv_path = os.path.join(f.run_dir, "pv_steps.csv")
print(f"   pv_steps.csv: {csv_path}")

print("   analysis (scripts/analyze_pv_steps.py, line from the push branch):")
res = subprocess.run([sys.executable, os.path.join(W, "scripts", "analyze_pv_steps.py"), f.run_dir,
                      "--line-from-push", "--json"], capture_output=True, text=True)
print("   " + res.stdout.strip().replace("\n", "\n   "))
if res.returncode != 0:
    print(res.stderr)
    ok_all = False
else:
    import json
    fit = json.loads(res.stdout.strip().splitlines()[-1])
    ok = (abs(fit["gas_ul"] - a.gas) <= max(0.35, 0.15 * a.gas) and abs(fit["c_struct_ul_mbar"] - a.c_ch) <= 0.0008
          and abs(fit["p_crack_mbar"] - a.crack) <= 5.0 and fit["gate"].get("verdict") in ("GO", "NO-GO"))
    print(f"   recovered gas {fit['gas_ul']:.2f} uL (true {a.gas}; {fit['gas_ul_no_pcr']:.2f} without the P_cr "
          f"correction), structure {fit['c_struct_ul_mbar']:.4f} uL/mbar (true {a.c_ch}), P_cr "
          f"{fit['p_crack_mbar']:.1f} mbar (true {a.crack}), gate {fit['gate'].get('verdict')} -> "
          f"{'OK' if ok else 'FAIL'}")
    ok_all &= ok

print("case 2: room_steps 3 -> ends on the room limit")
f = make_fake()
f._exec_pv_steps(block("pull", -300.0, 1, "room", room_steps=3), lambda: None)
pos_span = abs(f.ctl.syr.position() - f._pv_origin)
ok = pos_span <= 3 * MS and any("room" in s for s in f.statuses) and not f.faults
print(f"   moved {pos_span} counts (room {3 * MS}) -> {'OK' if ok else 'FAIL'}")
ok_all &= ok
f.stop_flag.set()
f.ctl.close()
print("case 3: syringe mounted the other way round -> the block raises, the run stops")
f = make_fake()
_move = f.ctl.syr.move
f.ctl.syr.move = lambda steps, timeout_s=10.0: _move(-steps, timeout_s)    # every move goes the other way
try:
    f._exec_pv_steps(block("pull", -300.0, 1, "reversed"), lambda: None)
    raised = None
except RuntimeError as e:
    raised = str(e)
p_max_seen = max(h[1] for h in f.ctl.hist)
ok = raised is not None and "wrong way" in raised and p_max_seen < 30 and not f.faults
print(f"   raised: {raised}; highest pressure {p_max_seen:.1f} mbar -> {'OK' if ok else 'FAIL'}")
ok_all &= ok
f.stop_flag.set()
f.ctl.close()
for name, setup in (("case 4: air in the line (0.5 uL/mbar, as on the rig 26 Nov) -> stops in the slack search",
                     lambda pl: setattr(pl, "c_line", 0.5)),
                    ("case 5: inlet left open (pressure stalls at -25 mbar) -> 'line too soft' stop",
                     lambda pl: setattr(pl, "inlet_open", True))):
    print(name)
    f = make_fake()
    setup(f.plant)
    t1 = time.perf_counter()
    try:
        f._exec_pv_steps(block("pull", -300.0, 1, "soft"), lambda: None)
        raised = None
    except RuntimeError as e:
        raised = str(e)
    ok = raised is not None and ("too soft" in raised or "no 4-count move" in raised) and not f.faults
    print(f"   raised after {time.perf_counter() - t1:.0f} s: {raised} -> {'OK' if ok else 'FAIL'}")
    ok_all &= ok
    f.stop_flag.set()
    f.ctl.close()
print("case 6: a hand change leaves the line above the limit (clamping) -> released, no fault, then the staircase runs")
f = make_fake()
f._pv_recover_after_handling = types.MethodType(m.PumpTestRunner._pv_recover_after_handling, f)
f.ctl.set_handling(True)
with f.plant.lock:
    f.plant.v_in += 150.0 * f.plant.c_line      # +150 mbar in the line (limit 100 in this sim)
time.sleep(1.0)
p_hand = f.ctl.p
f._pv_recover_after_handling(f.ctl, "clamp", {"room_steps": 150})
f.ctl.set_handling(False)
p_after = f.ctl.p
time.sleep(0.5)
try:
    f._exec_pv_steps(block("pull", -100.0, 1, "after_recover"), lambda: None)
    raised = None
except RuntimeError as e:
    raised = str(e)
ok = p_hand > 100 and p_after <= 31.0 and not f.faults and raised is None and f.ctl.p <= -95
print(f"   hand change {p_hand:.0f} mbar -> released to {p_after:.1f}; staircase to {f.ctl.p:.0f} mbar; faults {f.faults}; "
      f"raised {raised} -> {'OK' if ok else 'FAIL'}")
ok_all &= ok
f.stop_flag.set()
f.ctl.close()
print("ALL OK" if ok_all else "FAILED")
sys.exit(0 if ok_all else 1)
