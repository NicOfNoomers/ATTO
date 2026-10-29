"""Pressure control with the stepper syringe, for PQ curves (pump flow vs outlet pressure).

Rig (NM, 29 Oct 2026): a 1 mL liquid-filled syringe on the pump OUTLET, the pressure sensor
(a Fluigent channel) between the outlet and the syringe, a stepper on an Arduino Uno.
"F<n>" pushes the plunger in (raises pressure, position counts down), "B<n>" pulls it out.
Position 0 = empty syringe; the old tool's range is 0..2470 steps. Steps per revolution and
microstepping are unknown and the motor loses steps / sticks near the range ends, so the
controller learns the pressure gain of a step while it runs instead of trusting the steps.

Pieces:
  SyringeV1     the existing firmware (blocking F/B moves answered by "DONE POS=..")
  SyringeV2     firmware v2 (firmware/syringe_v2): non-blocking moves, velocity mode, STOP,
                position stream, position kept in EEPROM across the Uno's reset-on-connect
  DummySyringe  + PressurePlant: a simulated line (compliance, pump PQ, leak, backlash, lost
                steps, sticking near the ends, sensor offset/noise/range) for tests without rig
  PressureController
                two threads: a sampler (sensor at log_hz, logging, safety) and a controller
                (moves). Modes: idle, hold(target) with a ramp limit, release (to ~0 mbar, then
                idle), jog. Learns g = mbar per step (separately for pushing and pulling),
                q = pressure drift (pump flow into the closed line), the backlash on a direction
                reversal, and flags sticking. Safety: |P| above the limit (default 95 % of the
                sensor range, at most limit_mbar) or a saturated / stale sensor -> fault: the
                motor stops, overpressure is released, on_fault() runs (the runner stops the pump).
  Calibration   sensor zero, gain/backlash sweep, leak test, step response; pressure_cal.json.

All times use time.perf_counter (the dashboard's run clock).
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import re
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

clock = time.perf_counter

PRESSURE_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "backend": "arduino",         # "arduino" | "dummy"
    "port": "COM8",
    "baud": 115200,
    "firmware": "auto",           # "auto" (asks VER?) | "v1" | "v2"
    "sensor": {"backend": "fluigent", "channel": 1, "server": "localhost:8765"},
    "limit_mbar": 300.0,          # never above 95 % of the sensor range either
    "min_mbar": -100.0,
    "rate_mbar_s": 40.0,          # setpoint ramp limit (protects the membrane)
    "tolerance_mbar": 2.0,
    "settle_s": 3.0,              # within tolerance this long = settled
    "pos_min": 0,
    "pos_max": 2470,
    "pos_margin": 30,             # stay this far from the ends (sticking)
    "max_chunk": 200,             # v1: largest move per cycle (steps)
    "move_settle_s": 0.25,        # v1: wait after a move before judging it
    "kp": 0.6,                    # fraction of the error corrected per move (v1) / gain (v2)
    "ki": 0.2,                    # v2 integral gain (1/s)
    "vmax_sps": 300.0,            # v2 speed limit (steps/s)
    "gain_init_mbar_step": 1.0,   # until calibrated / learned
    "ul_per_step": None,          # syringe volume per step, if known (syringe flow in uL/min)
    "log_hz": 25.0,
    "stale_s": 1.0,
    "release_on_stop": True,
    "clear_reserve_steps": 150,   # PQ: ask to clear the syringe when fewer steps than this remain
    "state_file": None,           # where the last syringe position is kept (None = default path)
}


def merged_cfg(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    c = {**PRESSURE_DEFAULTS, **(cfg or {})}
    c["sensor"] = {**PRESSURE_DEFAULTS["sensor"], **((cfg or {}).get("sensor") or {})}
    return c


# ---------------- simulated rig ----------------
class PressurePlant:
    """Closed liquid line between pump outlet and syringe.

    P = (V_in) / C with V_in the excess volume (uL) pushed into the line; the pump adds
    Q(P) = Q0(f) x (1 - P / P_shutoff) uL/min, a small leak removes P / R_leak, the syringe adds
    ul_per_step for every step it pushes in (F). Backlash: after a direction reversal the first
    `backlash` steps move nothing. Lost steps: each step is lost with p_lose. Near the range ends
    (within stick_zone) a move does nothing with p_stick.
    """

    def __init__(self, get_freq: Callable[[], float] = lambda: 0.0, compliance_ul_mbar: float = 0.25,
                 ul_per_step: float = 0.405, p_shutoff: float = 200.0, q_per_hz: float = 0.67,
                 q_max: float = 400.0, r_leak: float = 400.0, backlash: int = 6, p_lose: float = 0.02,
                 stick_zone: int = 80, p_stick: float = 0.3, pos_max: int = 2470, sensor_range=(-312.0, 312.0),
                 offset_mbar: float = 1.5, noise_mbar: float = 0.3, seed: int = 1):
        self.get_freq = get_freq
        self.C, self.ul_step = compliance_ul_mbar, ul_per_step
        self.p_shutoff, self.q_per_hz, self.q_max, self.r_leak = p_shutoff, q_per_hz, q_max, r_leak
        self.backlash, self.p_lose, self.stick_zone, self.p_stick = backlash, p_lose, stick_zone, p_stick
        self.pos_max, self.range, self.offset, self.noise = pos_max, sensor_range, offset_mbar, noise_mbar
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.v_in = 0.0                 # uL in the line above atmospheric
        self.pos = pos_max // 2         # true plunger position (steps)
        self.last_dir = 0
        self.slack = 0                  # backlash still to take up
        self.t = clock()

    def pump_flow_ul_min(self, p: float) -> float:
        f = float(self.get_freq() or 0.0)
        if f <= 0:
            return 0.0
        q0 = min(self.q_max, self.q_per_hz * f)
        return max(0.0, q0 * (1.0 - p / self.p_shutoff))

    def _advance(self):
        now = clock()
        dt, self.t = now - self.t, now
        if dt <= 0:
            return
        p = self.v_in / self.C
        dv = (self.pump_flow_ul_min(p) / 60.0 - p / self.r_leak) * dt
        self.v_in += dv

    def pressure(self) -> float:
        with self.lock:
            self._advance()
            return self.v_in / self.C

    def read_sensor(self) -> Tuple[bool, float]:
        p = self.pressure() + self.offset + self.rng.gauss(0.0, self.noise)
        return True, max(self.range[0], min(self.range[1], p))

    def step(self, direction: int, n: int) -> int:
        """direction +1 = push in (raise P, position down). Returns the position change seen by
        the firmware (it always counts all commanded steps)."""
        with self.lock:
            self._advance()
            if direction != self.last_dir:
                self.slack = self.backlash if self.last_dir != 0 else 0
                self.last_dir = direction
            near_end = self.pos < self.stick_zone or self.pos > self.pos_max - self.stick_zone
            if near_end and self.rng.random() < self.p_stick:
                return n                                  # stuck: nothing moves
            moved = 0
            for _ in range(n):
                if self.slack > 0:
                    self.slack -= 1
                    continue
                if self.rng.random() < self.p_lose:
                    continue
                moved += 1
            moved = min(moved, self.pos) if direction > 0 else min(moved, self.pos_max - self.pos)
            self.pos -= direction * moved
            self.v_in += direction * moved * self.ul_step
            return n


# ---------------- syringe drivers ----------------
class SyringeBase:
    protocol = "v1"

    def __init__(self, pos_min: int, pos_max: int):
        self.pos_min, self.pos_max = pos_min, pos_max
        self.pos = 0                    # firmware's step count
        self.offset = 0                 # position = firmware count + offset (v1 forgets on reset)
        self.lock = threading.Lock()

    def open(self): ...
    def close(self): ...

    def position(self) -> int:
        with self.lock:
            return self.pos + self.offset

    def set_position(self, steps: int):
        """Tell the software where the plunger is (e.g. 0 after emptying it by hand)."""
        with self.lock:
            self.offset = int(steps) - self.pos

    def within(self, steps: int, margin: int = 0) -> bool:
        """Would a move of `steps` (+ = push in = position down) stay inside the range?"""
        new = self.position() - steps
        return self.pos_min + margin <= new <= self.pos_max - margin

    def move(self, steps: int, timeout_s: float = 10.0) -> bool:
        raise NotImplementedError

    def set_velocity(self, sps: float):
        raise RuntimeError("velocity mode needs firmware v2")

    def stop(self): ...
    def zero(self): ...
    def set_hold(self, on: bool): ...


class _SerialSyringe(SyringeBase):
    """Line-based serial link with a reader thread."""

    def __init__(self, port: str, baud: int, pos_min: int, pos_max: int):
        super().__init__(pos_min, pos_max)
        self.port, self.baud = port, baud
        self.ser = None
        self._lines: deque = deque(maxlen=200)
        self._cv = threading.Condition()
        self._reader: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._wlock = threading.Lock()
        self.last_error: Optional[str] = None

    def open(self):
        import serial
        s = serial.Serial()
        s.port, s.baudrate, s.timeout = self.port, self.baud, 0.1
        s.dtr = False          # an Uno resets on DTR; keeping it low avoids that on many PCs
        s.open()
        self.ser = s
        self._stop.clear()
        self._reader = threading.Thread(target=self._read_loop, name="syringe-reader", daemon=True)
        self._reader.start()
        # If opening the port reset the Uno, its bootloader swallows commands for ~1.5 s: ask
        # for the position until it answers (at once when there was no reset)
        for _ in range(3):
            if self.query_position() is not None:
                break
        else:
            self.close()
            raise RuntimeError(f"syringe Arduino on {self.port} does not answer POS? (wrong port, "
                               f"or not the syringe firmware?)")

    def close(self):
        self._stop.set()
        try:
            self.stop()
        except Exception:
            pass
        if self._reader:
            self._reader.join(timeout=1.0)
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None

    def send(self, cmd: str):
        with self._wlock:
            self.ser.write((cmd + "\n").encode("ascii"))
            self.ser.flush()

    def _read_loop(self):
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = self.ser.read(256)
            except Exception as e:
                self.last_error = f"serial read failed: {e!r}"
                time.sleep(0.2)
                continue
            if not chunk:
                continue
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                text = line.decode("ascii", "ignore").strip()
                if text:
                    self._on_line(text)

    def _on_line(self, text: str):
        m = re.search(r"POS=(-?\d+)", text) or re.fullmatch(r"P=(-?\d+).*", text)
        if m:
            with self.lock:
                self.pos = int(m.group(1))
        if text.startswith("ERR"):
            self.last_error = text
        with self._cv:
            self._lines.append((clock(), text))
            self._cv.notify_all()

    def wait_for(self, pattern: str, timeout_s: float, since: float) -> Optional[str]:
        rx = re.compile(pattern)
        t_end = clock() + timeout_s
        with self._cv:
            while True:
                for t, line in list(self._lines):
                    if t >= since and (rx.search(line) or line.startswith("ERR")):
                        return line
                left = t_end - clock()
                if left <= 0:
                    return None
                self._cv.wait(min(left, 0.05))

    def query_position(self) -> Optional[int]:
        t = clock()
        self.send("POS?")
        line = self.wait_for(r"^POS=", 1.0, t)
        return self.position() if line and not line.startswith("ERR") else None

    def zero(self):
        t = clock()
        self.send("ZERO")
        line = self.wait_for(r"ZERO", 1.0, t)
        if not line or line.startswith("ERR"):
            raise RuntimeError(f"syringe did not confirm ZERO ({line})")
        with self.lock:
            self.pos, self.offset = 0, 0

    def detect_version(self) -> str:
        """'v2' if the firmware answers VER?, else 'v1' (it answers ERR UNKNOWN_CMD)."""
        t = clock()
        self.send("VER?")
        line = self.wait_for(r"^VER|UNKNOWN", 1.5, t)
        return "v2" if line and line.startswith("VER") and "v2" in line else "v1"


class SyringeV1(_SerialSyringe):
    """Existing firmware: F<n>/B<n> block until 'DONE POS=<p>'; no stop during a move."""
    protocol = "v1"

    def move(self, steps: int, timeout_s: float = 10.0) -> bool:
        steps = int(steps)
        if steps == 0:
            return True
        t = clock()
        self.send(f"{'F' if steps > 0 else 'B'}{abs(steps)}")
        line = self.wait_for(r"DONE POS=", max(1.0, min(timeout_s, 0.003 * abs(steps) + 1.0)), t)
        return bool(line) and not line.startswith("ERR")


class SyringeV2(_SerialSyringe):
    """Firmware v2 (firmware/syringe_v2/syringe_v2.ino): M<n> non-blocking move, V<sps>
    velocity (+ = push in), S stop, HOLD 0/1, streams 'P=<pos> V=<sps>'; keeps v1 F/B."""
    protocol = "v2"

    def open(self):
        super().open()
        self.send("STREAM 1")

    def move(self, steps: int, timeout_s: float = 10.0) -> bool:
        steps = int(steps)
        if steps == 0:
            return True
        t = clock()
        self.send(f"M{steps}")
        line = self.wait_for(r"DONE POS=", max(1.0, min(timeout_s, 0.01 * abs(steps) + 1.0)), t)
        return bool(line) and not line.startswith("ERR")

    def set_velocity(self, sps: float):
        self.send(f"V{int(round(sps))}")

    def stop(self):
        if self.ser:
            self.send("S")

    def set_hold(self, on: bool):
        self.send(f"HOLD {1 if on else 0}")


class DummySyringe(SyringeBase):
    """Simulated syringe on a PressurePlant; v1 timing (2 ms/step, blocking) or v2 velocity."""

    def __init__(self, plant: PressurePlant, pos_min: int, pos_max: int, protocol: str = "v1"):
        super().__init__(pos_min, pos_max)
        self.plant, self.protocol = plant, protocol
        self.pos = plant.pos
        self._vel = 0.0
        self._vt = clock()
        self._acc = 0.0
        self._vel_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def open(self):
        if self.protocol == "v2":
            self._stop.clear()
            self._vel_thread = threading.Thread(target=self._vel_loop, daemon=True)
            self._vel_thread.start()

    def close(self):
        self._stop.set()
        self._vel = 0.0

    def _vel_loop(self):
        while not self._stop.is_set():
            now = clock()
            dt, self._vt = now - self._vt, now
            self._acc += self._vel * dt
            n = int(self._acc)
            if n:
                self._acc -= n
                self._do(n)
            time.sleep(0.005)

    def _do(self, steps: int):
        d = 1 if steps > 0 else -1
        self.plant.step(d, abs(steps))
        with self.lock:
            self.pos -= steps

    def move(self, steps: int, timeout_s: float = 10.0) -> bool:
        steps = int(steps)
        if steps:
            time.sleep(0.002 * abs(steps) + 0.01)     # v1 firmware speed: 2 ms per step
            self._do(steps)
        return True

    def set_velocity(self, sps: float):
        self._vel = float(sps)

    def stop(self):
        self._vel = 0.0

    def zero(self):
        with self.lock:
            self.pos, self.offset = 0, 0


def default_state_path() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, "ATTO", "syringe_state.json")


def load_state(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(path: str, state: Dict[str, Any]):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, path)
    except Exception:
        pass


def make_syringe(cfg: Dict[str, Any], plant: Optional[PressurePlant] = None) -> SyringeBase:
    """Syringe driver from the pressure config. backend 'dummy' needs the simulated plant.
    firmware 'auto' asks VER? (v2 answers, v1 says ERR UNKNOWN_CMD)."""
    c = merged_cfg(cfg)
    lo, hi = int(c["pos_min"]), int(c["pos_max"])
    fw = str(c.get("firmware") or "auto").lower()
    if str(c.get("backend")).lower() == "dummy":
        if plant is None:
            raise RuntimeError("dummy syringe needs a PressurePlant")
        return DummySyringe(plant, lo, hi, "v2" if fw == "v2" else "v1")
    if fw == "auto":
        probe = SyringeV1(str(c["port"]), int(c["baud"]), lo, hi)
        probe.open()
        try:
            fw = probe.detect_version()
        finally:
            probe.close()
    cls = SyringeV2 if fw == "v2" else SyringeV1
    return cls(str(c["port"]), int(c["baud"]), lo, hi)


def restore_position(syr: SyringeBase, state_path: str, port: str) -> Optional[int]:
    """The v1 firmware restarts at 0 whenever the Uno resets (often on connect): carry the last
    known position over from the state file. Returns the restored position, or None."""
    st = load_state(state_path)
    if st.get("port") != port or not isinstance(st.get("pos"), int):
        return None
    if isinstance(syr, _SerialSyringe):
        syr.query_position()
    syr.set_position(int(st["pos"]))
    return int(st["pos"])


# ---------------- learning ----------------
class GainEstimator:
    """Learns how much pressure a step makes, per direction, the drift, and the backlash.

    Model for a move: dp = d x g_d x n_eff + q x dt (n_eff = steps that engaged, i.e. minus the
    backlash after a reversal; q = drift in mbar/s from the pump flow into the closed line and
    leaks). During a hold the moves come at a steady rhythm, so g and q can't both be fitted
    freely from the same moves (tried: a joint least-squares fit diverged). Instead:
      * g is anchored to the calibrated value g0 (sweep with the pump off) and only adapts
        slowly, within 0.4..2.5 x g0, from moves whose effect is large compared with the drift;
      * q comes from what is left after the step effect: every move and every idle stretch;
      * backlash: EMA of the steps that moved nothing after a reversal;
      * sticking: a same-direction move whose drift-corrected effect is < 25 % of the
        prediction (not learned from).
    """

    def __init__(self, g_init: float = 1.0):
        self.g0 = {1: abs(g_init), -1: abs(g_init)}
        self._g = dict(self.g0)
        self.q = 0.0
        self.n = {1: 0, -1: 0}
        self.backlash = 0.0
        self.last_dir = 0
        self.stuck_in_row = 0

    @property
    def g(self) -> Dict[int, float]:
        return dict(self._g)

    @g.setter
    def g(self, val: Dict[int, float]):
        """Calibrated gains become the new anchors."""
        self.g0 = {1: abs(float(val[1])), -1: abs(float(val[-1]))}
        self._g = dict(self.g0)

    def gain(self, d: int) -> float:
        return max(0.02, self._g[1 if d > 0 else -1])

    def observe_idle(self, dp: float, dt: float):
        if dt > 0.05:
            self.q += 0.3 * (dp / dt - self.q)

    def observe_move(self, steps: int, dp: float, dt: float, noise: float = 0.5) -> Dict[str, Any]:
        """dp: pressure change across the move (+ = up), dt: its duration incl. settling."""
        d = 1 if steps > 0 else -1
        n = abs(steps)
        reversal = self.last_dir not in (0, d)
        self.last_dir = d
        res = {"reversal": reversal, "stuck": False}
        g = self.gain(d)
        effect = (dp - self.q * dt) * d             # drift-corrected, positive if it worked
        if reversal:
            if n >= 3:
                lost = max(0.0, n - effect / g)
                self.backlash += 0.3 * (min(lost, 60.0) - self.backlash)
            n_eff = max(0.0, n - self.backlash)
        else:
            n_eff = float(n)
            expected = n * g
            if n >= 2 and expected > 3 * noise and effect < 0.25 * expected:
                self.stuck_in_row += 1
                res["stuck"] = True
                return res                           # don't learn from a stuck move
            self.stuck_in_row = 0
        # Drift from the residual of this move
        if dt > 0.05:
            q_obs = (dp - d * g * n_eff) / dt
            self.q += 0.3 * (max(-500.0, min(500.0, q_obs)) - self.q)
        # Gain: only from moves that clearly dominate the drift, slowly, within bounds
        if not reversal and n_eff >= 3 and abs(d * g * n_eff) > 2.0 * abs(self.q * dt) + 3 * noise:
            ratio = (dp - self.q * dt) * d / n_eff
            g0 = self.g0[d]
            a = 0.3 if self.n[d] < 3 else 0.1
            self._g[d] = min(2.5 * g0, max(0.4 * g0, g + a * (ratio - g)))
        self.n[d] += 1
        return res


# ---------------- controller ----------------
class PressureController:
    def __init__(self, cfg: Dict[str, Any], syringe: SyringeBase,
                 read_sensor: Callable[[], Tuple[bool, Optional[float]]],
                 sensor_range: Optional[Tuple[float, float]] = None,
                 run_dir: Optional[str] = None,
                 emit: Callable[[str, Any], None] = lambda k, m: None,
                 on_fault: Optional[Callable[[str], None]] = None,
                 calibration: Optional[Dict[str, Any]] = None):
        self.cfg = merged_cfg(cfg)
        self.syr, self.read_sensor = syringe, read_sensor
        self.emit, self.on_fault = emit, on_fault
        lo, hi = sensor_range or (-1e9, 1e9)
        self.sensor_range = (float(lo), float(hi))
        self.p_max = min(float(self.cfg["limit_mbar"]), 0.95 * self.sensor_range[1])
        self.p_min = max(float(self.cfg["min_mbar"]), 0.95 * self.sensor_range[0])
        cal = calibration or {}
        self.offset = float(cal.get("sensor_offset_mbar", 0.0))
        self.est = GainEstimator(float(cal.get("gain_mbar_step", self.cfg["gain_init_mbar_step"])))
        if cal.get("gain_push_mbar_step") and cal.get("gain_pull_mbar_step"):
            self.est.g = {1: float(cal["gain_push_mbar_step"]), -1: float(cal["gain_pull_mbar_step"])}
            self.est.n = {1: 3, -1: 3}
        self.est.backlash = float(cal.get("backlash_steps", 0.0))
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.mode = "idle"                 # idle | hold | release | fault
        self.target = 0.0
        self.p_cmd = 0.0                   # ramped setpoint
        self.p = float("nan")              # filtered pressure (offset removed)
        self.p_raw = float("nan")
        self.p_1s = float("nan")            # 1 s mean (settling, display)
        self.t_last_sample = 0.0
        self.settled_since: Optional[float] = None
        self.fault_msg: Optional[str] = None
        self.t0: Optional[float] = None
        self.hist: deque = deque(maxlen=100000)     # (t, p, pos): ~1 h at 25 Hz, for statistics
        self.plot_buf: deque = deque(maxlen=6000)   # (t, p, target) for the live plot
        self.blocked: Optional[str] = None          # "empty" while a needed push is blocked at the end
        self._recent: deque = deque(maxlen=5)
        self._threads: List[threading.Thread] = []
        self._log_f = None
        self._log_w = None
        self.dir = os.path.join(run_dir, "pressure") if run_dir else None
        self.events: List[Dict[str, Any]] = []
        self._integral = 0.0
        self.state_path = self.cfg.get("state_file") or default_state_path()
        self.port = str(self.cfg.get("port")) if str(self.cfg.get("backend")).lower() != "dummy" else "dummy"
        self.restored_pos: Optional[int] = None
        self._saved_pos: Optional[int] = None

    # ---- lifecycle ----
    def open(self):
        self.syr.open()
        if self.port != "dummy":
            self.restored_pos = restore_position(self.syr, self.state_path, self.port)
        if self.dir:
            os.makedirs(self.dir, exist_ok=True)
            self._log_f = open(os.path.join(self.dir, "pressure.csv"), "a", newline="", encoding="utf-8")
            self._log_w = csv.writer(self._log_f, lineterminator="\n")
            if self._log_f.tell() == 0:
                self._log_w.writerow(["t_s", "p_mbar", "p_raw_mbar", "p_target_mbar", "p_cmd_mbar", "pos_steps",
                                      "mode", "g_push_mbar_step", "g_pull_mbar_step", "backlash_steps",
                                      "drift_mbar_s"])

    def start(self, t0: float):
        self.t0 = t0
        self.stop_flag.clear()
        for fn, name in ((self._sample_loop, "pressure-sampler"), (self._control_loop, "pressure-control")):
            th = threading.Thread(target=fn, name=name, daemon=True)
            th.start()
            self._threads.append(th)
        self._event("start", f"p_max {self.p_max:g} mbar, p_min {self.p_min:g}, protocol {self.syr.protocol}, "
                             f"offset {self.offset:g}")

    def close(self):
        if self.cfg.get("release_on_stop") and self.mode == "hold" and not math.isnan(self.p) \
                and abs(self.p) > 2 * float(self.cfg["tolerance_mbar"]):
            self.release()
            t_end = clock() + 20.0
            while self.mode == "release" and clock() < t_end:
                time.sleep(0.05)
        self.stop_flag.set()
        for th in self._threads:
            th.join(timeout=3.0)
        try:
            self.syr.stop()
            self.syr.set_hold(False)
        except Exception:
            pass
        try:
            self.syr.close()
        except Exception:
            pass
        self._event("stop", f"pos {self.syr.position()}")
        self._save_pos(force=True)
        if self._log_f:
            try:
                self._log_f.flush()
                os.fsync(self._log_f.fileno())
                self._log_f.close()
            except Exception:
                pass
        if self.dir:
            try:
                with open(os.path.join(self.dir, "events.json"), "w", encoding="utf-8") as f:
                    json.dump({"events": self.events, "learned": self.learned()}, f, indent=2)
            except Exception:
                pass

    # ---- commands (any thread) ----
    def set_target(self, mbar: float):
        mbar = float(mbar)
        if not (self.p_min <= mbar <= self.p_max):
            raise ValueError(f"{mbar:g} mbar is outside the allowed {self.p_min:g}..{self.p_max:g} mbar")
        with self.lock:
            if self.mode == "fault":
                raise RuntimeError(f"pressure controller in fault: {self.fault_msg}")
            if self.mode != "hold" or math.isnan(self.p_cmd):
                self.p_cmd = self.p if not math.isnan(self.p) else 0.0
            self.target = mbar
            self.mode = "hold"
            self.settled_since = None
            self._integral = 0.0
        self._event("target", f"{mbar:g} mbar")
        try:
            self.syr.set_hold(True)
        except Exception:
            pass

    def release(self):
        with self.lock:
            if self.mode == "fault":
                return
            self.target = 0.0
            self.mode = "release"
            self.settled_since = None
        self._event("release", "")

    def idle(self):
        with self.lock:
            if self.mode != "fault":
                self.mode = "idle"
        try:
            self.syr.stop()
        except Exception:
            pass

    def jog(self, steps: int) -> bool:
        if self.mode in ("hold", "release"):
            raise RuntimeError("stop the pressure control before jogging")
        if not self.syr.within(steps):
            raise RuntimeError("jog would leave the syringe range")
        return self.syr.move(steps)

    def zero_sensor(self, seconds: float = 2.0) -> float:
        """Vent to atmosphere first: the mean raw reading becomes the offset."""
        vals = []
        t_end = clock() + seconds
        while clock() < t_end:
            ok, v = self.read_sensor()
            if ok and v is not None:
                vals.append(float(v))
            time.sleep(0.02)
        if not vals:
            raise RuntimeError("no pressure readings")
        self.offset = sum(vals) / len(vals)
        self._event("zero_sensor", f"offset {self.offset:.3f} mbar")
        return self.offset

    def apply_calibration(self, cal: Dict[str, Any]):
        """Use calibrated values (sweep results, sensor offset) as the new anchors."""
        if cal.get("gain_push_mbar_step") and cal.get("gain_pull_mbar_step"):
            self.est.g = {1: float(cal["gain_push_mbar_step"]), -1: float(cal["gain_pull_mbar_step"])}
            self.est.n = {1: 3, -1: 3}
        if cal.get("backlash_steps") is not None:
            self.est.backlash = float(cal["backlash_steps"])
        if cal.get("sensor_offset_mbar") is not None:
            self.offset = float(cal["sensor_offset_mbar"])
        self._event("calibration", json.dumps({k: v for k, v in cal.items() if k != "points"}))

    def reset_fault(self):
        with self.lock:
            if self.mode == "fault":
                self.mode = "idle"
                self.fault_msg = None
        self._event("fault_reset", "")

    def set_position(self, steps: int = 0):
        """The operator emptied the syringe by hand (0) or put the plunger somewhere known."""
        if self.mode in ("hold", "release"):
            raise RuntimeError("stop the pressure control before setting the position")
        try:
            if steps == 0:
                self.syr.zero()
            else:
                self.syr.set_position(steps)
        except Exception:
            self.syr.set_position(steps)      # old firmware without ZERO reply: software only
        self.est.last_dir = 0
        self._event("set_position", str(steps))
        self._save_pos(force=True)

    def note(self, kind: str, detail: str = ""):
        """Add an entry to pressure/events.json (runner and UI actions)."""
        self._event(kind, detail)

    def steps_rate(self) -> float:
        """Pull rate (steps/s) the hold needs against the current drift, from the learned values."""
        return max(0.0, self.est.q) / max(0.02, self.est.gain(-1))

    def _save_pos(self, force: bool = False):
        if self.port == "dummy":
            return
        pos = self.syr.position()
        if force or pos != self._saved_pos:
            self._saved_pos = pos
            save_state(self.state_path, {"port": self.port, "pos": pos, "time": time.time()})

    # ---- status / statistics ----
    def settled(self) -> bool:
        s = self.settled_since
        return self.mode == "hold" and s is not None and clock() - s >= float(self.cfg["settle_s"])

    def wait_settled(self, timeout_s: float, stop_flag: Optional[threading.Event] = None) -> bool:
        t_end = clock() + timeout_s
        while clock() < t_end:
            if stop_flag is not None and stop_flag.is_set():
                return False
            if self.mode == "fault":
                return False
            if self.settled():
                return True
            time.sleep(0.05)
        return False

    def window_stats(self, t_from: float, t_to: float) -> Dict[str, Any]:
        """Pressure mean/sd and syringe movement between two run times (s)."""
        rows = [h for h in list(self.hist) if t_from <= h[0] <= t_to]
        if len(rows) < 2:
            return {"n": len(rows)}
        ps = [r[1] for r in rows]
        mean = sum(ps) / len(ps)
        sd = math.sqrt(sum((x - mean) ** 2 for x in ps) / (len(ps) - 1))
        dpos = rows[-1][2] - rows[0][2]
        dt = rows[-1][0] - rows[0][0]
        out = {"n": len(rows), "p_mean": mean, "p_sd": sd, "dpos_steps": dpos, "dt_s": dt,
               "steps_per_s": dpos / dt if dt > 0 else float("nan")}
        ulps = self.cfg.get("ul_per_step")
        if ulps:
            out["syringe_flow_ul_min"] = dpos / dt * float(ulps) * 60.0 if dt > 0 else float("nan")
        return out

    def remaining_steps(self, direction: int = -1) -> int:
        """Steps left before the margin in a direction (-1 = pulling, which absorbs pump flow)."""
        pos, m = self.syr.position(), int(self.cfg["pos_margin"])
        return (self.syr.pos_max - m - pos) if direction < 0 else (pos - self.syr.pos_min - m)

    def learned(self) -> Dict[str, Any]:
        return {"gain_push_mbar_step": round(self.est.g[1], 4), "gain_pull_mbar_step": round(self.est.g[-1], 4),
                "backlash_steps": round(self.est.backlash, 2), "drift_mbar_s": round(self.est.q, 4),
                "sensor_offset_mbar": round(self.offset, 4)}

    def status(self) -> Dict[str, Any]:
        with self.lock:
            st = {"mode": self.mode, "target": self.target, "p_cmd": self.p_cmd, "p": self.p, "p_1s": self.p_1s,
                  "fault": self.fault_msg, "p_max": self.p_max, "p_min": self.p_min}
        st.update(pos=self.syr.position(), settled=self.settled(), protocol=self.syr.protocol,
                  blocked=self.blocked, remaining_pull=self.remaining_steps(-1),
                  steps_per_s=round(self.steps_rate(), 2),
                  stale=(clock() - self.t_last_sample) > float(self.cfg["stale_s"]), **self.learned())
        return st

    # ---- threads ----
    def _event(self, kind: str, detail: str):
        t = (clock() - self.t0) if self.t0 is not None else 0.0
        self.events.append({"t_s": round(t, 4), "event": kind, "detail": detail})

    def _fault(self, msg: str):
        with self.lock:
            if self.mode == "fault":
                return
            self.mode = "fault"
            self.fault_msg = msg
        self._event("fault", msg)
        try:
            self.syr.stop()
        except Exception:
            pass
        self.emit("error", f"Pressure control: {msg}")
        if self.on_fault:
            try:
                self.on_fault(msg)
            except Exception:
                pass

    def _sample_loop(self):
        dt = 1.0 / max(1.0, float(self.cfg["log_hz"]))
        n_flush = 0
        last_1s: deque = deque(maxlen=max(1, int(round(1.0 / dt))))    # for the 1 s mean
        while not self.stop_flag.is_set():
            t_next = clock() + dt
            ok, raw = self.read_sensor()
            now = clock()
            if ok and raw is not None:
                raw = float(raw)
                self._recent.append(raw - self.offset)
                p = sorted(self._recent)[len(self._recent) // 2]          # median of the last 5
                t_s = now - self.t0
                self.hist.append((t_s, p, self.syr.position()))
                self.plot_buf.append((t_s, p, self.target if self.mode == "hold" else float("nan")))
                # Settled = the 1 s mean within tolerance (a stop-start v1 hold is a sawtooth)
                last_1s.append(p)
                p1s = sum(last_1s) / len(last_1s)
                with self.lock:
                    self.p, self.p_raw, self.p_1s = p, raw, p1s
                    self.t_last_sample = now
                    tol = float(self.cfg["tolerance_mbar"])
                    if self.mode == "hold" and abs(p1s - self.target) <= tol and abs(self.p_cmd - self.target) < 1e-6:
                        if self.settled_since is None:
                            self.settled_since = now
                    else:
                        self.settled_since = None
                lo, hi = self.sensor_range
                if raw >= 0.99 * hi or raw <= 0.99 * lo:
                    self._fault(f"pressure sensor saturated ({raw:.1f} mbar): pressure unknown")
                elif p > self.p_max:
                    self._fault(f"pressure {p:.1f} mbar above the {self.p_max:g} mbar limit")
                elif p < self.p_min:
                    self._fault(f"pressure {p:.1f} mbar below the {self.p_min:g} mbar limit")
                if self._log_w:
                    with self.lock:
                        row = [f"{t_s:.4f}", f"{p:.3f}", f"{raw:.3f}", f"{self.target:.2f}", f"{self.p_cmd:.2f}",
                               self.syr.position(), self.mode, f"{self.est.g[1]:.4f}", f"{self.est.g[-1]:.4f}",
                               f"{self.est.backlash:.2f}", f"{self.est.q:.4f}"]
                    self._log_w.writerow(row)
                    n_flush += 1
                    if n_flush >= int(1.0 / dt):
                        n_flush = 0
                        self._log_f.flush()
                        self._save_pos()
            elif now - self.t_last_sample > float(self.cfg["stale_s"]) and self.t_last_sample > 0:
                if self.mode in ("hold", "release"):
                    self._fault(f"no pressure reading for {self.cfg['stale_s']:g} s")
            time.sleep(max(0.0, t_next - clock()))

    def _wait_sample(self, timeout: float = 1.0) -> float:
        t0 = self.t_last_sample
        t_end = clock() + timeout
        while self.t_last_sample == t0 and clock() < t_end and not self.stop_flag.is_set():
            time.sleep(0.005)
        return self.p

    def _control_loop(self):
        if self.syr.protocol == "v2":
            return self._control_v2()
        idle_p, idle_t = None, None
        t_ramp = clock()
        kp_eff, last_err = float(self.cfg["kp"]), None
        t_cycle = float(self.cfg["move_settle_s"]) + 0.15      # learned below
        t_last_move = None
        t_integ = None
        while not self.stop_flag.is_set():
            with self.lock:
                mode = self.mode
            if mode == "fault":
                self._fault_recovery()
                time.sleep(0.1)
                continue
            if mode == "idle" or math.isnan(self.p):
                idle_p = idle_t = None
                t_ramp = clock()
                time.sleep(0.05)
                continue
            p = self._wait_sample(0.5)
            now = clock()
            # Ramp the setpoint at rate_mbar_s (real elapsed time, capped)
            with self.lock:
                rate = float(self.cfg["rate_mbar_s"])
                step = rate * min(0.5, now - t_ramp)
                t_ramp = now
                if abs(self.target - self.p_cmd) <= step:
                    self.p_cmd = self.target
                else:
                    self.p_cmd += math.copysign(step, self.target - self.p_cmd)
                p_cmd, tol = self.p_cmd, float(self.cfg["tolerance_mbar"])
            if mode == "release" and abs(p) <= tol and p_cmd == 0.0:
                self._event("released", f"{p:.2f} mbar")
                self.idle()
                continue
            # With a steady drift q the stop-start hold is a sawtooth: each move must cancel the
            # drift until the next correction (feed-forward q x T, T = measured cycle time) on
            # top of a proportional share of the error. The reading is taken at the end of the
            # drift, so centre the sawtooth with half a cycle's drift.
            q, T = self.est.q, t_cycle
            err = p_cmd - p + 0.5 * q * T
            if abs(err) <= 0.5 * tol and abs(q * T) <= 0.5 * tol:
                if idle_p is not None:
                    self.est.observe_idle(p - idle_p, now - idle_t)
                idle_p, idle_t = p, now
                t_last_move = None
                time.sleep(0.1)
                continue
            idle_p = idle_t = None
            # Damp if the error keeps flipping sign without shrinking (overshoot)
            if last_err is not None and (last_err > 0) != (err > 0) and abs(err) > 0.7 * abs(last_err):
                kp_eff = max(0.2, kp_eff * 0.7)
            else:
                kp_eff = min(float(self.cfg["kp"]), kp_eff * 1.05)
            last_err = err
            # Integral action removes what the model gets wrong (gain error, lost steps); only
            # near the setpoint, so ramps don't wind it up
            dt_i = min(1.0, now - (t_integ if t_integ else now))
            t_integ = now
            if abs(err) < 5 * tol and p_cmd == self.target:
                self._integral = max(-30.0, min(30.0, self._integral + err * dt_i))
            delta = kp_eff * err + float(self.cfg["ki"]) * self._integral - q * T   # wanted from this move
            d = 1 if delta > 0 else -1
            g = self.est.gain(d)
            if abs(delta) < 0.5 * g:                  # less than half a step: wait
                time.sleep(0.05)
                continue
            if t_last_move is not None:
                t_cycle = 0.7 * t_cycle + 0.3 * min(2.0, now - t_last_move)
            t_last_move = now
            n = int(round(abs(delta) / g))
            n = max(1, min(n, int(self.cfg["max_chunk"])))
            if self.est.last_dir not in (0, d):
                n += int(round(self.est.backlash))
            steps = d * n
            if not self.syr.within(steps, int(self.cfg["pos_margin"])):
                avail = self.remaining_steps(-1 if d < 0 else 1)
                if avail <= 0:
                    if mode == "release":
                        # Can't pull further: leave the rest to the operator (vent by hand)
                        self._event("release_blocked", f"{p:.1f} mbar left at pos {self.syr.position()}")
                        self.idle()
                    elif d > 0:
                        # Empty end: can't raise P. Safe (P stays below the target); wait, the
                        # pump may raise it. The PQ runner treats a lasting block as "above shut-off".
                        self.blocked = "empty"
                        time.sleep(0.1)
                    else:
                        self._fault("syringe range used up (full): clear the syringe by hand "
                                    "before holding pressure")
                    continue
                steps = d * min(n, avail)
            p0, t0 = p, clock()
            self.blocked = None
            ok = self.syr.move(steps)
            if not ok:
                self._fault(f"syringe did not confirm a move of {steps} steps"
                            + (f" ({self.syr.last_error})" if getattr(self.syr, "last_error", None) else ""))
                continue
            time.sleep(float(self.cfg["move_settle_s"]))
            p1 = self._wait_sample(0.5)
            res = self.est.observe_move(steps, p1 - p0, clock() - t0)
            if res["stuck"]:
                self._event("stuck", f"move {steps} steps changed P by {p1 - p0:+.2f} mbar at pos "
                                     f"{self.syr.position()}")
                if self.est.stuck_in_row >= 4:
                    self._fault(f"syringe appears stuck at position {self.syr.position()} "
                                f"({self.est.stuck_in_row} moves without effect)")

    def _control_v2(self):
        """Velocity control: v = (kp*err + ki*integral)/g + feed-forward against the drift."""
        last_v, last_p, last_t = 0.0, None, None
        t_sent = 0.0
        while not self.stop_flag.is_set():
            with self.lock:
                mode = self.mode
            if mode == "fault":
                self._fault_recovery()
                time.sleep(0.1)
                continue
            if mode == "idle" or math.isnan(self.p):
                if last_v:
                    self.syr.set_velocity(0)
                    last_v = 0.0
                time.sleep(0.05)
                continue
            p = self._wait_sample(0.5)
            now = clock()
            dt = (now - last_t) if last_t else 0.04
            with self.lock:
                step = float(self.cfg["rate_mbar_s"]) * dt
                self.p_cmd = self.target if abs(self.target - self.p_cmd) <= step else \
                    self.p_cmd + math.copysign(step, self.target - self.p_cmd)
                p_cmd, tol = self.p_cmd, float(self.cfg["tolerance_mbar"])
            if mode == "release" and abs(p) <= tol and p_cmd == 0.0:
                self.syr.set_velocity(0)
                last_v = 0.0
                self._event("released", f"{p:.2f} mbar")
                self.idle()
                continue
            err = p_cmd - p
            self._integral = max(-200.0, min(200.0, self._integral + err * dt))
            d = 1 if err > 0 else -1
            g = self.est.gain(d)
            v = (float(self.cfg["kp"]) * 2.0 * err + float(self.cfg["ki"]) * self._integral - self.est.q) / g
            vmax = float(self.cfg["vmax_sps"])
            v = max(-vmax, min(vmax, v))
            if v and not self.syr.within(int(math.copysign(5, v)), int(self.cfg["pos_margin"])):
                direction = 1 if v > 0 else -1
                v = 0.0
                if self.remaining_steps(-1 if direction < 0 else 1) <= 0:
                    if mode == "release":
                        self.syr.set_velocity(0)
                        last_v = 0.0
                        self._event("release_blocked", f"{p:.1f} mbar left at pos {self.syr.position()}")
                        self.idle()
                        continue
                    if direction > 0:
                        self.blocked = "empty"
                    else:
                        self._fault("syringe range used up (full): clear the syringe by hand "
                                    "before holding pressure")
            elif v:
                self.blocked = None
            # Resend at least every 0.5 s: the v2 firmware stops on its own after 2 s of silence
            if abs(v - last_v) >= 2.0 or (v and now - t_sent > 0.5):
                self.syr.set_velocity(v)
                last_v, t_sent = v, now
            if last_p is not None and abs(last_v) < 1.0:
                self.est.observe_idle(p - last_p, now - last_t)
            last_p, last_t = p, now
            time.sleep(0.04)

    def _fault_recovery(self):
        """In a fault: if the pressure is too high, still try to let it down."""
        p = self.p
        if not math.isnan(p) and p > 0.9 * self.p_max:
            n = int(min(50, max(5, (p - 0.5 * self.p_max) / max(0.05, self.est.gain(-1)))))
            if self.syr.within(-n):
                try:
                    self.syr.move(-n)
                except Exception:
                    pass


# ---------------- calibration ----------------
def calibration_path(base_dir: str) -> str:
    return os.path.join(base_dir, "pressure_cal.json")


def load_calibration(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_calibration(path: str, cal: Dict[str, Any]):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)
    os.replace(tmp, path)


def calib_gain_sweep(ctl: PressureController, step: int = 10, p_top: float = 60.0, max_moves: int = 60,
                     progress: Callable[[str], None] = lambda s: None, first: Optional[int] = None) -> Dict[str, Any]:
    """Pump OFF, line closed. Move `step` steps at a time away from the start pressure by p_top,
    then back. Pushes first (pressure up) when the syringe has room to push, else pulls first
    (pressure down, at most 80 % of the lower limit). Returns gain per direction (mbar/step),
    backlash (steps, from the first move after the reversal) and the raw points."""
    if ctl.mode in ("hold", "release"):
        raise RuntimeError("stop the pressure control first")
    margin = int(ctl.cfg["pos_margin"])
    if first is None:
        need = int(p_top / max(0.05, ctl.est.gain(1))) + 2 * step
        first = 1 if ctl.remaining_steps(1) >= need else -1
    span = p_top if first > 0 else min(p_top, 0.8 * abs(ctl.p_min))
    pts = []
    p_start = ctl._wait_sample(1.0)
    for direction in (first, -first):
        moved = 0
        for _ in range(max_moves):
            if ctl.stop_flag.is_set():
                break
            p0 = ctl._wait_sample(1.0)
            if direction == first and (p0 - p_start) * first >= span:
                break
            if direction != first and (p0 - p_start) * first <= 1.0:
                break
            if not ctl.syr.within(direction * step, margin):
                progress("syringe range end reached")
                break
            ctl.syr.move(direction * step)
            time.sleep(0.4)
            p1 = ctl._wait_sample(1.0)
            pts.append({"dir": direction, "steps": step, "p0": p0, "p1": p1, "pos": ctl.syr.position()})
            moved += 1
            progress(f"{'push' if direction > 0 else 'pull'} {moved}: {p1:.1f} mbar")
        if ctl.stop_flag.is_set():
            break
    out: Dict[str, Any] = {"sweep_first": "push" if first > 0 else "pull"}
    for d, key in ((1, "gain_push_mbar_step"), (-1, "gain_pull_mbar_step")):
        moves = [p for p in pts if p["dir"] == d]
        body = moves if d == first else moves[1:]          # skip the first move after the reversal
        gains = sorted((p["p1"] - p["p0"]) * d / p["steps"] for p in body if (p["p1"] - p["p0"]) * d > 0)
        out[key] = round(gains[len(gains) // 2], 4) if gains else None
    back = [p for p in pts if p["dir"] == -first]
    g_back = out["gain_push_mbar_step" if -first > 0 else "gain_pull_mbar_step"]
    if back and g_back:
        eff = (back[0]["p1"] - back[0]["p0"]) * (-first) / g_back
        out["backlash_steps"] = round(max(0.0, back[0]["steps"] - eff), 1)
    out["points"] = pts
    return out


def calib_leak_test(ctl: PressureController, p_test: float = 80.0, seconds: float = 60.0,
                    progress: Callable[[str], None] = lambda s: None) -> Dict[str, Any]:
    """Pump OFF. Bring the line to p_test, stop the syringe, watch the decay."""
    ctl.set_target(p_test)
    if not ctl.wait_settled(60.0):
        ctl.idle()
        raise RuntimeError(f"could not settle at {p_test:g} mbar for the leak test")
    ctl.idle()
    t0, p0 = clock(), ctl._wait_sample(1.0)
    while clock() - t0 < seconds and not ctl.stop_flag.is_set():
        time.sleep(1.0)
        progress(f"leak test {clock() - t0:.0f}/{seconds:.0f} s: {ctl.p:.2f} mbar")
    p1 = ctl._wait_sample(1.0)
    rate = (p1 - p0) / (clock() - t0)
    return {"leak_p_start_mbar": round(p0, 3), "leak_p_end_mbar": round(p1, 3), "leak_mbar_s": round(rate, 5)}


def calib_step_response(ctl: PressureController, targets=(25.0, 50.0, 100.0), timeout_s: float = 60.0,
                        progress: Callable[[str], None] = lambda s: None) -> Dict[str, Any]:
    """Pump OFF. Settle at each target; report time to settle and overshoot."""
    res = []
    for tgt in targets:
        t0 = clock()
        p_prev = ctl._wait_sample(1.0)
        ctl.set_target(tgt)
        peak = p_prev
        while clock() - t0 < timeout_s and not ctl.settled() and ctl.mode != "fault":
            p = ctl._wait_sample(0.5)
            peak = max(peak, p) if tgt >= p_prev else min(peak, p)
        ok = ctl.settled()
        res.append({"target": tgt, "settled": ok, "t_settle_s": round(clock() - t0, 2),
                    "overshoot_mbar": round((peak - tgt) if tgt >= p_prev else (tgt - peak), 2)})
        progress(f"{tgt:g} mbar: {'settled' if ok else 'NOT settled'} in {clock() - t0:.1f} s")
    ctl.release()
    return {"step_response": res}
