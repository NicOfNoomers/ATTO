"""Pinch-valve pressure control: the syringe motor squeezes a flex tube in a flow loop (NM, 23 Nov 2026).

Why: the syringe method adds volume to a closed line. The syringe fills and needs clears, releases pull
(suction: the -62 mbar undershoot that degraded 261030-1 on 2 Nov), and the flow sensor sees the syringe,
not the pump. NM's new loop keeps the pump flowing and throttles it instead:

    reservoir (outlet low) -> pump -> flow sensor -> pressure sensor -> FLEX TUBE (squeezed) -> reservoir (inlet high)

The same stepper, driver and firmware (syringe_v2) push the plunger onto the tube. More squeeze = more
resistance = more back-pressure: P ~ Q x R(squeeze). Fully closed = blocked outlet (maximum pressure).
Tube: ID 1/16 in (1.59 mm), thin wall. No mechanical limits or switch: the pressure sensor shows contact.

Sign convention (as the syringe): a + move pushes in (squeezes); squeeze = start position - position.
Steps in the config are FULL steps (x microsteps for the motor), as everywhere in pressure_control.

PinchController:
  - sampler thread: pressure at log_hz (offset = zero_sensor() with the pump off), filtered p and p_1s,
    pinch/pressure.csv, safety on every sample: above p_max (limit_mbar, at most 95 % of the sensor range),
    a saturated or stale sensor -> fault: the motor retracts retract_steps, on_fault(msg) (the runner stops).
  - find_touch(): with the pump running, retract start_retract_steps, take the baseline, then squeeze in
    touch_step steps until the pressure rises touch_mbar above it (twice in a row): touch = the squeeze one
    step before. Abort after max_search_steps.
  - go_open(): to touch - open_back_steps (or back to the start before a touch is known).
  - set_squeeze(s): to touch + s full steps (position mode), never past max_squeeze_steps.
  - hold(target): closed loop at quasi-steady state. Each hold_dt: error -> step = 0.7 x error / g, with g
    (mbar per full step) learned from the last moves (the gain grows steeply near closure); clamp to
    max_step; settled after hold_settle_s within tol. Returns why it stopped (settled / timeout / at_cap /
    cannot_reach: squeezing no longer raises the pressure).
  - close_until(stop_mbar): squeeze step by step until stop_mbar, a plateau (fully closed: the pump's
    blocked-outlet pressure), or the cap.
PinchPlant: simulated loop for tests (pump Q0(f) x (1 - P / Pmax(f)), slit resistance ~ 1/gap^3,
compliance lag, slow tube relaxation, sensor noise), usable with pressure_control.DummySyringe.
"""
from __future__ import annotations

import csv
import json
import math
import os
import random
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

clock = time.perf_counter

PINCH_DEFAULTS: Dict[str, Any] = {
    "touch_mbar": 3.0,            # rise above the open-tube baseline that marks contact
    "touch_step": 2,              # full steps per probe while searching
    "touch_settle_s": 0.4,
    "start_retract_steps": 40,    # full steps retracted before the search (the plunger may sit on the tube)
    "max_search_steps": 3000,     # full steps of search before giving up
    "max_squeeze_steps": 400,     # full steps past touch: hard cap (tube protection)
    "open_back_steps": 10,        # open position = touch - this
    "retract_steps": 40,          # full steps retracted on a fault
    "hold_tol_mbar": 2.0,
    "hold_settle_s": 3.0,
    "hold_dt": 0.6,               # s between corrections
    "max_step": 6,                # full steps per correction
    "gain_init": 2.0,             # mbar per full step before anything is learned
    "gain_min": 0.05,
    "plateau_mbar": 0.5,          # close_until: less than this over plateau_n steps = fully closed
    "plateau_n": 8,
    "plateau_min_rise_mbar": 20.0,  # close_until: a plateau counts only this far above its start
    "log_hz": 25.0,
    "stale_s": 2.0,
    "move_timeout_s": 10.0,
}


def pinch_cfg(pcfg: Dict[str, Any]) -> Dict[str, Any]:
    return {**PINCH_DEFAULTS, **{k: v for k, v in (pcfg.get("pinch") or {}).items()}}


class PinchController:
    def __init__(self, cfg: Dict[str, Any], syr, read_sensor: Callable[[], Tuple[bool, float]],
                 sensor_range: Tuple[float, float] = (-344.74, 344.74), run_dir: Optional[str] = None,
                 on_fault: Callable[[str], None] = lambda msg: None,
                 emit: Callable[[str, Any], None] = lambda k, m: None, microsteps: int = 1):
        self.cfg = cfg                                  # merged pressure config (motor counts)
        self.pc = pinch_cfg(cfg)
        self.syr, self.read_sensor, self.sensor_range = syr, read_sensor, tuple(sensor_range)
        self.run_dir, self.on_fault, self.emit = run_dir, on_fault, emit
        self.m = max(1, int(microsteps))
        self.p_max = min(float(cfg.get("limit_mbar", 300.0)), 0.95 * float(self.sensor_range[1]))
        self.p_min = float(cfg.get("min_mbar", -60.0))
        self.lock = threading.Lock()
        self.mlock = threading.RLock()                  # one motor move at a time (sampler retracts too)
        self.mode = "idle"
        self.target = float("nan")
        self.fault_msg: Optional[str] = None
        self.offset = 0.0
        self.p = float("nan")
        self.p_1s = float("nan")
        self.raw = float("nan")
        self.blocked = None                             # syringe-controller field the dashboard reads
        self.sensor_obj = None
        self.t0: Optional[float] = None
        self.t_last_sample = clock()
        self.start_pos: Optional[int] = None
        self.touch: Optional[int] = None                # squeeze (motor counts) at contact
        self.gain = float(self.pc["gain_init"])         # mbar per FULL step
        self.pmap: List[Tuple[float, float]] = []       # (steps past touch, mbar) from the last closing sweep
        self.plot_buf: deque = deque(maxlen=6000)       # (t, p, target) for the live plot
        self.hist: deque = deque(maxlen=20000)          # (t, p, squeeze)
        self._win: deque = deque(maxlen=200)
        self.events: List[Dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._csv = None
        self._writer = None

    # ---- life cycle ----
    def open(self):
        self.syr.open()
        self.start_pos = self.syr.position()
        if self.run_dir:
            d = os.path.join(self.run_dir, "pinch")
            os.makedirs(d, exist_ok=True)
            self._csv = open(os.path.join(d, "pressure.csv"), "w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._csv)
            self._writer.writerow(["t_s", "p_mbar", "p_raw_mbar", "target_mbar", "squeeze_steps", "mode"])

    def start(self, t0: float):
        self.t0 = t0
        self._event("start", f"p_max {self.p_max:g} mbar, microsteps {self.m}, start position {self.start_pos}")
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def close(self):
        """Open the pinch (pressure off the loop), stop the sampler, close files and the motor."""
        try:
            if self.mode != "fault":
                self.go_open()
        except Exception:
            pass
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self.syr.close()
        finally:
            if self._csv is not None:
                self._csv.close()
            if self.run_dir:
                try:
                    with open(os.path.join(self.run_dir, "pinch", "events.json"), "w", encoding="utf-8") as f:
                        json.dump(self.events, f, indent=1)
                except Exception:
                    pass

    def zero_sensor(self, seconds: float = 2.0) -> float:
        """With the pump OFF and the pinch open: the mean raw reading becomes the offset."""
        vals = []
        t_end = clock() + seconds
        while clock() < t_end:
            ok, v = self.read_sensor()
            if ok and v is not None and v == v:
                vals.append(float(v))
            time.sleep(0.02)
        if not vals:
            raise RuntimeError("no pressure readings")
        self.offset = sum(vals) / len(vals)
        self._event("zero_sensor", f"offset {self.offset:.3f} mbar")
        return self.offset

    # ---- state ----
    def squeeze(self) -> int:
        return (self.start_pos or 0) - self.syr.position()

    def squeeze_full(self) -> float:
        return self.squeeze() / self.m

    def past_touch_full(self) -> Optional[float]:
        return None if self.touch is None else (self.squeeze() - self.touch) / self.m

    def note(self, kind: str, detail: str = ""):
        self._event(kind, detail)

    def _event(self, kind: str, detail: str):
        t = (clock() - self.t0) if self.t0 is not None else 0.0
        self.events.append({"t_s": round(t, 3), "event": kind, "detail": detail})

    def status(self) -> Dict[str, Any]:
        return {"mode": self.mode, "target": self.target, "p": self.p, "p_1s": self.p_1s, "fault": self.fault_msg,
                "p_max": self.p_max, "p_min": self.p_min, "pos": self.syr.position(),
                "squeeze_steps": round(self.squeeze_full(), 2), "touch_steps": None if self.touch is None
                else round(self.touch / self.m, 2), "gain_mbar_step": round(self.gain, 3), "pinch": True,
                "stale": (clock() - self.t_last_sample) > float(self.pc["stale_s"])}

    def settled(self) -> bool:
        return False

    def window_stats(self, t_from: float, t_to: float) -> Dict[str, Any]:
        rows = [h for h in list(self.hist) if t_from <= h[0] <= t_to]
        if len(rows) < 2:
            return {"n": len(rows)}
        ps = [r[1] for r in rows]
        mean = sum(ps) / len(ps)
        sd = math.sqrt(sum((x - mean) ** 2 for x in ps) / (len(ps) - 1))
        return {"n": len(rows), "p_mean": mean, "p_sd": sd, "p_start": ps[0], "p_end": ps[-1],
                "squeeze_steps": rows[-1][2] / self.m}

    # ---- sampler and safety ----
    def _sample_loop(self):
        dt = 1.0 / max(1.0, float(self.pc["log_hz"]))
        ema = None
        lo, hi = self.sensor_range
        while not self._stop.is_set():
            t = clock()
            try:
                ok, raw = self.read_sensor()
            except Exception as e:  # noqa: BLE001
                ok, raw = False, None
                self._event("sensor_error", str(e))
            if ok and raw is not None and raw == raw:
                raw = float(raw)
                p = raw - self.offset
                ema = p if ema is None else 0.5 * ema + 0.5 * p
                self.t_last_sample = t
                with self.lock:
                    self.raw, self.p = raw, ema
                    self._win.append((t, p))
                    while self._win and t - self._win[0][0] > 1.0:
                        self._win.popleft()
                    self.p_1s = sum(x[1] for x in self._win) / len(self._win)
                t_s = t - (self.t0 or t)
                sq = self.squeeze()
                self.hist.append((t_s, p, sq))
                self.plot_buf.append((t_s, p, self.target if self.mode == "hold" else float("nan")))
                if self._writer is not None:
                    self._writer.writerow([f"{t_s:.4f}", f"{p:.3f}", f"{raw:.3f}",
                                           "" if self.target != self.target else f"{self.target:.2f}",
                                           f"{sq / self.m:.2f}", self.mode])
                if self.mode != "fault":
                    if raw >= 0.99 * hi or raw <= 0.99 * lo:
                        self._fault(f"pressure sensor saturated ({raw:.1f} mbar raw)")
                    elif p > self.p_max:
                        self._fault(f"pressure {p:.1f} mbar above the {self.p_max:g} mbar limit")
            elif self.mode != "fault" and t - self.t_last_sample > float(self.pc["stale_s"]):
                self._fault(f"no pressure reading for {t - self.t_last_sample:.1f} s")
            time.sleep(max(0.0, dt - (clock() - t)))

    def _fault(self, msg: str):
        with self.lock:
            if self.mode == "fault":
                return
            self.mode, self.fault_msg = "fault", msg
        self._event("fault", msg)
        try:
            with self.mlock:
                self.syr.move(-int(self.pc["retract_steps"]) * self.m, timeout_s=float(self.pc["move_timeout_s"]))
        except Exception as e:  # noqa: BLE001
            self._event("fault_retract_failed", str(e))
        try:
            self.on_fault(msg)
        except Exception:
            pass

    # ---- motion ----
    def _check(self):
        if self.mode == "fault":
            raise RuntimeError(f"pinch control in fault: {self.fault_msg}")

    def _move_full(self, n_full: float) -> int:
        """Relative move in full steps (+ = squeeze), capped at max_squeeze_steps past touch. Returns counts moved."""
        self._check()
        n = int(round(n_full * self.m))
        if n > 0 and self.touch is not None:
            cap = self.touch + int(self.pc["max_squeeze_steps"]) * self.m
            n = min(n, cap - self.squeeze())
        if n == 0:
            return 0
        with self.mlock:
            ok = self.syr.move(n, timeout_s=float(self.pc["move_timeout_s"]))
        if not ok:
            self._event("move_failed", f"{n} counts")
        return n

    def _goto_squeeze(self, counts: int):
        self._move_full((counts - self.squeeze()) / self.m)

    def _p_mean(self, seconds: float) -> float:
        t_end = clock() + seconds
        vals = []
        while clock() < t_end:
            p = self.p
            if p == p:
                vals.append(p)
            time.sleep(0.04)
        return sum(vals) / len(vals) if vals else float("nan")

    def find_touch(self, stopped: Callable[[], bool] = lambda: False) -> Optional[float]:
        """Pump running. Returns the touch squeeze in full steps (None if stopped or not found)."""
        pc = self.pc
        self.mode = "search"
        self._move_full(-float(pc["start_retract_steps"]))
        time.sleep(1.0)
        p0 = self._p_mean(1.5)
        hits, moved = 0, 0
        while moved < int(pc["max_search_steps"]) and not stopped():
            self._move_full(float(pc["touch_step"]))
            moved += int(pc["touch_step"])
            p = self._p_mean(float(pc["touch_settle_s"]))
            hits = hits + 1 if p - p0 >= float(pc["touch_mbar"]) else 0
            if hits >= 2:
                self.touch = self.squeeze() - 2 * int(pc["touch_step"]) * self.m
                self._event("touch", f"at squeeze {self.touch / self.m:.1f} full steps; baseline {p0:.1f} mbar, "
                                     f"now {p:.1f} mbar")
                self.mode = "idle"
                return self.touch / self.m
        self.mode = "idle"
        if not stopped():
            self._event("touch_not_found", f"{moved} full steps searched, baseline {p0:.1f} mbar")
        return None

    def go_open(self):
        self._check()
        self.mode = "open"
        self.target = float("nan")
        if self.touch is not None:
            self._goto_squeeze(self.touch - int(self.pc["open_back_steps"]) * self.m)
        else:
            self._goto_squeeze(0)
        self.mode = "idle"

    def set_squeeze(self, steps_past_touch: float):
        if self.touch is None:
            raise RuntimeError("set_squeeze needs the touch point (find_touch first)")
        s = max(-float(self.pc["open_back_steps"]), min(float(steps_past_touch), float(self.pc["max_squeeze_steps"])))
        self.mode = "position"
        self._goto_squeeze(self.touch + int(round(s * self.m)))

    def _map_lookup(self, target: float) -> Tuple[Optional[float], Optional[float]]:
        """From the last closing sweep (pmap: (steps past touch, mbar)): the squeeze where the pressure first
        reaches target, and the local gain there (mbar per full step)."""
        pts = sorted(self.pmap)
        for (s0, p0), (s1, p1) in zip(pts, pts[1:]):
            if p0 <= target <= p1 and p1 > p0 and s1 > s0:
                return s0 + (target - p0) * (s1 - s0) / (p1 - p0), (p1 - p0) / (s1 - s0)
        return None, None

    def _map_gain(self, s_past: float) -> Optional[float]:
        pts = sorted(self.pmap)
        if len(pts) < 2:
            return None
        for (s0, p0), (s1, p1) in zip(pts, pts[1:]):
            if s0 <= s_past <= s1 and s1 > s0:
                return (p1 - p0) / (s1 - s0)
        (s0, p0), (s1, p1) = (pts[0], pts[1]) if s_past < pts[0][0] else (pts[-2], pts[-1])
        return (p1 - p0) / (s1 - s0) if s1 > s0 else None

    def hold(self, target: float, timeout_s: float = 60.0, stopped: Callable[[], bool] = lambda: False) -> str:
        """Squeeze to hold target mbar. Returns settled / timeout / at_cap / cannot_reach / stopped / fault.
        With a map from a closing sweep (close_until, same frequency) it first moves to the mapped squeeze and
        takes the local gain from the map; a target above the mapped maximum is cannot_reach at once. Without a
        map it learns the gain from its own moves (only from moves big enough to rise above the noise)."""
        pc = self.pc
        if target > self.p_max - 5.0:
            raise ValueError(f"target {target:g} mbar is above the limit {self.p_max - 5:g} mbar")
        tol, dt = float(pc["hold_tol_mbar"]), float(pc["hold_dt"])
        gmin, smax = float(pc["gain_min"]), float(pc["max_step"])
        self.target = float(target)
        if self.pmap and self.touch is not None:
            p_top = max(pp for _, pp in self.pmap)
            if target > p_top + tol:
                self._event("hold_cannot_reach", f"{target:g} mbar is above the mapped maximum {p_top:.1f} mbar")
                return "cannot_reach"
            s_goal, _ = self._map_lookup(target)
            if s_goal is not None:
                self.set_squeeze(s_goal)
        self.mode = "hold"
        t_end = clock() + timeout_s
        in_tol_since = None
        last: Optional[Tuple[float, float]] = None
        self._p_mean(2 * dt)                              # let the last move settle
        while clock() < t_end:
            if stopped():
                return "stopped"
            if self.mode == "fault":
                return "fault"
            p = self._p_mean(dt)
            if p != p:
                continue
            s_now = self.squeeze_full()
            e = target - p
            if abs(e) <= tol:
                in_tol_since = in_tol_since or clock()
                if clock() - in_tol_since >= float(pc["hold_settle_s"]):
                    return "settled"
                continue
            in_tol_since = None
            g = self._map_gain(self.past_touch_full()) if (self.pmap and self.touch is not None) else None
            if g is None:
                if last is not None and abs(s_now - last[0]) >= 1.0 and abs(p - last[1]) > 3 * tol:
                    self.gain = 0.7 * self.gain + 0.3 * abs((p - last[1]) / (s_now - last[0]))
                g = self.gain
            step = max(-smax, min(smax, 0.6 * e / max(g, gmin)))
            if abs(step) < 1.0 / self.m:
                step = math.copysign(1.0 / self.m, e)
            last = (s_now, p)
            moved = self._move_full(step)
            if moved == 0 and e > 0:
                self._event("hold_at_cap", f"{target:g} mbar: {p:.1f} mbar at the squeeze cap")
                return "at_cap"
            self.mode = "hold"
        return "timeout"

    def close_until(self, stop_mbar: float, step_full: float = 1.0,
                    stopped: Callable[[], bool] = lambda: False) -> Tuple[float, str]:
        """Squeeze step by step until stop_mbar, a plateau (fully closed: the blocked-outlet pressure) or the cap;
        records the map (steps past touch, mbar) that hold() uses. A plateau only counts once the pressure has
        risen plateau_min_rise_mbar above where the sweep started (near touch the tube adds little resistance,
        which must not read as 'closed'). Returns (pressure, reason)."""
        pc = self.pc
        stop_mbar = min(stop_mbar, self.p_max - 5.0)
        self.mode = "closing"
        self.pmap = []
        hist: deque = deque(maxlen=int(pc["plateau_n"]) + 1)
        p_first = None
        while not stopped():
            if self.mode == "fault":
                return self.p_1s, "fault"
            p = self._p_mean(float(pc["touch_settle_s"]))
            p_first = p if p_first is None else p_first
            pt = self.past_touch_full()
            if pt is not None:
                self.pmap.append((pt, p))
            hist.append(p)
            if p >= stop_mbar:
                self.mode = "position"
                return p, "stop"
            if (len(hist) == hist.maxlen and hist[-1] - hist[0] < float(pc["plateau_mbar"])
                    and p - p_first >= float(pc["plateau_min_rise_mbar"])):
                self.mode = "position"
                return p, "plateau"
            if self._move_full(step_full) == 0:
                self.mode = "position"
                return p, "at_cap"
        return self.p_1s, "stopped"


class PinchPlant:
    """Simulated loop: pump -> sensors -> squeezed flex tube -> reservoir (for tests; no hardware).
    Pump: Q0(f) (1 - P/Pmax(f)), never negative. Tube: gap h = h0 - k (x - x_touch) for x past touch;
    resistance R = R_line + R_tube (h0/h)^3, closed (h <= 0) = tiny leak. The pressure follows the operating
    point with a lag tau_s; the tube relaxes (creep_frac of the squeeze, creep_tau_s). Usable with
    pressure_control.DummySyringe (step(), pos, read_sensor(), range, pressure(), pump_flow_ul_min())."""

    def __init__(self, get_freq: Callable[[], float] = lambda: 0.0, x_touch: int = 600, k_mm_per_count: float = 0.0014,
                 h0_mm: float = 1.59, r_line: float = 0.05, r_tube: float = 0.02, q0: float = 200.0, pmax: float = 250.0,
                 tau_s: float = 0.6, creep_frac: float = 0.05, creep_tau_s: float = 20.0, noise_mbar: float = 0.3,
                 sensor_range=(-344.74, 344.74), seed: int = 3, pos_max: int = 2470 * 16):
        self.get_freq = get_freq
        self.x_touch, self.k, self.h0 = x_touch, k_mm_per_count, h0_mm
        self.r_line, self.r_tube, self.q0, self.pmax = r_line, r_tube, q0, pmax   # R in mbar per (uL/min)
        self.tau, self.cf, self.ct, self.noise = tau_s, creep_frac, creep_tau_s, noise_mbar
        self.range = tuple(sensor_range)
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.x = 0                         # squeeze (motor counts) relative to the start
        self.x_relax = 0.0                 # part of the squeeze the tube has relaxed back
        self.pos = pos_max // 2
        self.P = 0.0
        self.t = clock()

    def step(self, direction: int, n: int) -> int:
        with self.lock:
            self._advance()
            self.x += direction * n
            self.pos -= direction * n
        return n

    def _gap(self) -> float:
        past = (self.x - self.x_relax) - self.x_touch
        return self.h0 - self.k * past if past > 0 else self.h0

    def _p_target(self) -> float:
        f = float(self.get_freq() or 0.0)
        if f <= 0:
            return 0.0
        h = self._gap()
        r = self.r_line + (self.r_tube * (self.h0 / h) ** 3 if h > 0.01 else 1e6)
        # operating point: P = Q R, Q = q0 (1 - P/pmax)  ->  P = q0 R / (1 + q0 R / pmax)
        return self.q0 * r / (1.0 + self.q0 * r / self.pmax)

    def _advance(self):
        now = clock()
        dt, self.t = now - self.t, now
        if dt <= 0:
            return
        self.P += (self._p_target() - self.P) * (1.0 - math.exp(-dt / self.tau))
        past = max(0.0, self.x - self.x_touch)
        self.x_relax += (self.cf * past - self.x_relax) * (1.0 - math.exp(-dt / self.ct))

    def pressure(self) -> float:
        with self.lock:
            self._advance()
            return self.P

    def pump_flow_ul_min(self, p: float) -> float:
        f = float(self.get_freq() or 0.0)
        return max(0.0, self.q0 * (1.0 - p / self.pmax)) if f > 0 else 0.0

    def read_sensor(self) -> Tuple[bool, float]:
        p = self.pressure() + self.rng.gauss(0.0, self.noise)
        return True, max(self.range[0], min(self.range[1], p))
