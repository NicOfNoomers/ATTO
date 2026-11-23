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
    "start_retract_steps": 0,     # full steps retracted before the search (clamped by back_limit_steps)
    "back_limit_steps": 0,        # HARD floor: the squeeze never goes below this (0 = the start position).
                                  # NM 23 Nov: the plunger starts just off the tube; further back it falls
                                  # off its stage. Every backward move (open, retract, fault) is clamped.
    "max_search_steps": 800,      # full steps of search before giving up (~18 mm at 0.023 mm/step; NM 23 Nov:
                                  # the plunger sits ~1.5 cm from the tube; the pressure limit is the guard)
    "max_squeeze_steps": 60,      # full steps past touch: hard cap (tube protection; 23 Nov rig: closed at
                                  # 12-37 steps, the hold then drove to 123 and the tube gave way)
    "open_back_steps": 10,        # open position = touch - this
    "retract_steps": 40,          # full steps retracted on a fault
    "hold_tol_mbar": 2.0,
    "hold_tol_frac": 0.01,        # tolerance = max(hold_tol_mbar, frac x target): 300 mbar -> 3 mbar
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
    "max_move_counts": 800,      # motor counts per move command (long moves are chunked)
    "back_limit_pos": None,      # ABSOLUTE motor count of the hard back limit (syringe_v2 keeps its count in
                                  # EEPROM). Set: the floor no longer depends on where the last run left the
                                  # plunger; a start behind it is refused. None: the start position is the floor.
    "approach_steps": 0,          # full steps from the back limit moved at once before the fine touch search
                                  # (23 Nov: touch at ~530 steps from the limit; the 2-step search took ~160 s)
    "guard_stop_margin_mbar": 10.0, # position and closing moves stop this far below their stop pressure
    "guard_factor": 0.25,         # near closure: step = factor x remaining mbar / local gain
    "settle_window_s": 0.3,       # near closure: read in windows until two agree within settle_tol_mbar
    "settle_tol_mbar": 1.0,
    "settle_max_s": 4.0,
    "closed_rise_mbar": 3.0,      # close_until: closed when the squeeze has gone closed_after_steps past the last
    "closed_after_steps": 3.0,    # point that rose >= closed_rise_mbar (23 Nov 1090 Hz: +12 mbar over 14 steps)
    "over_closed_steps": 3.0,     # hold() never squeezes more than this past the closed point (or the guard stop)
    "lost_drop_frac": 0.5,        # a forward move after which the pressure fell by more than this fraction of a
    "lost_min_mbar": 40.0,        # reading >= lost_min_mbar = tube lost (23 Nov: 195 -> 5 mbar at +123): fault
    "hold_factor": 0.5,           # hold(): a forward correction covers this fraction of the error (approach from below)
    "closed_flow_frac": 0.05,     # with a flow reading (flow_fn): closed = flow <= this x the flow at the sweep start
    "closed_flow_min": 3.0,       # ... or <= this (uL/min), whichever is larger
    "closed_after_steps_flow": 12.0,  # with flow known, the pressure-only rule needs no 3 mbar rise over this many
                                  # steps (23 Nov new holder: < 3 mbar per 3 steps far from closure, 354 uL/min
                                  # still flowing at the "closed" point)
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
        self.s_ceiling: Optional[float] = None          # steps past touch hold() may not pass (closing sweep)
        self.flow_fn: Optional[Callable[[], float]] = None   # set by the runner: mean flow (uL/min) of the last ~1 s
        self.q_open = float("nan")                      # flow at the start of the last closing sweep
        self.back_limit = int(round(float(self.pc["back_limit_steps"]) * self.m))   # squeeze floor (counts)
        self.min_squeeze_seen = 0                       # for the record: the lowest squeeze ever commanded
        self.g_hist: List[float] = []                   # last two local gains, for the growth ratio
        self.g_ratio: Optional[float] = None            # gain growth per step near closure
        self.g_local: Optional[float] = None            # last measured local gain (mbar per full step), kept
                                                        # across position points (23 Nov rig: each point started
                                                        # with a full step and +9 -> +10 jumped 247 -> 328 mbar)
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
        # 23 Nov: the motor's count is only known once the firmware has reported it (POS? / stream); reading
        # it right after opening gives 0 (the driver's initial value) while the rig sat at 27133. A wrong start
        # would put the back limit in the wrong place, so ask until it answers, or refuse to start.
        if hasattr(self.syr, "query_position"):
            got = None
            for _ in range(8):                         # the Uno resets on connect: ~2 s before it answers
                got = self.syr.query_position()
                if got is not None:
                    break
                time.sleep(0.5)
            if got is None:
                raise RuntimeError("pinch: the syringe motor does not report its position (POS?): refusing to "
                                   "start, the back limit would be unknown")
            time.sleep(0.3)
            again = self.syr.query_position()
            if again is None or again != got:
                raise RuntimeError(f"pinch: the motor position is not stable at start ({got} then {again}): "
                                   f"refusing to start")
        self.start_pos = self.syr.position()
        blp = self.pc.get("back_limit_pos")
        if blp is not None:
            # position counts DOWN as the plunger goes in; behind the limit = a larger count
            if self.start_pos > int(blp):
                raise RuntimeError(f"pinch: the plunger (count {self.start_pos}) is BEHIND the back limit {int(blp)}: "
                                   f"refusing to start")
            self.back_limit = self.start_pos - int(blp)            # <= 0: the floor lies behind the start
        if self.run_dir:
            d = os.path.join(self.run_dir, "pinch")
            os.makedirs(d, exist_ok=True)
            self._csv = open(os.path.join(d, "pressure.csv"), "w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._csv)
            self._writer.writerow(["t_s", "p_mbar", "p_raw_mbar", "target_mbar", "squeeze_steps", "mode"])

    def start(self, t0: float):
        self.t0 = t0
        self._event("start", f"p_max {self.p_max:g} mbar, microsteps {self.m}, start position {self.start_pos} "
                             f"(= back limit + {self.back_limit} counts), protocol {getattr(self.syr, 'protocol', '?')}")
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
                "stale": (clock() - self.t_last_sample) > float(self.pc["stale_s"]),
                "back_limit_steps": self.back_limit / self.m, "min_squeeze_seen": self.min_squeeze_seen / self.m,
                # fields the dashboard's pressure panel reads for the syringe controller
                "settled": False, "protocol": getattr(self.syr, "protocol", "?"), "microsteps": self.m,
                "resolution_mbar": None, "limit_source": "pinch limit_mbar / 95 % of the sensor range",
                "blocked": None, "remaining_pull": max(0, self.squeeze() - self.back_limit),
                "backlash_steps": 0.0, "drift_mbar_s": 0.0, "gain_push_mbar_step": round(self.gain, 3),
                "gain_pull_mbar_step": round(self.gain, 3)}

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
                back = min(int(self.pc["retract_steps"]) * self.m, max(0, self.squeeze() - self.back_limit))
                if back > 0:
                    self.syr.move(-back, timeout_s=float(self.pc["move_timeout_s"]))
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
        if n < 0:                                      # never behind the back limit (start position)
            allowed = max(0, self.squeeze() - self.back_limit)   # how far back we may still go (>= 0)
            n = max(n, -allowed)                       # a backward request can only shrink, never turn forward
        if n > 0 and self.touch is not None:
            cap = self.touch + int(self.pc["max_squeeze_steps"]) * self.m
            n = min(n, cap - self.squeeze())
        if n == 0:
            return 0
        # 23 Nov: the v1 driver waits at most min(timeout, 3 ms/count + 1 s) for DONE and cannot stop a move,
        # so a long move (e.g. back to the start from 800 steps = 12800 counts) is sent in chunks that each finish
        # well inside their wait; a failed chunk ends the move.
        chunk = max(1, int(self.pc["max_move_counts"]))
        done = 0
        with self.mlock:
            while done != n:
                if self.mode == "fault" and n > 0:
                    break
                c = max(-chunk, min(chunk, n - done))
                ok = self.syr.move(c, timeout_s=0.004 * abs(c) + 2.0)
                if not ok:
                    self._event("move_failed", f"chunk {c} counts after {done} of {n}")
                    break
                done += c
        self.min_squeeze_seen = min(self.min_squeeze_seen, self.squeeze())
        return done

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
        self.g_local, self.g_hist, self.g_ratio = None, [], None
        self.s_ceiling = None
        self._move_full(-float(pc["start_retract_steps"]))
        appr = float(pc.get("approach_steps") or 0)
        if appr > 0:
            goal = self.back_limit + int(round(appr * self.m))      # squeeze counts
            if self.squeeze() < goal:
                self._event("approach", f"to {appr:g} full steps from the back limit")
                self._goto_squeeze(goal)
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
        # NM 23 Nov: no touch -> go back to where the search started (never behind the back limit)
        self._event("touch_not_found", f"{moved} full steps searched, baseline {p0:.1f} mbar; back to the start")
        try:
            self._goto_squeeze(max(self.back_limit, 0))
        except Exception as e:  # noqa: BLE001
            self._event("return_failed", str(e))
        self.mode = "idle"
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

    def _adaptive_step(self, p: float, stop: float, g: Optional[float], step_max: float,
                       r: Optional[float] = None) -> float:
        """Full steps for the next move toward stop, at least one motor count, at most step_max.
        Near closure the gain grows step by step (23 Nov rig, 520 Hz: x1.5-3 per step; 167 -> 247 -> > 328 mbar),
        so with a growth ratio r per step the predicted rise over s steps is g (r^s - 1) / ln r, and s is chosen
        so that rise = guard_factor x the remaining pressure. Without r: s = factor x remaining / g."""
        if g is None or g <= 0:
            return step_max
        budget = float(self.pc["guard_factor"]) * (stop - p)
        if budget <= 0:
            return 1.0 / self.m
        if r is not None and r > 1.05:
            lr = math.log(min(r, 6.0))
            s = math.log(1.0 + budget * lr / g) / lr
        else:
            s = budget / g
        return max(1.0 / self.m, min(step_max, s))

    def _settled_p(self) -> float:
        """Pressure after it has stopped rising: windows of settle_window_s until two agree within
        settle_tol_mbar, at most settle_max_s (23 Nov sim: reading too early under-reads, the next step overshoots)."""
        pc = self.pc
        t_end = clock() + float(pc["settle_max_s"])
        prev = self._p_mean(float(pc["settle_window_s"]))
        while clock() < t_end:
            cur = self._p_mean(float(pc["settle_window_s"]))
            if abs(cur - prev) <= float(pc["settle_tol_mbar"]):
                return cur
            prev = cur
        return prev

    def _note_gain(self, g_new: float):
        """Keep the last two local gains (per full step) for the growth ratio r = g_new / g_old."""
        if g_new <= 0:
            return
        self.g_hist.append(g_new)
        self.g_hist = self.g_hist[-2:]
        if len(self.g_hist) == 2 and self.g_hist[0] > 0:
            self.g_ratio = max(1.0, self.g_hist[1] / self.g_hist[0])

    def _flow_now(self) -> float:
        """Mean flow of the last ~1 s from the runner (nan without a flow reading)."""
        if self.flow_fn is None:
            return float("nan")
        try:
            q = float(self.flow_fn())
        except Exception:  # noqa: BLE001
            return float("nan")
        return q

    def _lost_check(self, p_before: float, p_after: float) -> bool:
        """After a FORWARD move: a pressure fall by more than lost_drop_frac from a reading >= lost_min_mbar means
        the tube is no longer pinched (slipped out, split, or a joint came off: 23 Nov, 195 -> 5 mbar at 123 steps
        past touch). Fault: the motor retracts and the run stops; a person has to look at the tube."""
        pc = self.pc
        if p_before == p_before and p_after == p_after and p_before >= float(pc["lost_min_mbar"]) \
                and p_after < (1.0 - float(pc["lost_drop_frac"])) * p_before:
            pt = self.past_touch_full()
            self._fault(f"tube lost? pressure fell from {p_before:.0f} to {p_after:.0f} mbar on a forward move at "
                        f"{pt if pt is None else round(pt, 2)} steps past touch: check the flex tube under the "
                        f"plunger and the joints")
            return True
        return False

    def set_squeeze_guarded(self, steps_past_touch: float, stop_mbar: float,
                            stopped: Callable[[], bool] = lambda: False) -> Tuple[bool, float]:
        """Go to touch + steps (position mode) in small adaptive moves, watching the pressure; stops short when
        the pressure reaches stop_mbar - guard margin. Returns (reached, pressure)."""
        if self.touch is None:
            raise RuntimeError("set_squeeze_guarded needs the touch point")
        pc = self.pc
        stop = min(stop_mbar, self.p_max - 5.0) - float(pc["guard_stop_margin_mbar"])
        goal = int(round(self.touch + max(-float(pc["open_back_steps"]),
                                          min(float(steps_past_touch), float(pc["max_squeeze_steps"]))) * self.m))
        self.mode = "position"
        if goal <= self.squeeze():                      # backward: no pressure risk
            self._goto_squeeze(goal)
            return True, self._p_mean(0.3)
        p_prev, s_prev = self._settled_p(), self.squeeze_full()
        g = self.g_local
        while self.squeeze() < goal and not stopped():
            if self.mode == "fault":
                return False, self.p_1s
            if p_prev >= stop:
                self._event("guard_stop", f"{p_prev:.1f} mbar >= {stop:.1f} at {self.past_touch_full():.2f} steps past touch")
                return False, p_prev
            # no gain known yet and already above half the stop: creep (quarter steps) until one is measured
            step_max = 0.25 if (g is None and p_prev > 0.5 * stop) else 1.0
            step = min(self._adaptive_step(p_prev, stop, g, step_max, self.g_ratio), (goal - self.squeeze()) / self.m)
            if self._move_full(step) == 0:
                break
            p = self._settled_p()                         # always settled: an early read under-reads (+6: 123 vs 264)
            if self.mode == "fault" or self._lost_check(p_prev, p):
                return False, self.p_1s
            ds = self.squeeze_full() - s_prev
            if ds > 0 and p > p_prev:
                g_new = (p - p_prev) / ds
                self._note_gain(g_new)
                g = g_new if g is None else max(g_new, 0.5 * g)   # gains only grow toward closure: keep the bigger
                self.g_local = g
            p_prev, s_prev = p, self.squeeze_full()
        return self.squeeze() >= goal, p_prev

    def hold(self, target: float, timeout_s: float = 60.0, stopped: Callable[[], bool] = lambda: False) -> str:
        """Squeeze to hold target mbar. Returns settled / timeout / at_cap / cannot_reach / stopped / fault.
        With a map from a closing sweep (close_until, same frequency) it first moves to the mapped squeeze and
        takes the local gain from the map; a target above the mapped maximum is cannot_reach at once. Without a
        map it learns the gain from its own moves (only from moves big enough to rise above the noise)."""
        pc = self.pc
        if target > self.p_max - 5.0:
            raise ValueError(f"target {target:g} mbar is above the limit {self.p_max - 5:g} mbar")
        tol = max(float(pc["hold_tol_mbar"]), float(pc["hold_tol_frac"]) * abs(float(target)))
        dt = float(pc["hold_dt"])
        gmin, smax = float(pc["gain_min"]), float(pc["max_step"])
        self.target = float(target)
        if self.pmap and self.touch is not None:
            p_top = max(pp for _, pp in self.pmap)
            if target > p_top + tol:
                self._event("hold_cannot_reach", f"{target:g} mbar is above the mapped maximum {p_top:.1f} mbar")
                return "cannot_reach"
            s_goal, _ = self._map_lookup(target)
            if s_goal is not None:
                # guarded: after opening and re-squeezing, the tube gives more pressure at the same squeeze
                # (relaxation/hysteresis), so the mapped squeeze can overshoot on the steep part (23 Nov sim)
                cap = min(target + 15.0, self.p_max - 15.0) + float(pc["guard_stop_margin_mbar"])
                self.set_squeeze_guarded(s_goal, cap, stopped)
                if self.mode == "fault":
                    return "fault"
        self.mode = "hold"
        t_end = clock() + timeout_s
        in_tol_since = None
        last: Optional[Tuple[float, float]] = None
        p_fwd: Optional[float] = None                     # reading before the last forward move (tube-lost check)
        hf = float(pc["hold_factor"])
        while clock() < t_end:
            if stopped():
                return "stopped"
            if self.mode == "fault":
                return "fault"
            p = self._settled_p()                         # settled: the tube lags and creeps (23 Nov oscillation)
            if p != p:
                continue
            if self.mode == "fault":
                return "fault"
            if p_fwd is not None and self._lost_check(p_fwd, p):
                return "fault"
            p_fwd = None
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
            g = max(g, gmin)
            if e > 0:                                     # forward: a fraction of the error, never past the ceiling
                step = max(1.0 / self.m, min(smax, hf * e / g))
                room = (self.p_max - 15.0) - p            # never predict past p_max - 15 mbar
                if room <= 0 or g * (1.0 / self.m) > room:
                    self._event("hold_at_limit", f"{target:g} mbar: {p:.1f} mbar, one count would pass the limit")
                    return "at_limit"
                step = min(step, room / g)
                if self.s_ceiling is not None and self.touch is not None:
                    allowed = self.s_ceiling - self.past_touch_full()
                    if allowed < 1.0 / self.m:
                        self._event("hold_at_closed", f"{target:g} mbar: {p:.1f} mbar at {self.past_touch_full():.2f} "
                                                      f"steps past touch, the closed point + margin "
                                                      f"({self.s_ceiling:.2f}): the pump cannot make it now")
                        return "cannot_reach"
                    step = min(step, allowed)
                last, p_fwd = (s_now, p), p
                if self._move_full(step) == 0:
                    self._event("hold_at_cap", f"{target:g} mbar: {p:.1f} mbar at the squeeze cap")
                    return "at_cap"
            else:                                         # overshoot: back off
                step = max(-smax, min(-1.0 / self.m, 0.6 * e / g))
                last = (s_now, p)
                self._move_full(step)
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
        self.s_ceiling = None
        hist: deque = deque(maxlen=int(pc["plateau_n"]) + 1)
        p_first = None
        s_rise: Optional[float] = None               # last squeeze that raised the pressure
        p_rise = float("nan")
        p_fwd: Optional[float] = None
        over = float(pc["over_closed_steps"])
        self.q_open = self._flow_now()                  # the sweep starts at the open point
        flow_known = self.q_open == self.q_open and self.q_open > 0
        q_closed = (max(float(pc["closed_flow_frac"]) * self.q_open, float(pc["closed_flow_min"]))
                    if flow_known else float("nan"))
        after = float(pc["closed_after_steps_flow"] if flow_known else pc["closed_after_steps"])
        while not stopped():
            if self.mode == "fault":
                return self.p_1s, "fault"
            p = self._settled_p()                         # always settled (an early read under-reads near closure)
            if self.mode == "fault" or (p_fwd is not None and self._lost_check(p_fwd, p)):
                return self.p_1s, "fault"
            p_first = p if p_first is None else p_first
            pt = self.past_touch_full()
            if pt is not None:
                self.pmap.append((pt, p))
                q = self._flow_now() if flow_known else float("nan")
                if flow_known and q == q and q <= q_closed:      # closed by flow: the outlet is blocked
                    self.s_ceiling = pt + over
                    self._event("closed", f"{p:.1f} mbar: flow {q:.1f} uL/min <= {q_closed:.1f} "
                                          f"({float(pc['closed_flow_frac']):g} x {self.q_open:.1f} at the open point) "
                                          f"at {pt:.2f} steps past touch")
                    self.mode = "position"
                    return p, "closed"
                if s_rise is None or p - p_rise >= float(pc["closed_rise_mbar"]):
                    s_rise, p_rise = pt, p
                elif pt - s_rise >= after and p - p_first >= float(pc["plateau_min_rise_mbar"]):
                    self.s_ceiling = s_rise + over
                    self._event("closed", f"{p:.1f} mbar: no rise of {float(pc['closed_rise_mbar']):g} mbar over "
                                          f"{pt - s_rise:.2f} steps after {s_rise:.2f} steps past touch"
                                          + (f"; flow {q:.1f} uL/min (open {self.q_open:.1f}): the tube does not "
                                             f"close further" if flow_known else ""))
                    self.mode = "position"
                    return p, ("plateau_open" if flow_known else "closed")
            hist.append(p)
            if p >= stop_mbar:
                self.s_ceiling = (pt or 0.0) + over
                self.mode = "position"
                return p, "stop"
            if (len(hist) == hist.maxlen and hist[-1] - hist[0] < float(pc["plateau_mbar"])
                    and p - p_first >= float(pc["plateau_min_rise_mbar"])):
                self.s_ceiling = (s_rise if s_rise is not None else (pt or 0.0)) + over
                self.mode = "position"
                return p, "plateau"
            g = self.g_local
            if len(self.pmap) >= 2 and self.pmap[-1][0] > self.pmap[-2][0]:
                g_new = (self.pmap[-1][1] - self.pmap[-2][1]) / (self.pmap[-1][0] - self.pmap[-2][0])
                if g_new > 0:
                    self._note_gain(g_new)
                    g = g_new if g is None else max(g_new, 0.5 * g)
                    self.g_local = g
            if g is None and p > 0.5 * stop_mbar:
                g = None                                   # unknown: _adaptive_step gets a quarter step below
            guard = stop_mbar - float(pc["guard_stop_margin_mbar"])
            if p >= guard:
                self.s_ceiling = (pt or 0.0) + over
                self.mode = "position"
                return p, "stop"
            smax = 0.25 if (g is None and p > 0.5 * guard) else step_full
            p_fwd = p
            if self._move_full(self._adaptive_step(p, guard, g, smax, self.g_ratio)) == 0:
                self.s_ceiling = (pt or 0.0) + over
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
                 sensor_range=(-344.74, 344.74), seed: int = 3, pos_max: int = 2470 * 16,
                 escape_past: Optional[int] = None):
        self.get_freq = get_freq
        self.x_touch, self.k, self.h0 = x_touch, k_mm_per_count, h0_mm
        self.r_line, self.r_tube, self.q0, self.pmax = r_line, r_tube, q0, pmax   # R in mbar per (uL/min)
        self.tau, self.cf, self.ct, self.noise = tau_s, creep_frac, creep_tau_s, noise_mbar
        self.range = tuple(sensor_range)
        self.rng = random.Random(seed)
        self.lock = threading.Lock()
        self.x = 0                         # squeeze (motor counts) relative to the start
        self.x_max = 0                     # deepest squeeze reached (tests)
        self.x_relax = 0.0                 # part of the squeeze the tube has relaxed back
        self.escape_past = escape_past     # counts past touch at which the tube slips out (tests); then open
        self.escaped = False
        self.pos = pos_max // 2
        self.P = 0.0
        self.t = clock()

    def step(self, direction: int, n: int) -> int:
        with self.lock:
            self._advance()
            self.x += direction * n
            self.x_max = max(self.x_max, self.x)
            self.pos -= direction * n
        return n

    def _gap(self) -> float:
        past = (self.x - self.x_relax) - self.x_touch
        if self.escape_past is not None and (self.escaped or past > self.escape_past):
            self.escaped = True
            return self.h0
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
