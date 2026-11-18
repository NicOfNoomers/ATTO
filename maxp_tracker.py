"""Maximum-pressure point tracking on the drive frequency, blocked outlet (NM, 18 Nov 2026).

Why: fixed frequency grids miss narrow pressure peaks, and the best frequency moves with the pump state.
On 18 Nov NM reached 378 mbar by hand at 650 Hz, 2.5 V, between the 600 / 800 Hz points of the max-pressure
scans. So instead of assuming frequencies, the tracker climbs to the frequency that holds the highest pressure.

Timing (NM 18 Nov: a pressure change needs more time to show than 20 s). The 2 Nov blocked-outlet steps on
261030-1 (60 s each) reach 90 % in 3-5 s, but they creep for 30-50 s afterwards: 98 % at 13-51 s, and still
0.04-0.39 mbar/s at 25-35 s. 800 Hz even fell within its step. So the defaults are long windows
(settle 3 s + probe 15 s), a plateau below 0.2 mbar/s over 15 s (build up to 90 s), and a decision on the
pressure GAIN at the end of the probe, with the creep the centre was still showing subtracted.

Method: perturb and observe at quasi-steady state (the line is closed, e.g. the syringe in hold_position):
 1. BUILD: drive f until the pressure levels off: the slope fitted over probe_s drops below plateau_slope, or
    build_max_s passes.
 2. PROBE: from that plateau, drive f + step (then f - step; the order alternates) for settle_s + probe_s.
    gain = mean of the last third of the probe - the centre's plateau - the centre's remaining creep
    (its last slope x the time between the two). A better frequency raises the pressure it holds (gain above
    gain_mbar); a worse one doesn't (or lets it sag).
    - better: move there, BUILD to its plateau, keep the step, probe again;
    - worse: back to f and REBUILD to its plateau before probing the other side (a worse frequency can let
      the pressure sag, which would fake a gain on the next probe);
    - neither side better: halve the step.
 3. Stop when the step falls below step_min, at max_s, or when the pressure reaches stop_above_mbar (pump to
    0 Hz at once; the maximum is then reported as >= stop_above_mbar).
 4. HOLD at the best frequency for hold_s (the mean of its second half is the maximum), then LEAK: 0 Hz for
    leak_s (the decay slope is what the closed line loses back through the pump).

It is a local method: it finds the peak nearest its start. To cover more than one peak, run it from several
starts (separate blocks with a pressure release between them; make_config_maxp_track.py does that).

Pure logic: the hardware is reached only through the Io callbacks, so the same code runs on the rig
(pump_test_dashboard's 'maxp_track' block) and in simulation (scripts/sim_maxp_tracker.py).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Callable, Dict, List, Optional, Tuple

LOG_COLS = ["t_s", "start_hz", "phase", "f_hz", "step_hz", "dur_s", "p_start_mbar", "p_end_mbar", "p_mean_mbar",
            "p_tail_mbar", "slope_mbar_s", "gain_mbar", "n", "decision", "note"]


@dataclass
class Params:
    start_hz: int = 650
    f_min: int = 200
    f_max: int = 2000
    step0_hz: float = 50.0          # first perturbation
    step_min_hz: float = 5.0        # resolution: stop when the step is below this
    settle_s: float = 3.0           # after each frequency change, not used for the slope (90 % in 3-5 s)
    probe_s: float = 15.0           # window for the slope and the end-of-probe pressure
    plateau_slope: float = 0.2      # mbar/s over probe_s: a build ends below this (2 Nov: reached at 25-35 s)
    gain_mbar: float = 5.0          # a probe must hold this much more than the centre (creep removed) to win
    build_max_s: float = 90.0
    max_s: float = 1200.0           # whole tracking (hold and leak come on top)
    stop_above_mbar: float = 320.0
    hold_s: float = 60.0            # the maximum = mean of the second half
    leak_s: float = 30.0
    sample_dt: float = 0.05

    @classmethod
    def from_block(cls, block: Dict[str, Any]) -> "Params":
        names = {f.name for f in fields(cls)}
        kw = {k: block[k] for k in block if k in names}
        p = cls(**kw)
        p.check()
        return p

    def check(self):
        if not (1 <= self.f_min <= self.start_hz <= self.f_max <= 5000):
            raise ValueError(f"need 1 <= f_min ({self.f_min}) <= start_hz ({self.start_hz}) <= f_max ({self.f_max})")
        if not (0 < self.step_min_hz <= self.step0_hz <= (self.f_max - self.f_min)):
            raise ValueError("need 0 < step_min_hz <= step0_hz <= f_max - f_min")
        if self.probe_s < 1.0 or self.settle_s < 0 or self.build_max_s < self.probe_s:
            raise ValueError("need probe_s >= 1, settle_s >= 0, build_max_s >= probe_s")
        if not (0 < self.plateau_slope and 0 < self.gain_mbar):
            raise ValueError("plateau_slope and gain_mbar must be > 0")
        if not (0 < self.stop_above_mbar):
            raise ValueError("stop_above_mbar must be > 0")
        if self.max_s <= 0 or self.hold_s < 0 or self.leak_s < 0:
            raise ValueError("max_s > 0, hold_s >= 0, leak_s >= 0")


@dataclass
class Io:
    set_freq: Callable[[int], None]
    read_p: Callable[[], float]                  # latest outlet pressure, mbar (NaN when not available)
    now: Callable[[], float]                     # seconds
    sleep: Callable[[float], None]
    stopped: Callable[[], bool] = lambda: False  # the run's Stop
    log: Callable[[Dict[str, Any]], None] = lambda row: None
    status: Callable[[str], None] = lambda msg: None


class _StopAbove(Exception):
    def __init__(self, p: float, f: int):
        super().__init__(f"{p:.1f} mbar at {f} Hz")
        self.p, self.f = p, f


class _Stopped(Exception):
    pass


class _TimeUp(Exception):
    pass


def fit_slope(ts: List[float], ps: List[float]) -> Tuple[float, float]:
    """Least-squares slope (mbar/s) and mean of p(t); NaN when fewer than 3 points."""
    n = len(ts)
    if n < 3:
        return float("nan"), (sum(ps) / n if n else float("nan"))
    mt, mp = sum(ts) / n, sum(ps) / n
    sxx = sum((t - mt) ** 2 for t in ts)
    if sxx <= 0:
        return float("nan"), mp
    return sum((t - mt) * (p - mp) for t, p in zip(ts, ps)) / sxx, mp


class Tracker:
    def __init__(self, prm: Params, io: Io):
        prm.check()
        self.prm, self.io = prm, io
        self.t_run0 = io.now()
        self.f_now: Optional[int] = None
        self.best_f, self.best_p = prm.start_hz, -math.inf      # highest plateau (window mean) seen
        self.peak_f, self.peak_p = prm.start_hz, -math.inf      # highest single sample seen
        self.plateaus: Dict[int, float] = {}
        self.n_moves = 0

    # ---- low level ----
    def _elapsed(self) -> float:
        return self.io.now() - self.t_run0

    def _set(self, f: int):
        if f != self.f_now:
            self.io.set_freq(int(f))
            self.f_now = int(f)

    def _sample(self, dur: float, limited: bool = True) -> Tuple[List[float], List[float]]:
        """Read the pressure for dur seconds; safety on every sample."""
        ts: List[float] = []
        ps: List[float] = []
        t_end = self.io.now() + dur
        while True:
            if self.io.stopped():
                raise _Stopped()
            t = self.io.now()
            p = self.io.read_p()
            if p == p:                                   # not NaN
                ts.append(t)
                ps.append(p)
                if p > self.peak_p and (self.f_now or 0) > 0:
                    self.peak_p, self.peak_f = p, int(self.f_now or 0)
                if p >= self.prm.stop_above_mbar and (self.f_now or 0) > 0:
                    f = int(self.f_now or 0)
                    self._set(0)
                    raise _StopAbove(p, f)
            if limited and self._elapsed() > self.prm.max_s:
                raise _TimeUp()
            if t >= t_end:
                return ts, ps
            self.io.sleep(min(self.prm.sample_dt, max(0.0, t_end - t)))

    def _window(self, f: int, phase: str, step: float, dur: float, settle: bool, start_hz: int,
                limited: bool = True) -> Dict[str, Any]:
        self._set(f)
        if settle and self.prm.settle_s > 0:
            self._sample(self.prm.settle_s, limited)
        t_a = self.io.now()
        ts, ps = self._sample(dur, limited)
        slope, pm = fit_slope(ts, ps)
        k = max(3, len(ps) // 3)                         # last third: the pressure the window ends at
        p_tail = sum(ps[-k:]) / len(ps[-k:]) if ps else float("nan")
        t_tail = sum(ts[-k:]) / len(ts[-k:]) if ts else float("nan")
        row = {"t_s": round(t_a - self.t_run0, 3), "start_hz": start_hz, "phase": phase, "f_hz": f,
               "step_hz": round(step, 2), "dur_s": round(dur, 2),
               "p_start_mbar": round(ps[0], 2) if ps else float("nan"),
               "p_end_mbar": round(ps[-1], 2) if ps else float("nan"),
               "p_mean_mbar": round(pm, 2) if pm == pm else float("nan"),
               "p_tail_mbar": round(p_tail, 2) if p_tail == p_tail else float("nan"),
               "slope_mbar_s": round(slope, 3) if slope == slope else float("nan"), "gain_mbar": float("nan"),
               "n": len(ps), "decision": "", "note": "", "_t_tail": t_tail}
        return row

    def _build(self, f: int, step: float, phase: str, start_hz: int) -> Tuple[float, float, float]:
        """Drive f until the slope is below plateau_slope (or build_max_s).
        Returns (last slope, plateau = mean of the last third of the last window, its time)."""
        t_b = self.io.now()
        first = True
        while True:
            row = self._window(f, phase, step, self.prm.probe_s, settle=first, start_hz=start_hz)
            first = False
            s, pt = row["slope_mbar_s"], row["p_tail_mbar"]
            done = (s == s and s < self.prm.plateau_slope) or (self.io.now() - t_b >= self.prm.build_max_s)
            row["decision"] = "plateau" if done else "rising"
            self.io.log(row)
            if pt == pt:
                self.plateaus[f] = pt
                if pt > self.best_p:
                    self.best_p, self.best_f = pt, f
            if done:
                return (s if s == s else 0.0), pt, row["_t_tail"]

    # ---- the tracking ----
    def run(self) -> Dict[str, Any]:
        prm, io = self.prm, self.io
        start = int(min(max(prm.start_hz, prm.f_min), prm.f_max))
        result: Dict[str, Any] = {"start_hz": start, "stop_above": False, "stopped": False, "time_up": False}
        f, step = start, float(prm.step0_hz)
        io.status(f"max-pressure tracking from {start} Hz, step {step:g} Hz")
        try:
            s_c, p_c, t_c = self._build(f, step, "build", start)
            order = +1
            while step >= prm.step_min_hz:
                better = None
                for sgn in (order, -order):
                    cand = int(round(f + sgn * step))
                    if cand == f or not (prm.f_min <= cand <= prm.f_max):
                        continue
                    row = self._window(cand, "probe", step, prm.probe_s, settle=True, start_hz=start)
                    pt = row["p_tail_mbar"]
                    creep = max(0.0, s_c) * max(0.0, row["_t_tail"] - t_c)   # what the centre would still have added
                    gain = pt - p_c - creep if pt == pt and p_c == p_c else float("nan")
                    win = gain == gain and gain > prm.gain_mbar
                    row["gain_mbar"] = round(gain, 2) if gain == gain else float("nan")
                    row["decision"] = "better" if win else "not better"
                    row["note"] = f"centre {f} Hz plateau {p_c:.1f} mbar, creep {creep:.1f} mbar"
                    io.log(row)
                    if win:
                        better = cand
                        break
                    s_c, p_c, t_c = self._build(f, step, "rebuild", start)
                order = -order
                if better is not None:
                    f = better
                    self.n_moves += 1
                    io.status(f"moved to {f} Hz (step {step:g} Hz), best plateau {self.best_p:.1f} mbar")
                    s_c, p_c, t_c = self._build(f, step, "build", start)
                else:
                    step /= 2.0
                    io.status(f"no better neighbour of {f} Hz; step -> {step:g} Hz")
            result["converged"] = True
        except _StopAbove as e:
            result.update(stop_above=True, converged=False)
            io.log({"t_s": round(self._elapsed(), 3), "start_hz": start, "phase": "stop_above", "f_hz": e.f,
                    "step_hz": round(step, 2), "dur_s": 0, "p_start_mbar": round(e.p, 2), "p_end_mbar": round(e.p, 2),
                    "p_mean_mbar": float("nan"), "slope_mbar_s": float("nan"), "n": 1,
                    "decision": "pump 0 Hz", "note": f"pressure reached the {prm.stop_above_mbar:g} mbar stop"})
            self.best_f, self.best_p = e.f, max(self.best_p, e.p)
        except _TimeUp:
            result.update(time_up=True, converged=False)
        except _Stopped:
            result.update(stopped=True, converged=False)

        hold_mean = float("nan")
        if not result["stopped"] and not result["stop_above"] and prm.hold_s > 0:
            try:
                # first half settles at the best frequency; the mean of the second half is the maximum it holds
                half_s = max(1.0, prm.hold_s / 2.0)
                row = self._window(self.best_f, "hold", step, half_s, settle=True, start_hz=start, limited=False)
                io.log(row)
                end = self._window(self.best_f, "hold_end", step, half_s, settle=False, start_hz=start, limited=False)
                io.log(end)
                hold_mean = end["p_mean_mbar"]
            except _StopAbove as e:
                result["stop_above"] = True
                self.best_f, self.best_p = e.f, max(self.best_p, e.p)
            except _Stopped:
                result["stopped"] = True
        leak_slope = float("nan")
        if not result["stopped"] and prm.leak_s > 0:
            self._set(0)
            try:
                row = self._window(0, "leak", step, prm.leak_s, settle=False, start_hz=start, limited=False)
                io.log(row)
                leak_slope = row["slope_mbar_s"]
            except _Stopped:
                result["stopped"] = True
        self._set(0)
        result.update(best_hz=self.best_f, best_plateau_mbar=round(self.best_p, 2) if self.best_p > -math.inf else None,
                      hold_mean_mbar=hold_mean, peak_sample_mbar=round(self.peak_p, 2) if self.peak_p > -math.inf else None,
                      peak_sample_hz=self.peak_f, final_step_hz=step, moves=self.n_moves, leak_slope_mbar_s=leak_slope,
                      elapsed_s=round(self._elapsed(), 1), params=asdict(prm))
        io.log({"t_s": round(self._elapsed(), 3), "start_hz": start, "phase": "result", "f_hz": self.best_f,
                "step_hz": round(step, 2), "dur_s": 0, "p_start_mbar": float("nan"), "p_end_mbar": float("nan"),
                "p_mean_mbar": hold_mean, "slope_mbar_s": leak_slope, "n": 0,
                "decision": ("stop_above" if result["stop_above"] else "stopped" if result["stopped"]
                             else "time_up" if result["time_up"] else "converged"),
                "note": f"best plateau {result['best_plateau_mbar']} mbar at {self.best_f} Hz; peak sample "
                        f"{result['peak_sample_mbar']} mbar at {self.peak_f} Hz; {self.n_moves} moves"})
        io.status(f"max-pressure tracking from {start} Hz done: {result['best_plateau_mbar']} mbar at {self.best_f} Hz"
                  + (f" (stopped at the {prm.stop_above_mbar:g} mbar limit: maximum >= that)" if result["stop_above"] else ""))
        return result


def track(prm: Params, io: Io) -> Dict[str, Any]:
    return Tracker(prm, io).run()
