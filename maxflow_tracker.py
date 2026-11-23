"""Maximum-flow point tracking on the drive frequency, free flow, with flushes (NM, 18 Nov 2026).

Why: NM expects the frequency that gives the most free flow to be close to the one that holds the most
pressure, and free flow answers within seconds (no syringe, no plateau to wait for). So track the
maximum-flow frequency first, then start the pressure tracker (maxp_tracker.py) there.
NM also asked for flushing runs built in: priming at the current best frequency to force out gas and keep
the pump fully primed. Otherwise a gas pocket can make a frequency look worse than it is (260930-2, 12 Nov:
the flow slid while pumping and recovered when a gas slug left).

Method (perturb and observe; the pump state drifts, so every comparison is against a fresh centre):
 0. FLUSH at the start frequency: flush_rounds x [f for 30 s, 0.8 f for 10 s, 1.2 f for 10 s, with 5 s at
    0 Hz between], like make_config_prime_at.py, plus any flush_extra_hz for 10 s each.
 1. MEASURE the centre f: settle_s, then the mean flow over window_s.
 2. PROBE f + step (then f - step; the order alternates), then re-measure the centre.
    gain = probe - mean(centre before, centre after) (cancels the drift between them).
    A win needs gain > max(gain_abs, gain_frac x centre). Win: move there and keep the step.
    No side wins: halve the step.
 3. FLUSH again at the current best frequency every flush_every_s. Also flush when the centre falls below
    drop_frac x the best centre seen (a gas event), then re-measure.
 4. Stop when the step is below step_min_hz or at max_s. HOLD the best frequency for hold_s: its mean flow is
    the result.
A local method: one start per suspected peak (separate blocks; make_config_maxflow_track.py).

Pure logic behind the maxp_tracker.Io callbacks: read_p returns the latest FLOW sample here (uL/min). The rig
block is 'maxflow_track' in pump_test_dashboard.py; scripts/sim_maxp_tracker.py --flow simulates it.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional, Tuple

from maxp_tracker import Io

LOG_COLS = ["t_s", "start_hz", "phase", "f_hz", "step_hz", "dur_s", "q_mean", "q_sd", "n", "ref_q", "gain",
            "decision", "note"]


@dataclass
class FlowParams:
    start_hz: int = 650
    f_min: int = 200
    f_max: int = 2000
    step0_hz: float = 50.0
    step_min_hz: float = 5.0
    settle_s: float = 3.0
    window_s: float = 10.0
    gain_abs: float = 3.0           # uL/min
    gain_frac: float = 0.03         # of the centre flow
    max_s: float = 1800.0           # tracking cap per start (flushes included; the hold comes on top)
    hold_s: float = 60.0
    flush_rounds: int = 3           # at the start: rounds of [f 30 s, 0.8 f 10 s, 1.2 f 10 s]
    flush_every_s: float = 300.0    # later flushes (1 round each) at the current best frequency; 0 = never
    flush_extra_hz: Tuple[int, ...] = ()   # extra flush frequencies, 10 s each per round (e.g. 1300)
    drop_frac: float = 0.8          # centre below this x best -> flush now
    min_flow: float = 5.0           # uL/min: below this at the start (flush and first centre) -> abort "no flow"
    seed_from_flush: bool = True    # start the climb at the best frequency of the start flush's last round
    sample_dt: float = 0.04

    @classmethod
    def from_block(cls, block: Dict[str, Any]) -> "FlowParams":
        names = {f.name for f in fields(cls)}
        kw = {k: block[k] for k in block if k in names}
        if "flush_extra_hz" in kw:
            kw["flush_extra_hz"] = tuple(int(x) for x in (kw["flush_extra_hz"] or ()))
        p = cls(**kw)
        p.check()
        return p

    def check(self):
        if not (1 <= self.f_min <= self.start_hz <= self.f_max <= 5000):
            raise ValueError(f"need 1 <= f_min ({self.f_min}) <= start_hz ({self.start_hz}) <= f_max ({self.f_max})")
        if not (0 < self.step_min_hz <= self.step0_hz <= (self.f_max - self.f_min)):
            raise ValueError("need 0 < step_min_hz <= step0_hz <= f_max - f_min")
        if self.window_s < 2.0 or self.settle_s < 0:
            raise ValueError("need window_s >= 2 and settle_s >= 0")
        if self.gain_abs < 0 or not 0 <= self.gain_frac < 1:
            raise ValueError("need gain_abs >= 0 and 0 <= gain_frac < 1")
        if not 0 < self.drop_frac <= 1:
            raise ValueError("need 0 < drop_frac <= 1")
        if self.max_s <= 0 or self.hold_s < 0 or self.flush_rounds < 0 or self.flush_every_s < 0:
            raise ValueError("max_s > 0; hold_s, flush_rounds, flush_every_s >= 0")
        for f in self.flush_extra_hz:
            if not 1 <= f <= 5000:
                raise ValueError(f"flush_extra_hz {f} outside 1..5000 Hz")


class _Stopped(Exception):
    pass


class _TimeUp(Exception):
    pass


class _NoFlow(Exception):
    pass


class FlowTracker:
    def __init__(self, prm: FlowParams, io: Io):
        prm.check()
        self.prm, self.io = prm, io
        self.t0 = io.now()
        self.f_now: Optional[int] = None
        self.best_f, self.best_q = prm.start_hz, -math.inf
        self.centres: Dict[int, List[float]] = {}
        self.n_moves = 0
        self.n_flush = 0
        self.last_flush = self.t0
        self.flush_max_q = -math.inf                    # highest flush-block flow (the no-flow check)
        self.flush_last: Dict[int, float] = {}          # flush block means of the latest round, per frequency

    def _elapsed(self) -> float:
        return self.io.now() - self.t0

    def _set(self, f: int):
        if f != self.f_now:
            self.io.set_freq(int(f))
            self.f_now = int(f)

    def _sample(self, dur: float, limited: bool) -> List[float]:
        qs: List[float] = []
        t_end = self.io.now() + dur
        while True:
            if self.io.stopped():
                raise _Stopped()
            t = self.io.now()
            q = self.io.read_p()
            if q is not None and q == q:
                qs.append(float(q))
            if limited and self._elapsed() > self.prm.max_s:
                raise _TimeUp()
            if t >= t_end:
                return qs
            self.io.sleep(min(self.prm.sample_dt, max(0.0, t_end - t)))

    def _measure(self, f: int, phase: str, step: float, start: int, limited: bool = True,
                 window: Optional[float] = None) -> Dict[str, Any]:
        self._set(f)
        if self.prm.settle_s > 0:
            self._sample(self.prm.settle_s, limited)
        w = self.prm.window_s if window is None else window
        t_a = self._elapsed()
        qs = self._sample(w, limited)
        n = len(qs)
        m = sum(qs) / n if n else float("nan")
        sd = math.sqrt(sum((q - m) ** 2 for q in qs) / (n - 1)) if n > 1 else float("nan")
        return {"t_s": round(t_a, 3), "start_hz": start, "phase": phase, "f_hz": f, "step_hz": round(step, 2),
                "dur_s": round(w, 2), "q_mean": round(m, 2) if m == m else float("nan"),
                "q_sd": round(sd, 2) if sd == sd else float("nan"), "n": n, "ref_q": float("nan"),
                "gain": float("nan"), "decision": "", "note": ""}

    def _centre(self, f: int, step: float, start: int, phase: str = "centre") -> float:
        row = self._measure(f, phase, step, start)
        self.io.log(row)
        q = row["q_mean"]
        if q == q:
            self.centres.setdefault(f, []).append(q)
            if q > self.best_q:
                self.best_q, self.best_f = q, f
        return q

    def _flush(self, f: int, rounds: int, step: float, start: int, why: str):
        """prime_at pattern around f: rounds x [f 30 s, 0 Hz 5 s, 0.8 f 10 s, 0 Hz 5 s, 1.2 f 10 s, 0 Hz 5 s]
        (+ flush_extra_hz 10 s each); the flow of each block is logged (phase 'flush')."""
        if rounds <= 0:
            return
        self.n_flush += 1
        self.io.status(f"flush at {f} Hz ({rounds} round(s)): {why}")
        plan: List[Tuple[int, float]] = []
        for _ in range(rounds):
            for k, d in ((1.0, 30.0), (0.8, 10.0), (1.2, 10.0)):
                fk = int(round(min(max(f * k, self.prm.f_min), self.prm.f_max)))
                plan += [(fk, d), (0, 5.0)]
            for fx in self.prm.flush_extra_hz:
                plan += [(int(fx), 10.0), (0, 5.0)]
        for fk, d in plan:
            self._set(fk)
            qs = self._sample(d, limited=False)
            if fk > 0:
                tail = qs[len(qs) * 2 // 5:]
                m = sum(tail) / len(tail) if tail else float("nan")
                if m == m:
                    self.flush_max_q = max(self.flush_max_q, m)
                    self.flush_last[fk] = m
                self.io.log({"t_s": round(self._elapsed(), 3), "start_hz": start, "phase": "flush", "f_hz": fk,
                             "step_hz": round(step, 2), "dur_s": d, "q_mean": round(m, 2) if m == m else float("nan"),
                             "q_sd": float("nan"), "n": len(tail), "ref_q": float("nan"), "gain": float("nan"),
                             "decision": "", "note": why})
        self.last_flush = self.io.now()

    def run(self) -> Dict[str, Any]:
        prm, io = self.prm, self.io
        start = int(min(max(prm.start_hz, prm.f_min), prm.f_max))
        f, step = start, float(prm.step0_hz)
        res: Dict[str, Any] = {"start_hz": start, "stopped": False, "time_up": False, "no_flow": False}
        io.status(f"max-flow tracking from {start} Hz, step {step:g} Hz")
        try:
            self._flush(f, prm.flush_rounds, step, start, "start")
            # 18 Nov (260930-2): the start flush's 0.8 f blocks beat what the climb from f found (520 Hz 65-67 vs
            # 750 Hz 54 uL/min). The flush is a free 3-point scan, so the climb starts at its best frequency
            # (last round = the current state; the first round at 1250 Hz read 229, then 74-88).
            if prm.seed_from_flush and self.flush_last:
                f_seed = max(self.flush_last, key=self.flush_last.get)
                if f_seed != f and prm.f_min <= f_seed <= prm.f_max:
                    io.status(f"seed: flush block {f_seed} Hz ({self.flush_last[f_seed]:.0f} uL/min) beat {f} Hz "
                              f"({self.flush_last.get(f, float('nan')):.0f}); climbing from {f_seed} Hz")
                    f = f_seed
            q_c = self._centre(f, step, start)
            # 18 Nov: the outlet was still closed from a manual pressure test; the flow sat at -1.4 uL/min at
            # every frequency and the tracker "converged" on noise. A dead line is reported, not tracked.
            q_any = max(q_c if q_c == q_c else -math.inf, self.flush_max_q)
            if not q_any > prm.min_flow:
                raise _NoFlow(f"flow {q_any:.1f} uL/min (centre and flush blocks) is below {prm.min_flow:g}: "
                              f"outlet closed, line blocked or flow sensor not in line?")
            order = +1
            while step >= prm.step_min_hz:
                if prm.flush_every_s > 0 and io.now() - self.last_flush >= prm.flush_every_s:
                    self._flush(self.best_f, 1, step, start, f"every {prm.flush_every_s:g} s")
                    q_c = self._centre(f, step, start)
                if self.best_q > 0 and q_c == q_c and q_c < prm.drop_frac * self.best_q:
                    self._flush(f, 1, step, start, f"centre {q_c:.0f} < {prm.drop_frac:g} x best {self.best_q:.0f}")
                    q_c = self._centre(f, step, start)
                better = None
                for sgn in (order, -order):
                    cand = int(round(f + sgn * step))
                    if cand == f or not (prm.f_min <= cand <= prm.f_max):
                        continue
                    row = self._measure(cand, "probe", step, start)
                    q_after = self._centre(f, step, start, phase="centre_after")
                    ref = (q_c + q_after) / 2.0 if q_c == q_c and q_after == q_after else q_after
                    q = row["q_mean"]
                    gain = q - ref if q == q and ref == ref else float("nan")
                    thr = max(prm.gain_abs, prm.gain_frac * abs(ref)) if ref == ref else float("inf")
                    win = gain == gain and gain > thr
                    row.update(ref_q=round(ref, 2) if ref == ref else float("nan"),
                               gain=round(gain, 2) if gain == gain else float("nan"),
                               decision="better" if win else "not better", note=f"centre {f} Hz, threshold {thr:.1f}")
                    io.log(row)
                    if q == q and q > self.best_q:
                        self.best_q, self.best_f = q, cand
                    q_c = q_after
                    if win:
                        better = cand
                        break
                order = -order
                if better is not None:
                    f = better
                    self.n_moves += 1
                    io.status(f"moved to {f} Hz (step {step:g} Hz)")
                    q_c = self._centre(f, step, start)
                else:
                    step /= 2.0
                    io.status(f"no better neighbour of {f} Hz; step -> {step:g} Hz")
            res["converged"] = True
        except _NoFlow as e:
            res.update(no_flow=True, converged=False, error=str(e))
            io.status(f"max-flow tracking from {start} Hz ABORTED: {e}")
        except _TimeUp:
            res.update(time_up=True, converged=False)
        except _Stopped:
            res.update(stopped=True, converged=False)

        # the tracked frequency is where the climb ended (f); best_f is the single highest reading (can be noise)
        final_f = f
        hold_q = float("nan")
        if not res["stopped"] and not res["no_flow"] and prm.hold_s > 0:
            try:
                row = self._measure(final_f, "hold", step, start, limited=False, window=prm.hold_s)
                io.log(row)
                hold_q = row["q_mean"]
            except _Stopped:
                res["stopped"] = True
        self._set(0)
        res.update(seed_hz=None, flush_last={int(k): round(v, 2) for k, v in self.flush_last.items()})
        res.update(best_hz=final_f, hold_flow=hold_q, highest_reading=round(self.best_q, 2) if self.best_q > -math.inf
                   else None, highest_reading_hz=self.best_f, final_step_hz=step, moves=self.n_moves,
                   flushes=self.n_flush, elapsed_s=round(self._elapsed(), 1), params=asdict(prm))
        io.log({"t_s": round(self._elapsed(), 3), "start_hz": start, "phase": "result", "f_hz": final_f,
                "step_hz": round(step, 2), "dur_s": 0, "q_mean": hold_q, "q_sd": float("nan"), "n": 0,
                "ref_q": float("nan"), "gain": float("nan"),
                "decision": ("no_flow" if res["no_flow"] else "stopped" if res["stopped"]
                             else "time_up" if res["time_up"] else "converged"),
                "note": f"hold {hold_q} uL/min at {final_f} Hz; highest reading {res['highest_reading']} at "
                        f"{self.best_f} Hz; {self.n_moves} moves, {self.n_flush} flushes"})
        io.status(f"max-flow tracking from {start} Hz done: {hold_q} uL/min at {final_f} Hz")
        return res


def track_flow(prm: FlowParams, io: Io) -> Dict[str, Any]:
    return FlowTracker(prm, io).run()
