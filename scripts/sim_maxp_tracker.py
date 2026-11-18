"""Offline check of maxp_tracker.py against a simulated blocked outlet (no hardware, simulated clock).

Plant: closed line with compliance C (15 nL/mbar, the 2 Nov 261030-1 estimate) plus a slow creep compliance;
the pump adds Q0(f) x (1 - P / Pmax(f)) (never negative: the valves block backflow), the valves leak G x P
(1.5 uL/min at 150 mbar, 261030-1 on 2 Nov). Pmax(f) has a narrow peak (default 650 Hz, like NM's 378 mbar
by hand on 18 Nov) and a broad one (1250 Hz); the sensor adds Gaussian noise.

usage: python scripts/sim_maxp_tracker.py            -> runs the checks below, prints a table, exit 1 on failure
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import maxp_tracker as mt  # noqa: E402


def lor(f, f0, w):
    return 1.0 / (1.0 + ((f - f0) / w) ** 2)


class Plant:
    def __init__(self, peaks=((650, 35, 380.0), (1250, 120, 250.0)), base=60.0, noise=0.3, seed=1,
                 C=0.015, G=1.5 / 60.0 / 150.0, creep_frac=0.3, creep_tau=20.0):
        self.peaks, self.base, self.noise = peaks, base, noise
        self.C, self.G, self.cf, self.ct = C, G, creep_frac, creep_tau
        self.rng = random.Random(seed)
        self.t, self.v, self.vc, self.f = 0.0, 0.0, 0.0, 0

    def pmax(self, f):
        return self.base + sum(a * lor(f, f0, w) for f0, w, a in self.peaks) if f > 0 else 0.0

    def q0(self, f):                                    # uL/s at zero pressure
        return (40.0 + 250.0 * lor(f, 1250, 150) + 150.0 * lor(f, 650, 60)) / 60.0 if f > 0 else 0.0

    def p(self):
        return (self.v - self.vc) / self.C

    def advance(self, dt):
        n = max(1, int(dt / 0.002))
        h = dt / n
        for _ in range(n):
            p = self.p()
            pm = self.pmax(self.f)
            q = self.q0(self.f) * max(0.0, 1.0 - p / pm) if pm > 0 else 0.0
            self.v += (q - self.G * p) * h
            self.vc += (self.cf * self.C * p - self.vc) * (h / self.ct)
        self.t += dt

    def io(self, rows):
        return mt.Io(set_freq=lambda f: setattr(self, "f", int(f)),
                     read_p=lambda: self.p() + self.rng.gauss(0.0, self.noise),
                     now=lambda: self.t, sleep=self.advance, stopped=lambda: False,
                     log=rows.append, status=lambda m: None)


def run_case(name, start, stop_above=500.0, noise=0.3, expect_hz=None, tol_hz=10, expect_stop=False, **kw):
    plant = Plant(noise=noise, **kw)
    rows = []
    prm = mt.Params(start_hz=start, stop_above_mbar=stop_above)
    res = mt.track(prm, plant.io(rows))
    ok = True
    if expect_stop:
        ok = res["stop_above"]
    elif expect_hz is not None:
        ok = (not res["stop_above"]) and abs(res["best_hz"] - expect_hz) <= tol_hz
    true_best = max(range(200, 2001), key=plant.pmax) if expect_hz is None else expect_hz
    print(f"{name:34s} start {start:5d}  best {res['best_hz']:5d} Hz  plateau {res['best_plateau_mbar']!s:>7} "
          f"hold {res['hold_mean_mbar']!s:>7}  Pmax(best) {plant.pmax(res['best_hz']):6.1f}  "
          f"stop {res['stop_above']!s:5}  moves {res['moves']:2d}  {res['elapsed_s']:6.1f} s  "
          f"{'OK' if ok else 'FAIL'}")
    return ok


class FlowPlant:
    """Free flow with a gas state g (0 = fully primed): the flow falls (x (1 - 0.5 g)) and the peaks move down
    (f x (1 + 0.3 g)) with gas; pumping adds gas (r_gas per s) and flow purges it (k_purge x Q / 300 x g)."""

    def __init__(self, peaks=((650, 60, 230.0), (1250, 150, 300.0)), base=40.0, noise=2.0, seed=2,
                 g0=0.0, r_gas=0.0, k_purge=0.002):
        self.peaks, self.base, self.noise = peaks, base, noise
        self.rng = random.Random(seed)
        self.t, self.f, self.g = 0.0, 0, g0
        self.r_gas, self.k_purge = r_gas, k_purge

    def q_static(self, f, g=0.0):
        if f <= 0:
            return 0.0
        fe = f * (1.0 + 0.3 * g)
        return (1.0 - 0.5 * g) * (self.base + sum(a * lor(fe, f0, w) for f0, w, a in self.peaks))

    def advance(self, dt):
        if self.f > 0:
            q = self.q_static(self.f, self.g)
            self.g = min(1.0, max(0.0, self.g + (self.r_gas - self.k_purge * q / 300.0 * self.g) * dt))
        self.t += dt

    def io(self, rows):
        return mt.Io(set_freq=lambda f: setattr(self, "f", int(f)),
                     read_p=lambda: self.q_static(self.f, self.g) * (1 + 0.03 * self.rng.gauss(0, 1))
                     + self.rng.gauss(0.0, self.noise),
                     now=lambda: self.t, sleep=self.advance, stopped=lambda: False,
                     log=rows.append, status=lambda m: None)


def run_flow_case(name, start, expect_hz, tol_hz=10, **kw):
    import maxflow_tracker as mf
    plant = FlowPlant(**kw)
    rows = []
    res = mf.track_flow(mf.FlowParams(start_hz=start), plant.io(rows))
    ok = abs(res["best_hz"] - expect_hz) <= tol_hz
    print(f"{name:34s} start {start:5d}  tracked {res['best_hz']:5d} Hz  hold {res['hold_flow']!s:>7} uL/min  "
          f"Q(tracked, g) {plant.q_static(res['best_hz'], plant.g):6.1f}  g {plant.g:.2f}  moves {res['moves']:2d}  "
          f"flushes {res['flushes']:2d}  {res['elapsed_s']:6.0f} s  {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__" and "--flow" in sys.argv:
    cases = [
        ("flow: narrow peak, start below", 600, dict(expect_hz=650)),
        ("flow: narrow peak, start above", 710, dict(expect_hz=650)),
        ("flow: broad peak, start below", 1100, dict(expect_hz=1250, tol_hz=20)),
        ("flow: broad peak, start above", 1450, dict(expect_hz=1250, tol_hz=20)),
        ("flow: noisy (5 uL/min)", 1100, dict(expect_hz=1250, tol_hz=25, noise=5.0)),
        # gas builds while pumping and flow purges it: the peak sits a little lower (steady g ~0.15)
        ("flow: gas drift", 1100, dict(expect_hz=1195, tol_hz=45, r_gas=0.0004)),
    ]
    results = [run_flow_case(n, s, **kw) for n, s, kw in cases]
    # 18 Nov rig case: outlet still closed, the sensor reads a constant offset -> must abort, not "converge"
    import maxflow_tracker as mf
    dead = FlowPlant(peaks=((650, 60, 0.0),), base=0.0, noise=0.1)
    dead.q_static = lambda f, g=0.0: -1.45
    r = mf.track_flow(mf.FlowParams(start_hz=650), dead.io([]))
    ok = r["no_flow"] and not r.get("converged")
    print(f"{'flow: dead line (closed outlet)':34s} no_flow {r['no_flow']}  {r.get('error', '')[:60]}  {'OK' if ok else 'FAIL'}")
    results.append(ok)
    print(f"{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)

if __name__ == "__main__":
    cases = [
        ("narrow peak, start below", 600, dict(expect_hz=650)),
        ("narrow peak, start above", 720, dict(expect_hz=650)),
        ("broad peak, start below", 1100, dict(expect_hz=1250, tol_hz=15)),
        ("broad peak, start above", 1400, dict(expect_hz=1250, tol_hz=15)),
        ("noisy sensor (1 mbar)", 600, dict(expect_hz=650, noise=1.0)),
        ("stop above 320 mbar", 600, dict(stop_above=320.0, expect_stop=True)),
        ("no creep", 600, dict(expect_hz=650, creep_frac=0.0)),
        # 2 Nov blocked-outlet steps crept for 30-50 s (98 % at 13-51 s): a slow, large creep compliance
        ("slow strong creep (50 %, 40 s)", 600, dict(expect_hz=650, creep_frac=0.5, creep_tau=40.0)),
        ("slow strong creep, broad peak", 1100, dict(expect_hz=1250, tol_hz=15, creep_frac=0.5, creep_tau=40.0)),
    ]
    results = [run_case(n, s, **kw) for n, s, kw in cases]
    # open line / syringe not holding: no rise -> must abort
    pl = Plant(noise=0.3)
    pl.q0 = lambda f: 0.0
    r = mt.track(mt.Params(start_hz=650), pl.io([]))
    ok = r["no_rise"] and not r.get("converged")
    print(f"{'no pressure rise (open line)':34s} no_rise {r['no_rise']}  {r.get('error', '')[:60]}  {'OK' if ok else 'FAIL'}")
    results.append(ok)
    print(f"{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
