"""G test analysis: chamber compliance and gas volume from pv_steps.csv (syringe volume steps).

Per move: C = dv_line / dp (uL/mbar), with dv_line the syringe volume pushed into the line (counts x
ul_per_step / microsteps) and dp the change of the held pressure (means over the end of each hold).
  pull branch (outlet valve open, chamber follows):  C_pull = C_line + C_struct + C_gas(p_ch)
  push branch (outlet valve closed, chamber held):   C_push = C_line (outlet line only)
  baseline run (pump replaced by a stiff union):     C_base = C_line + the inlet stub
Chamber pressure: while pulling, the chamber sits above the line reading by the outlet valve's cracking
pressure (plus any height difference), P_cr. Each pull staircase starts with small fixed moves (fine_mbar):
the compliance steps up from C_line to C_line + chamber where the valve opens, at a line reading of -P_cr
(the chamber starts at the reservoir's 0). --p-crack auto finds that step per staircase (median over them).
Moves that began before the valve opened are dropped.
Gas, isothermal: V = V_g0 p_atm / p_abs, so over a step its secant compliance is
V_g0 p_atm / ((p_atm + p1 + P_cr)(p_atm + p2 + P_cr)). Fit, by least squares over the pull moves,
  C_pull - C_line(p) = C_struct + V_g0 x,   x = p_atm / ((p_atm + p1 + P_cr)(p_atm + p2 + P_cr))
C_line(p): a straight line in p fitted to the baseline run's PULL moves (--baseline, preferred: same direction,
includes the inlet stub) or to the push branch of the same run (--line-from-push: opposite direction, misses
the stub, so C_struct comes out high by the stub's compliance).
Gate (go/no-go): the line's repeatability between repeats (baseline pull staircases, or the push branches):
the spread of its level (at -150 mbar) against --gate (3 %), and what the spreads mean for the result:
the level spread goes into C_struct, the spread of its change from 0 to -300 mbar into V_g0
(divided by x(-300) - x(0)). The gas term itself changes the total by only ~8 % over the range.
The two-level numbers of the brief (mean chamber compliance near 0 and near -300 mbar) are printed as well.
1 uL/mbar = 1e-11 m^3/Pa.

usage: python scripts/analyze_pv_steps.py <run_dir> [<run_dir> ...] (--baseline <run_dir> | --line-from-push)
       [--p-crack auto|<mbar>] [--gate 0.03] [--p-atm 1013] [--min-dp 0.5] [--plot fig.png] [--json]
"""
import argparse
import csv
import json
import math
import os
import sys


def load(run_dir):
    path = run_dir if run_dir.endswith(".csv") else os.path.join(run_dir, "pv_steps.csv")
    rows = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("phase") != "step":
            continue
        try:
            dv, p1, p2 = float(r["dv_line_ul"]), float(r["p_before_mbar"]), float(r["p_after_mbar"])
        except (TypeError, ValueError):
            continue
        r.update(dv=dv, p1=p1, p2=p2, dp=p2 - p1, pm=0.5 * (p1 + p2), src=run_dir)
        rows.append(r)
    return rows


def lstsq(xs, ys, cols, w=None):
    """Weighted least squares y = sum b_j f_j(x); returns (b, se, weighted resid_sd). The compliance of a move,
    dv/dp, has an error ~ sigma_p / |dp|, so moves are weighted by dp^2 (fine 5 mbar moves count 1/16 of a
    20 mbar move)."""
    n, k = len(xs), len(cols)
    w = w or [1.0] * n
    sw = sum(w) / n
    w = [wi / sw for wi in w]
    A = [[f(x) for f in cols] for x in xs]
    ata = [[sum(w[i] * A[i][p] * A[i][q] for i in range(n)) for q in range(k)] for p in range(k)]
    aty = [sum(w[i] * A[i][p] * ys[i] for i in range(n)) for p in range(k)]
    inv = invert(ata)
    b = [sum(inv[p][q] * aty[q] for q in range(k)) for p in range(k)]
    res = [ys[i] - sum(b[p] * A[i][p] for p in range(k)) for i in range(n)]
    s2 = sum(w[i] * res[i] ** 2 for i in range(n)) / max(1, n - k)
    se = [math.sqrt(max(0.0, s2 * inv[p][p])) for p in range(k)]
    return b, se, math.sqrt(s2)


def wts(rows):
    return [r["dp"] ** 2 for r in rows]


def invert(mat):
    k = len(mat)
    a = [row[:] + [1.0 if i == j else 0.0 for j in range(k)] for i, row in enumerate(mat)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(a[r][c]))
        a[c], a[piv] = a[piv], a[c]
        if abs(a[c][c]) < 1e-30:
            raise SystemExit("singular fit (too few moves or no pressure range)")
        f = a[c][c]
        a[c] = [v / f for v in a[c]]
        for r in range(k):
            if r != c:
                g = a[r][c]
                a[r] = [vr - g * vc for vr, vc in zip(a[r], a[c])]
    return [row[k:] for row in a]


def median(v):
    v = sorted(v)
    n = len(v)
    return float("nan") if not n else (v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2]))


def branches(rows):
    out = {}
    for r in rows:
        out.setdefault((r["src"], r["section"], r["rep"]), []).append(r)
    return out


def crack_estimate(branch, c_line):
    """Line reading (mbar, > 0) at which the compliance steps up from the line's to line + chamber, or None."""
    c = [r["dv"] / r["dp"] for r in branch]
    hi = [ci for r, ci in zip(branch, c) if -110 <= r["pm"] <= -50]
    if len(hi) < 2 or len(branch) < 3:
        return None
    c_hi = median(hi)
    for i in range(1, len(branch)):
        if branch[i]["pm"] < -80:
            break
        thr_prev = 0.5 * (c_line(branch[i - 1]["pm"]) + c_hi)
        thr = 0.5 * (c_line(branch[i]["pm"]) + c_hi)
        if c[i - 1] < thr_prev and c[i] >= thr:
            f = (thr - c[i - 1]) / (c[i] - c[i - 1]) if c[i] != c[i - 1] else 0.5
            return -(branch[i - 1]["pm"] + f * (branch[i]["pm"] - branch[i - 1]["pm"]))
    return None


def line_gate(groups, pa, gate):
    """Repeatability of the line between repeats -> spreads, their effect on C_struct / V_g0, GO / NO-GO."""
    lev, chg = [], []
    for key, rs in sorted(groups.items()):
        if len(rs) < 4:
            continue
        (a0, b0), _, _ = lstsq([r["pm"] for r in rs], [r["dv"] / r["dp"] for r in rs], [lambda x: 1.0, lambda x: x], wts(rs))
        lev.append(a0 - 150.0 * b0)
        chg.append(-300.0 * b0)
    if len(lev) < 2:
        return {"verdict": "n/a (fewer than 2 repeats)", "n": len(lev)}
    spread = (max(lev) - min(lev)) / (sum(lev) / len(lev))
    dx = pa / (pa - 300.0) ** 2 - 1.0 / pa
    out = {"n": len(lev), "levels": lev, "level_spread_rel": spread, "c_struct_unc": max(lev) - min(lev),
           "change_spread": max(chg) - min(chg), "gas_unc_ul": (max(chg) - min(chg)) / dx,
           "verdict": "GO" if spread <= gate else "NO-GO"}
    return out


def main():
    ap = argparse.ArgumentParser(description="G test: chamber compliance and gas volume")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--baseline", action="append", default=[], help="line-only run(s) (pump replaced by a union)")
    ap.add_argument("--line-from-push", action="store_true", help="line = the push branch of the same run(s)")
    ap.add_argument("--p-crack", default="auto", help="chamber above the line reading while pulling (mbar), or auto")
    ap.add_argument("--gate", type=float, default=0.03, help="largest relative spread of the line between repeats")
    ap.add_argument("--p-atm", type=float, default=1013.0, help="absolute pressure at 0 gauge (mbar)")
    ap.add_argument("--skip-first", type=int, default=1,
                    help="without a cracking-pressure estimate: drop the first N pull moves of each staircase")
    ap.add_argument("--min-dp", type=float, default=0.5, help="ignore moves with |dp| below this (mbar)")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--json", action="store_true", help="last line: the fit as JSON")
    a = ap.parse_args()
    if not a.baseline and not a.line_from_push:
        raise SystemExit("give --baseline <run_dir> or --line-from-push")
    pa = a.p_atm

    rows = [r for run in a.runs for r in load(run) if abs(r["dp"]) >= a.min_dp]
    pull_all = [r for r in rows if r["direction"] == "pull"]
    if a.baseline:
        base = [r for b in a.baseline for r in load(b) if abs(r["dp"]) >= a.min_dp]
        line_rows = [r for r in base if r["direction"] == "pull"]
        base_push = [r for r in base if r["direction"] == "push"]
    else:
        line_rows, base_push = [r for r in rows if r["direction"] == "push"], []
    if len(pull_all) < 4 or len(line_rows) < 3:
        raise SystemExit(f"too few moves: {len(pull_all)} pull, {len(line_rows)} line")

    # line compliance: straight line in the line pressure
    lb, lse, lsd = lstsq([r["pm"] for r in line_rows], [r["dv"] / r["dp"] for r in line_rows],
                         [lambda x: 1.0, lambda x: x], wts(line_rows))
    c_line = lambda p: lb[0] + lb[1] * p
    src = "baseline pull moves" if a.baseline else "push branch"
    print(f"line ({src}, {len(line_rows)} moves): C_line = {lb[0]:.5f} + {lb[1]:.2e} x p uL/mbar "
          f"(at 0: {lb[0]:.5f}, at -300: {c_line(-300):.5f}; residual SD {lsd:.5f})")
    if base_push:
        bp = sum(r["dv"] / r["dp"] for r in base_push) / len(base_push)
        bl = sum(r["dv"] / r["dp"] for r in line_rows) / len(line_rows)
        print(f"baseline push vs pull mean compliance: {bp:.5f} vs {bl:.5f} uL/mbar ({bp / bl - 1:+.1%}; "
              f"a direction effect, e.g. the plunger seal)")

    # gate: line repeatability between repeats
    gate = line_gate(branches(line_rows), pa, a.gate)
    if gate["n"] >= 2:
        print(f"GATE {gate['verdict']}: the line's level differs by {gate['level_spread_rel']:.1%} between "
              f"{gate['n']} repeats (limit {a.gate:.0%}) -> C_struct uncertain by ~{gate['c_struct_unc']:.5f} "
              f"uL/mbar; its change from 0 to -300 mbar differs by {gate['change_spread']:.5f} -> V_g0 uncertain by "
              f"~{gate['gas_unc_ul']:.2f} uL")
    else:
        print(f"GATE {gate['verdict']}")

    # chamber offset while pulling: the outlet valve's cracking pressure (+ height)
    pcr_list = []
    if str(a.p_crack).lower() == "auto":
        for key, br in branches(pull_all).items():
            e = crack_estimate(br, c_line)
            if e is not None:
                pcr_list.append(e)
        pcr = median(pcr_list) if pcr_list else 0.0
        how = (f"auto, median of {len(pcr_list)} staircases: " + ", ".join(f"{e:.1f}" for e in pcr_list)
               if pcr_list else "auto: not found (no fine moves or no step), 0 used")
    else:
        pcr, how = float(a.p_crack), "given"
    print(f"cracking pressure P_cr = {pcr:.1f} mbar ({how})")
    if pcr_list:
        pulls = [r for r in pull_all if r["p1"] <= -pcr - 1.0]
    else:
        pulls, n_in = [], {}
        for r in pull_all:
            k = (r["src"], r["section"], r["rep"])
            n_in[k] = n_in.get(k, 0) + 1
            if n_in[k] > a.skip_first:
                pulls.append(r)
    if len(pulls) < 4:
        raise SystemExit(f"too few pull moves after the valve opened: {len(pulls)}")

    xg = lambda p1, p2, off: pa / ((pa + p1 + off) * (pa + p2 + off))
    xs = [xg(r["p1"], r["p2"], pcr) for r in pulls]
    ys = [r["dv"] / r["dp"] - c_line(r["pm"]) for r in pulls]
    (cs, vg), (se_cs, se_vg), sd = lstsq(xs, ys, [lambda x: 1.0, lambda x: x], wts(pulls))
    near0 = [r["dv"] / r["dp"] for r in pulls if r["pm"] > -60]
    c_tot0 = sum(near0) / len(near0) if near0 else float("nan")
    print(f"pull moves used: {len(pulls)} (p from {max(r['p1'] for r in pulls):.0f} to "
          f"{min(r['p2'] for r in pulls):.0f} mbar); total compliance near 0: {c_tot0:.5f} uL/mbar")
    print(f"chamber: C_struct = {cs:.5f} +- {se_cs:.5f} uL/mbar ({cs * 1e-11:.2e} m^3/Pa); "
          f"gas at 1 atm V_g0 = {vg:.2f} +- {se_vg:.2f} uL; residual SD {sd:.5f} uL/mbar (fit SE only; "
          f"the gate gives the line's share)")
    if c_tot0 == c_tot0:
        print(f"chamber share of the total compliance near 0 mbar: {(cs + vg / pa) / c_tot0:.0%}")
    (cs0, vg0), _, _ = lstsq([xg(r["p1"], r["p2"], 0.0) for r in pulls], ys, [lambda x: 1.0, lambda x: x], wts(pulls))
    (csh, vgh), _, _ = lstsq([xg(r["p1"], r["p2"], 0.5 * pcr) for r in pulls], ys, [lambda x: 1.0, lambda x: x],
                             wts(pulls))
    print(f"without the P_cr correction: V_g0 {vg0:.2f} uL, C_struct {cs0:.5f}; with half of it: V_g0 {vgh:.2f} uL")
    if pcr_list:
        print("  note: after an inlet reset the chamber refills through the inlet valve to about -P_cr(inlet), so the "
              "step found measures P_cr(outlet) + P_cr(inlet), an upper bound on the offset while pulling. If the "
              "valves match, half is right: the truth lies between the half and full results.")

    # The line subtraction dominates the uncertainty: refit with other line models
    lx = [xg(r["p1"], r["p2"], 0.0) for r in line_rows]
    lc = [r["dv"] / r["dp"] for r in line_rows]
    alt = {}
    for name, cols in (("constant", [lambda x: 1.0]), ("const + gas term", [lambda x: 1.0, lambda x: x])):
        b, _, _ = lstsq(lx, lc, cols, wts(line_rows))
        if name == "constant":
            line_f = lambda r, b=b: b[0]
        else:
            line_f = lambda r, b=b: b[0] + b[1] * xg(r["p1"], r["p2"], 0.0)
        y2 = [r["dv"] / r["dp"] - line_f(r) for r in pulls]
        (cs2, vg2), _, _ = lstsq(xs, y2, [lambda x: 1.0, lambda x: x], wts(pulls))
        alt[name] = {"c_struct": cs2, "gas_ul": vg2}
    print("other line models: " + "; ".join(f"{k}: V_g0 {v['gas_ul']:.2f} uL, C_struct {v['c_struct']:.5f}"
                                            for k, v in alt.items()))

    def level(lo, hi):
        sel = [y for r, y in zip(pulls, ys) if lo <= r["pm"] <= hi]
        return (sum(sel) / len(sel), len(sel)) if sel else (float("nan"), 0)
    (c0, n0), (c3, n3) = level(-60, 0), level(-330, -240)
    two = {}
    if n0 and n3:
        x0, x3 = pa / (pa - 30.0 + pcr) ** 2, pa / (pa - 285.0 + pcr) ** 2
        vg2 = (c3 - c0) / (x3 - x0)
        two = {"c_near_0": c0, "n0": n0, "c_near_m300": c3, "n300": n3, "gas_ul": vg2, "c_struct": c0 - vg2 * x0}
        print(f"two levels (brief): chamber {c0:.5f} uL/mbar near 0 (n {n0}), {c3:.5f} near -285 (n {n3}) -> "
              f"V_g0 {vg2:.2f} uL, C_struct {two['c_struct']:.5f}")
    drifts = sorted(abs(float(r["drift_mbar_s"])) for r in pulls if r.get("drift_mbar_s"))
    if drifts:
        print(f"drift during the holds: median {drifts[len(drifts) // 2]:.3f} mbar/s, max {drifts[-1]:.3f} "
              f"(creep up at a fixed volume = gas growing or a leak)")
    fl = sorted(float(r["dv_flow_ul"]) / -r["dv"] for r in pulls if r.get("dv_flow_ul") and abs(r["dv"]) > 0.1)
    if fl:
        print(f"flow-sensor volume / syringe volume (moves > 0.1 uL): median {fl[len(fl) // 2]:.2f} "
              f"(n {len(fl)}; ~1 = the steps move what they should)")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot([r["pm"] for r in pull_all], [r["dv"] / r["dp"] for r in pull_all], "o", ms=4, mfc="none",
                label="pull: all moves")
        ax.plot([r["pm"] for r in pulls], [r["dv"] / r["dp"] for r in pulls], "o", ms=4, label="pull: valve open")
        ax.plot([r["pm"] for r in line_rows], [r["dv"] / r["dp"] for r in line_rows], "s", ms=4, mfc="none",
                label="line (" + src + ")")
        ax.plot([r["pm"] for r in pulls], ys, "^", ms=4, label="chamber = pull - line")
        ps = [-320 + i for i in range(321)]
        ax.plot(ps, [cs + vg * pa / (pa + p + pcr) ** 2 for p in ps], "-", lw=1,
                label=f"fit: C_struct {cs:.4f}, V_g0 {vg:.2f} uL")
        if pcr:
            ax.axvline(-pcr, color="0.6", lw=0.8, ls="--")
        ax.set_xlabel("line pressure (mbar gauge)")
        ax.set_ylabel("compliance (uL/mbar)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=150)
        print(f"plot: {a.plot}")
    if a.json:
        print(json.dumps({"gas_ul": vg, "se_gas_ul": se_vg, "c_struct_ul_mbar": cs, "se_c_struct": se_cs,
                          "p_crack_mbar": pcr, "p_crack_estimates": pcr_list, "gas_ul_no_pcr": vg0, "gas_ul_half_pcr": vgh,
                          "c_line_at0": lb[0], "c_line_slope": lb[1], "n_pull": len(pulls), "n_line": len(line_rows),
                          "c_total_near0": c_tot0, "two_level": two, "other_line_models": alt, "gate": gate}))


if __name__ == "__main__":
    sys.exit(main())
