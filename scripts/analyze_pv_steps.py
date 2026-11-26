"""G test analysis: chamber compliance and gas volume from pv_steps.csv (syringe volume steps).

Per move: C = dv_line / dp (uL/mbar), with dv_line the syringe volume pushed into the line (counts x
ul_per_step / microsteps) and dp the change of the held pressure (means over the end of each hold).
  pull branch (outlet valve open, chamber follows):  C_pull = C_line + C_struct + C_gas(p)
  push branch (outlet valve closed, chamber held):   C_push = C_line (outlet line only)
  baseline run (pump replaced by a stiff union):     C_base = C_line + the inlet stub
Gas, isothermal: V = V_g0 p_atm / p_abs, so over a step from p1 to p2 (gauge) its secant compliance is
V_g0 p_atm / ((p_atm + p1)(p_atm + p2)). Fit, by least squares over the pull moves,
  C_pull - C_line(p) = C_struct + V_g0 x,   x = p_atm / ((p_atm + p1)(p_atm + p2))
C_line(p): a straight line in p fitted to the baseline run (--baseline, preferred: it includes the inlet stub)
or to the push branch of the same run (--line-from-push; misses the inlet stub, so C_struct comes out too
high by the stub's compliance).
The two-level numbers of the brief (mean chamber compliance near 0 and near -300 mbar) are printed as well.
1 uL/mbar = 1e-11 m^3/Pa.

usage: python scripts/analyze_pv_steps.py <run_dir> [<run_dir> ...] (--baseline <run_dir> | --line-from-push)
       [--p-atm 1013] [--skip-first 1] [--min-dp 0.5] [--plot fig.png] [--json]
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


def lstsq(xs, ys, cols):
    """Least squares y = sum b_j f_j(x); returns (b, se, resid_sd)."""
    n, k = len(xs), len(cols)
    A = [[f(x) for f in cols] for x in xs]
    ata = [[sum(A[i][p] * A[i][q] for i in range(n)) for q in range(k)] for p in range(k)]
    aty = [sum(A[i][p] * ys[i] for i in range(n)) for p in range(k)]
    inv = invert(ata)
    b = [sum(inv[p][q] * aty[q] for q in range(k)) for p in range(k)]
    res = [ys[i] - sum(b[p] * A[i][p] for p in range(k)) for i in range(n)]
    s2 = sum(e * e for e in res) / max(1, n - k)
    se = [math.sqrt(max(0.0, s2 * inv[p][p])) for p in range(k)]
    return b, se, math.sqrt(s2)


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


def main():
    ap = argparse.ArgumentParser(description="G test: chamber compliance and gas volume")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--baseline", action="append", default=[], help="line-only run(s) (pump replaced by a union)")
    ap.add_argument("--line-from-push", action="store_true", help="line = the push branch of the same run(s)")
    ap.add_argument("--p-atm", type=float, default=1013.0, help="absolute pressure at 0 gauge (mbar)")
    ap.add_argument("--skip-first", type=int, default=1,
                    help="drop the first N pull moves of each branch (the outlet valve's cracking pressure)")
    ap.add_argument("--min-dp", type=float, default=0.5, help="ignore moves with |dp| below this (mbar)")
    ap.add_argument("--plot", default=None)
    ap.add_argument("--json", action="store_true", help="last line: the fit as JSON")
    a = ap.parse_args()
    if not a.baseline and not a.line_from_push:
        raise SystemExit("give --baseline <run_dir> or --line-from-push")
    pa = a.p_atm

    rows = [r for run in a.runs for r in load(run) if abs(r["dp"]) >= a.min_dp]
    pulls, branch_n = [], {}
    for r in rows:
        if r["direction"] != "pull":
            continue
        key = (r["src"], r["section"], r["rep"])
        branch_n[key] = branch_n.get(key, 0) + 1
        if branch_n[key] > a.skip_first:
            pulls.append(r)
    line_rows = ([r for b in a.baseline for r in load(b) if abs(r["dp"]) >= a.min_dp] if a.baseline
                 else [r for r in rows if r["direction"] == "push"])
    if len(pulls) < 4 or len(line_rows) < 3:
        raise SystemExit(f"too few moves: {len(pulls)} pull, {len(line_rows)} line")

    # line compliance: straight line in the mid pressure
    lb, lse, lsd = lstsq([r["pm"] for r in line_rows], [r["dv"] / r["dp"] for r in line_rows],
                         [lambda x: 1.0, lambda x: x])
    c_line = lambda p: lb[0] + lb[1] * p
    print(f"line ({'baseline' if a.baseline else 'push branch'}, {len(line_rows)} moves): C_line = "
          f"{lb[0]:.5f} + {lb[1]:.2e} x p uL/mbar (at 0: {lb[0]:.5f}, at -300: {c_line(-300):.5f}; "
          f"residual SD {lsd:.5f})")

    xs = [pa / ((pa + r["p1"]) * (pa + r["p2"])) for r in pulls]
    ys = [r["dv"] / r["dp"] - c_line(r["pm"]) for r in pulls]
    (cs, vg), (se_cs, se_vg), sd = lstsq(xs, ys, [lambda x: 1.0, lambda x: x])
    c_tot0 = sum(r["dv"] / r["dp"] for r in pulls if r["pm"] > -60) / max(1, sum(1 for r in pulls if r["pm"] > -60))
    print(f"pull moves used: {len(pulls)} (p from {max(r['p1'] for r in pulls):.0f} to "
          f"{min(r['p2'] for r in pulls):.0f} mbar); total compliance near 0: {c_tot0:.5f} uL/mbar")
    print(f"chamber: C_struct = {cs:.5f} +- {se_cs:.5f} uL/mbar ({cs * 1e-11:.2e} m^3/Pa); "
          f"gas at 1 atm V_g0 = {vg:.2f} +- {se_vg:.2f} uL; residual SD {sd:.5f} uL/mbar")
    print(f"chamber share of the total compliance near 0 mbar: {(cs + vg / pa) / c_tot0:.0%}")
    # The line subtraction dominates the uncertainty: refit with other line models
    lx = [pa / ((pa + r["p1"]) * (pa + r["p2"])) for r in line_rows]
    lc = [r["dv"] / r["dp"] for r in line_rows]
    alt = {}
    for name, cols, feat in (("constant", [lambda x: 1.0], lambda r, x: x),
                             ("const + gas term", [lambda x: 1.0, lambda x: x], lambda r, x: x)):
        if name == "constant":
            b, _, _ = lstsq(lx, lc, cols)
            line_f = lambda r, b=b: b[0]
        else:
            b, _, _ = lstsq(lx, lc, cols)
            line_f = lambda r, b=b: b[0] + b[1] * pa / ((pa + r["p1"]) * (pa + r["p2"]))
        y2 = [r["dv"] / r["dp"] - line_f(r) for r in pulls]
        (cs2, vg2), _, _ = lstsq(xs, y2, [lambda x: 1.0, lambda x: x])
        alt[name] = {"c_struct": cs2, "gas_ul": vg2}
    print("other line models: " + "; ".join(f"{k}: V_g0 {v['gas_ul']:.2f} uL, C_struct {v['c_struct']:.5f}"
                                            for k, v in alt.items())
          + "  (the spread is the line's share of the uncertainty)")

    def level(lo, hi):
        sel = [y for r, y in zip(pulls, ys) if lo <= r["pm"] <= hi]
        return (sum(sel) / len(sel), len(sel)) if sel else (float("nan"), 0)
    (c0, n0), (c3, n3) = level(-60, 0), level(-330, -240)
    two = {}
    if n0 and n3:
        x0, x3 = pa / (pa - 30.0) ** 2, pa / (pa - 285.0) ** 2
        vg2 = (c3 - c0) / (x3 - x0)
        two = {"c_near_0": c0, "n0": n0, "c_near_m300": c3, "n300": n3, "gas_ul": vg2, "c_struct": c0 - vg2 * x0}
        print(f"two levels (brief): chamber {c0:.5f} uL/mbar near 0 (n {n0}), {c3:.5f} near -285 (n {n3}) -> "
              f"V_g0 {vg2:.2f} uL, C_struct {two['c_struct']:.5f}")
    drifts = sorted(abs(float(r["drift_mbar_s"])) for r in pulls if r.get("drift_mbar_s"))
    if drifts:
        print(f"drift during the holds: median {drifts[len(drifts) // 2]:.3f} mbar/s, max {drifts[-1]:.3f} "
              f"(creep up at a fixed volume = gas growing or a leak)")
    fl = [float(r["dv_flow_ul"]) / -r["dv"] for r in pulls if r.get("dv_flow_ul") and abs(r["dv"]) > 0.1]
    if fl:
        fl.sort()
        print(f"flow-sensor volume / syringe volume (moves > 0.1 uL): median {fl[len(fl) // 2]:.2f} "
              f"(n {len(fl)}; ~1 = the steps move what they should)")
    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.plot([r["pm"] for r in pulls], [r["dv"] / r["dp"] for r in pulls], "o", ms=4, label="pull: line + chamber")
        ax.plot([r["pm"] for r in line_rows], [r["dv"] / r["dp"] for r in line_rows], "s", ms=4, mfc="none",
                label="line (" + ("baseline" if a.baseline else "push branch") + ")")
        ax.plot([r["pm"] for r in pulls], ys, "^", ms=4, label="chamber = pull - line")
        ps = [-320 + i for i in range(321)]
        ax.plot(ps, [cs + vg * pa / (pa + p) ** 2 for p in ps], "-", lw=1,
                label=f"fit: C_struct {cs:.4f}, V_g0 {vg:.2f} uL")
        ax.set_xlabel("pressure (mbar gauge)")
        ax.set_ylabel("compliance (uL/mbar)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=150)
        print(f"plot: {a.plot}")
    if a.json:
        print(json.dumps({"gas_ul": vg, "se_gas_ul": se_vg, "c_struct_ul_mbar": cs, "se_c_struct": se_cs,
                          "c_line_at0": lb[0], "c_line_slope": lb[1], "n_pull": len(pulls), "n_line": len(line_rows),
                          "c_total_near0": c_tot0, "two_level": two, "other_line_models": alt}))


if __name__ == "__main__":
    sys.exit(main())
