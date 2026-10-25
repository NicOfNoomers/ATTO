#!/usr/bin/env python3
"""Check a run's commanded drive lengths against the config, from the command timestamps.

    python scripts/check_bursts.py <run folder>

Every frequency command is logged in events.csv with t_cmd_s (run clock just before the serial write). A drive's
commanded length is the next command's t_cmd_s minus its own. Compared with the drive block's duration_s in the
run's config.json; FLAG when they differ by more than 0.5 ms. Also prints the strokes that length gives,
f x length + 0.25 (the drives stop a quarter period early). This checks the host side only: how late the Arduino
starts and stops the square wave is not visible here (oscilloscope across the coil for that).
"""
import csv
import json
import os
import sys

TOL_S = 0.5e-3


def main(run_dir):
    rows = list(csv.DictReader(open(os.path.join(run_dir, "events.csv"))))
    if not rows or "t_cmd_s" not in rows[0]:
        print("FAIL: events.csv has no t_cmd_s column (old dashboard code?)")
        return 1
    cmds = []                                   # ordered unique commands (t_cmd_s, t_cmd_done_s, Hz)
    for r in rows:
        if not r.get("t_cmd_s"):
            continue
        c = (float(r["t_cmd_s"]), float(r["t_cmd_done_s"] or "nan"), int(float(r["freq_set_hz"])))
        if not cmds or abs(c[0] - cmds[-1][0]) > 1e-9:
            cmds.append(c)
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    blocks = [t for t in cfg.get("tests", []) if (t.get("type") or "constant") == "constant"
              and int(t.get("frequency_hz", 0)) > 0]
    starts = [i for i, c in enumerate(cmds) if c[2] > 0]
    print(f"{run_dir}\n  {len(starts)} drive commands in events.csv, {len(blocks)} drive blocks in config.json")
    flags = 0
    for j, i in enumerate(starts):
        t0, done0, hz = cmds[i]
        if i + 1 >= len(cmds):
            print(f"  drive {j + 1}: {hz} Hz, no stop command logged")
            flags += 1
            continue
        length = cmds[i + 1][0] - t0
        nom = float(blocks[j]["duration_s"]) if j < len(blocks) else float("nan")
        n_nom = blocks[j].get("strokes") if j < len(blocks) else None
        dev = length - nom
        bad = not abs(dev) <= TOL_S
        flags += bad
        print(f"  drive {j + 1}: {hz:4d} Hz, commanded {length * 1000:10.2f} ms vs {nom * 1000:10.2f} ms "
              f"({dev * 1000:+.2f} ms) -> {hz * length + 0.25:8.2f} strokes (config {n_nom}); write took "
              f"{(done0 - t0) * 1000:.2f} ms{'   FLAG' if bad else ''}")
    print(f"  {'FAIL' if flags else 'OK'}: {flags} drive(s) off by more than {TOL_S * 1000:.1f} ms")
    return 1 if flags else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
