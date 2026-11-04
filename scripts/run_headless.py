"""Run a dashboard config without anyone at the PC: the real dashboard, window hidden, started, watched,
stopped safely and summarised. For the lab PC's Claude session (skill .claude/skills/atto-lab-runner).

    py -3.10 scripts\\run_headless.py config_lowf_voltage.json --name lowf_0p4 --pump-id 261030-1 --dry-run
    py -3.10 scripts\\run_headless.py config_lowf_voltage.json --name lowf_0p4 --pump-id 261030-1

What it does
- Loads the config into a hidden DashboardApp exactly as the Load button does (panels included: supply mode,
  address, power, video), so the run is the same as one started by hand. The panel fields can be overridden
  with --name / --pump-id / --operator / --out / --no-video.
- Checks before anything is switched on, and refuses (exit 2) when one fails:
  * a software trip: the PPK2 watch (power.enabled) or the programmable supply's CC trip (supply.mode
    "rsd3305p"); never a run without either
  * supply.max_mv <= 2500 and no voltage above it
  * no step that needs a person: manual-supply voltage blocks, "pause": true blocks, pressure / PQ blocks
    (syringe clearing) are refused unless --on-prompt wait (someone is at the rig to answer in the window)
- --dry-run stops after the checks: prints the plan (duration, voltage steps, supply, power) and exits 0.
- During the run: progress every --report-s seconds; a prompt that appears anyway (operator, voltage, clear
  syringe) stops the run (pump 0 Hz) unless --on-prompt wait; Ctrl+C stops it the same way as Stop.
- Every dialog the dashboard raises is printed (errors, warnings) and questions are answered NO.
- At the end: run folder, final status.json, voltage steps, errors; headless_summary.json in the run folder.
  Exit 0 = finished, 3 = stopped / failed / prompt, 2 = refused before start.
Close the dashboard window (or at least have no session connected in it) before a headless run: the two would
fight over the serial ports, the PPK2, the Fluigent sensor and the camera.
"""
import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import pump_test_dashboard as d  # noqa: E402

MAX_MV = 2500


def out(msg=""):
    print(f"[{datetime.now():%H:%M:%S}] {msg}" if msg else "", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run a dashboard config headless (window hidden)")
    ap.add_argument("config", help="config JSON (e.g. config_lowf_voltage.json)")
    ap.add_argument("--name", default=None, help="run name (Test name field); default: the config's meta.name")
    ap.add_argument("--pump-id", default=None, help="Pump ID field (needed when the config requires it)")
    ap.add_argument("--operator", default=None)
    ap.add_argument("--out", default=None, help="output folder (default: the dashboard's Output folder)")
    ap.add_argument("--no-video", action="store_true", help="don't record video even if the config asks")
    ap.add_argument("--dry-run", action="store_true", help="checks and plan only; nothing is opened")
    ap.add_argument("--on-prompt", choices=("stop", "wait"), default="stop",
                    help="an operator prompt during the run: stop it (default) or wait for someone at the rig")
    ap.add_argument("--report-s", type=float, default=60.0, help="progress line every N s")
    ap.add_argument("--max-h", type=float, default=12.0, help="stop the run after this many hours (safety net)")
    a = ap.parse_args(argv)

    dialogs = []

    def note(kind):
        def f(title="", msg="", *x, **k):
            dialogs.append({"kind": kind, "title": str(title), "msg": str(msg)})
            out(f"DIALOG {kind}: {title}: {str(msg)[:400]}")
            return False if kind in ("question",) else None
        return f

    d.messagebox.showerror = note("error")
    d.messagebox.showwarning = note("warning")
    d.messagebox.showinfo = note("info")
    for q in ("askokcancel", "askyesno", "askyesnocancel", "askretrycancel"):
        setattr(d.messagebox, q, note("question"))

    cfg_path = os.path.abspath(a.config)
    app = d.DashboardApp(initial_config_path=cfg_path)
    app.root.withdraw()

    def pump(s):
        t_end = time.monotonic() + s
        while time.monotonic() < t_end:
            app.root.update()
            time.sleep(0.05)

    pump(1.0)
    if [x for x in dialogs if x["kind"] == "error"]:
        out("REFUSED: the config did not load cleanly (see the dialogs above)")
        return 2
    if a.name:
        app.var_test_name.set(a.name)
    if a.pump_id is not None:
        app.var_pump_id.set(a.pump_id)
    if a.operator is not None and hasattr(app, "var_operator"):
        app.var_operator.set(a.operator)
    if a.out:
        app.var_out_dir.set(a.out)
    if a.no_video:
        app.var_video_enabled.set(False)

    cfg = app.build_config_from_ui()
    tests = cfg.get("tests") or []
    supply = {**d.SUPPLY_DEFAULTS, **(cfg.get("supply") or {})}
    power = cfg.get("power") or {}
    auto = d.supply_is_auto(supply)
    problems = []
    if not tests:
        problems.append("no test blocks")
    if not (power.get("enabled") or auto):
        problems.append("no software trip: power (PPK2 watch) is off and the supply is not the programmable one")
    if d.safe_float(supply.get("max_mv"), MAX_MV) > MAX_MV:
        problems.append(f"supply.max_mv {supply.get('max_mv')} is above {MAX_MV} mV")
    total, changes, _ = d.plan_schedule(tests, float(cfg.get("break_seconds_between_tests", 0.0)), auto=auto)
    too_high = sorted({c["mv"] for c in changes if c["mv"] > min(MAX_MV, d.safe_float(supply.get("max_mv"), MAX_MV))})
    if too_high:
        problems.append(f"voltage steps above the maximum: {too_high} mV")
    if d.safe_float(supply.get("initial_mv") or 0, 0.0) > MAX_MV:
        problems.append(f"supply.initial_mv {supply.get('initial_mv')} is above {MAX_MV} mV")
    needs_person = []

    def walk(blocks):
        for b in blocks:
            t = str(b.get("type", "")).lower()
            if t == "group":
                walk(b.get("blocks") or [])
            elif t == "voltage" and (not auto or b.get("pause")):
                needs_person.append("voltage step " + ("with pause" if auto else "on the manual supply"))
    walk(tests)
    if d.contains_pressure_block(tests):
        needs_person.append("pressure / PQ blocks (syringe clearing by hand)")
    if needs_person and a.on_prompt == "stop":
        problems.append("needs a person at the rig: " + ", ".join(sorted(set(needs_person)))
                        + " (use --on-prompt wait only if someone is there)")
    if cfg.get("require_pump_id") and not (cfg.get("meta") or {}).get("pump_id"):
        problems.append("the config requires a pump ID: --pump-id")

    out(f"config {cfg_path}")
    out(f"run name {app.var_test_name.get()!r}, pump {(cfg.get('meta') or {}).get('pump_id')!r}, "
        f"output folder {app.var_out_dir.get()!r}, video {'on' if app.var_video_enabled.get() else 'off'}")
    out(f"{len(tests)} blocks, planned {total / 60:.1f} min" + (f" + {len(changes)} voltage steps "
        f"{[c['mv'] for c in changes][:20]}" if changes else ""))
    out(f"supply: mode {supply.get('mode')}, address {supply.get('address')!r}, limit {supply.get('current_limit_ma')} mA, "
        f"max {supply.get('max_mv')} mV, start {supply.get('initial_mv')} mV")
    out(f"power: {'ON' if power.get('enabled') else 'off'}, backend {power.get('backend')}, trip "
        f"{power.get('watch_ma')} mA / {power.get('watch_hold_ms')} ms")
    if problems:
        for p in problems:
            out(f"REFUSED: {p}")
        app.root.destroy()
        return 2
    if a.dry_run:
        out("dry run: checks passed, nothing opened")
        app.root.destroy()
        return 0

    stop_reason = []

    def on_signal(*_):
        stop_reason.append("Ctrl+C / terminate")

    signal.signal(signal.SIGINT, on_signal)
    try:
        signal.signal(signal.SIGTERM, on_signal)
    except Exception:
        pass

    n0 = len(dialogs)
    app.start_scheduled_run()
    runner, run_dir = app.active_stream, app.run_dir
    if not isinstance(runner, d.PumpTestRunner):
        out("FAILED to start: " + "; ".join(f"{x['title']}: {x['msg'][:200]}" for x in dialogs[n0:]))
        app.root.destroy()
        return 3
    out(f"STARTED: {run_dir}")
    t0 = time.monotonic()
    last = t0
    stopped = False
    while app.active_stream is not None:
        pump(0.2)
        p = runner.get_progress()
        prompt = p.get("voltage_request") or p.get("operator_request")
        if prompt and not stopped:
            if a.on_prompt == "stop":
                stop_reason.append(f"operator prompt: {json.dumps(prompt, default=str)[:300]}")
            elif time.monotonic() - last >= a.report_s:
                out(f"waiting for the operator: {json.dumps(prompt, default=str)[:300]}")
        if time.monotonic() - t0 > a.max_h * 3600 and not stopped:
            stop_reason.append(f"--max-h {a.max_h} reached")
        if stop_reason and not stopped:
            out(f"STOPPING: {stop_reason[-1]}")
            app.stop_active()
            stopped = True
        if time.monotonic() - last >= a.report_s:
            last = time.monotonic()
            pw = runner.get_power_live() or {}
            sp = runner.get_supply_live() or {}
            out(f"{p['elapsed_s'] / 60:.1f} min, nominal {p['nominal_done_s'] / 60:.1f}/"
                f"{(p['total_s'] / 60 if p['total_s'] != float('inf') else float('nan')):.1f} min, "
                f"{p.get('changes_done', 0)} steps, {app.var_status.get()[:120]!r}"
                + (f", PPK2 {pw.get('i_ma', float('nan')):.1f} mA" if pw else "")
                + (f", supply {sp.get('v', sp.get('v_out_v', float('nan')))} V" if sp else ""))
    pump(1.5)

    status = {}
    try:
        status = json.load(open(os.path.join(run_dir, "status.json"), encoding="utf-8"))
    except Exception:
        pass
    steps = []
    try:
        import csv
        steps = list(csv.DictReader(open(os.path.join(run_dir, "voltage_changes.csv"), encoding="utf-8")))
    except Exception:
        pass
    errors = [x for x in dialogs[n0:] if x["kind"] in ("error", "warning")]
    finished = status.get("status") == "finished" and not stop_reason and not errors
    summary = {"run_dir": run_dir, "config": cfg_path, "finished": finished, "status": status.get("status"),
               "elapsed_s": status.get("elapsed_s"), "stop_reason": stop_reason, "errors": errors,
               "voltage_steps": [{k: s.get(k) for k in ("requested_mv", "confirmed_mv", "set_by", "wait_s")} for s in steps],
               "power": status.get("power"), "supply": status.get("supply")}
    try:
        with open(os.path.join(run_dir, "headless_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
    except Exception:
        pass
    out(f"{'FINISHED' if finished else 'ENDED'}: {run_dir}")
    out(f"status {status.get('status')}, {status.get('elapsed_s')} s; steps "
        + ", ".join(f"{s.get('requested_mv')}->{s.get('confirmed_mv')}" for s in steps))
    for e in errors:
        out(f"error: {e['title']}: {e['msg'][:300]}")
    app.root.destroy()
    return 0 if finished else 3


if __name__ == "__main__":
    sys.exit(main())
