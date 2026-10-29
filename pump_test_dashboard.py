#!/usr/bin/env python3
"""
Pump Testing Suite Dashboard (v2)

Adds:
1) Manual control mode ("testing OFF"):
   - Connect/disconnect
   - Output ON/OFF (0 Hz when OFF)
   - Set frequency in real time
   - Live flow plot (optional logging)

2) Config builder UI (no hand-editing JSON):
   - Add Constant / Add Sweep blocks
   - Add Group / Group Selected: repeat a sequence of blocks N times or forever
   - Add Voltage Step: pause the run until the operator has set the bench
     supply to a new voltage and confirmed it (the "Voltage schedule..."
     button lists when each change falls due, for setting alarms)
   - Edit selected block
   - Duplicate / Remove
   - Move Up / Move Down (organize order)
   - Save config JSON (compatible with tests[] schema)

3) Power meter (optional, "Power meter (PPK2)" panel / cfg["power"]):
   - Nordic PPK2 as an ammeter between the bench supply and the pump driver
     (power_ppk2.py); starts and stops with each run or manual session
   - logs current, power and energy to <run>/power/ (10 ms bins in rotating,
     gzipped files, 1 s summary, raw snapshots); live readout + current trace
   - stops the run (pump to 0 Hz) if the current sits at the bench supply's
     limit, i.e. the supply is clamping

Serial protocol:
- freq_only (default): sends "<int_hz>\n"
- freq_duty (optional): sends "F <hz>\n" and "D <duty>\n"

Flow sensor:
- Fluigent SDK backend if available, otherwise dummy generator for UI testing.
"""

from __future__ import annotations

import os
import sys
import atexit
import json
import time
import csv
import copy
import math
import threading
import queue
import logging
import subprocess
import importlib.util
from datetime import datetime, timedelta
from collections import deque
from typing import Optional, Dict, Any, List, Tuple, Callable

import numpy as np    # matplotlib requires it anyway

# ---------------- logging utils ----------------
LOG = logging.getLogger("pump_test_dashboard")
if not LOG.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    LOG.addHandler(_handler)
LOG.setLevel(logging.INFO)
LOG.propagate = False

_LOG_THROTTLE: Dict[str, float] = {}

def log_exception(context: str) -> None:
    LOG.exception(context)

def log_exception_limited(key: str, context: str, interval_s: float = 5.0) -> None:
    now = time.monotonic()
    last = _LOG_THROTTLE.get(key, 0.0)
    if now - last >= interval_s:
        _LOG_THROTTLE[key] = now
        LOG.exception(context)

def log_warning(context: str) -> None:
    LOG.warning(context)

def log_warning_limited(key: str, context: str, interval_s: float = 5.0) -> None:
    now = time.monotonic()
    last = _LOG_THROTTLE.get(key, 0.0)
    if now - last >= interval_s:
        _LOG_THROTTLE[key] = now
        LOG.warning(context)

def log_error(context: str) -> None:
    LOG.error(context)

# ---- optional deps ----
try:
    import serial  # pyserial
except Exception:
    serial = None
    log_exception("Optional import failed: pyserial not available; serial features disabled.")

try:
    import cv2  # opencv for video capture
    CV2_AVAILABLE = True
except Exception:
    CV2_AVAILABLE = False
    log_exception("Optional import failed: OpenCV (cv2) not available; camera features disabled.")

try:
    import psutil  # optional: system/process memory monitoring for crash-safety
    PSUTIL_AVAILABLE = True
except Exception:
    psutil = None
    PSUTIL_AVAILABLE = False
    log_warning("Optional import failed: psutil not available; memory watchdog disabled (data flushing still active).")

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

FLUIGENT_AVAILABLE = False
try:
    from Fluigent.SDK import fgt_init, fgt_close, fgt_get_sensorValue, fgt_set_errorReportMode
    FLUIGENT_AVAILABLE = True
    try:
        from Fluigent.SDK import fgt_get_sensorRange, fgt_get_sensorChannelCount, fgt_get_sensorUnit
    except Exception:
        fgt_get_sensorRange = fgt_get_sensorChannelCount = fgt_get_sensorUnit = None
except Exception:
    FLUIGENT_AVAILABLE = False
    log_exception("Optional import failed: Fluigent SDK not available; Fluigent backend disabled.")

NETWORK_FLOW_AVAILABLE = False
try:
    from flow_client import FlowClient, parse_server  # shared flow server client (needs websockets)
    NETWORK_FLOW_AVAILABLE = True
except Exception as _e:
    NETWORK_FLOW_AVAILABLE = False
    # Not an error: only the "network" flow backend needs it. Name the Python that
    # is running, since a missing package usually means a different Python than
    # the one install.bat installed into.
    log_warning(f"Network flow backend disabled ({_e}). This dashboard runs on {sys.executable}; "
                f"install with: \"{sys.executable}\" -m pip install -r requirements.txt")

# Packages a feature needs at run time: (import name, pip name, what for). The
# dashboard can install missing ones into the Python it runs on (Install button).
FEATURE_PACKAGES = [("ppk2_api", "ppk2-api", "PPK2 power meter"),
                    ("websockets", "websockets", "network flow backend")]
FLUIGENT_SDK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fluigent_sdk-23.0.0")


def missing_feature_packages() -> List[Tuple[str, str, str]]:
    importlib.invalidate_caches()
    missing = [p for p in FEATURE_PACKAGES if importlib.util.find_spec(p[0]) is None]
    if importlib.util.find_spec("Fluigent") is None and os.path.isdir(FLUIGENT_SDK_DIR):
        missing.append(("Fluigent", "fluigent_sdk-23.0.0", "Fluigent flow sensor"))
    return missing


from ui_panels import PanelManager     # dockable/collapsible sidebar panels (repo root)

POWER_AVAILABLE = False
try:
    from power_ppk2 import PowerMeter, POWER_DEFAULTS, find_ppk2_ports  # PPK2 power meter (needs numpy)
    POWER_AVAILABLE = True
except Exception:
    POWER_DEFAULTS = {"enabled": False}
    log_exception("Optional import failed: power_ppk2 not available; power meter disabled.")

PRESSURE_AVAILABLE = False
try:
    import pressure_control as pc      # syringe pressure control for PQ curves
    PRESSURE_AVAILABLE = True
except Exception:
    pc = None
    log_exception("Optional import failed: pressure_control not available; pressure control disabled.")


# ---------------- run clock ----------------
# Every run time (events, merged, flow, power, frames) is CLOCK() - t0. On
# Python 3.10 / Windows time.monotonic() only ticks every 15.6 ms
# (GetTickCount64); perf_counter is QueryPerformanceCounter (0.1 us).
CLOCK = time.perf_counter


def clock_info() -> Dict[str, Any]:
    """Clock details for meta.json (acceptance: both clocks recorded at run start)."""
    info = {"python": sys.version.split()[0], "run_clock": "perf_counter"}
    for name in ("monotonic", "perf_counter"):
        c = time.get_clock_info(name)
        info[name] = {"implementation": c.implementation, "resolution": c.resolution,
                      "monotonic": c.monotonic, "adjustable": c.adjustable}
    return info


_HIRES_USERS = 0
_HIRES_LOCK = threading.Lock()
_SWITCH_INTERVAL_DEFAULT = sys.getswitchinterval()
RUN_SWITCH_INTERVAL_S = 0.0005


def hires_timer(on: bool) -> None:
    """While a session runs:
    * Windows: 1 ms timer resolution. Without it time.sleep(0.001) takes
      ~15.4 ms on Python 3.10, so short steps run long.
    * A short GIL switch interval (0.5 ms instead of 5 ms): the plot redraw on
      the UI thread holds the GIL for tens of ms, and the runner thread would
      otherwise send its commands up to ~15 ms late."""
    global _HIRES_USERS
    with _HIRES_LOCK:
        if on:
            if _HIRES_USERS == 0:
                sys.setswitchinterval(RUN_SWITCH_INTERVAL_S)
                _winmm_period(True)
            _HIRES_USERS += 1
        elif _HIRES_USERS > 0:
            _HIRES_USERS -= 1
            if _HIRES_USERS == 0:
                sys.setswitchinterval(_SWITCH_INTERVAL_DEFAULT)
                _winmm_period(False)


def _winmm_period(on: bool) -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        if on:
            _keep_timer_when_hidden()
        winmm = ctypes.WinDLL("winmm")
        (winmm.timeBeginPeriod if on else winmm.timeEndPeriod)(1)
    except Exception:
        log_exception_limited("hires_timer", "Could not change the Windows timer resolution")


def _keep_timer_when_hidden() -> None:
    """Windows 11 ignores timeBeginPeriod for a process whose window is covered or
    minimized (sleeps fall back to 15.6 ms). Opt out of that power throttling:
    SetProcessInformation(ProcessPowerThrottling, IGNORE_TIMER_RESOLUTION)."""
    import ctypes
    from ctypes import wintypes

    class _ThrottlingState(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        fn = getattr(k32, "SetProcessInformation", None)
        if fn is None:                       # before Windows 8
            return
        fn.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        fn.restype = wintypes.BOOL
        state = _ThrottlingState(1, 0x4, 0)  # CURRENT_VERSION, IGNORE_TIMER_RESOLUTION, state off
        ok = fn(k32.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state))   # 4 = ProcessPowerThrottling
        if not ok:
            log_warning_limited("timer_throttle", f"SetProcessInformation failed ({ctypes.get_last_error()}); "
                                                  f"keep the dashboard window visible for 1 ms timing")
    except Exception:
        log_exception_limited("timer_throttle", "Could not opt out of timer throttling")


# ---------------- crash-safety / memory config ----------------
FSYNC_INTERVAL_S = 2.0     # force logged data to physical disk at least this often
MEM_POLL_MS = 3000         # memory watchdog poll interval (ms)
MEM_WARN_MB = 800.0        # flush data + warn when system free memory drops below this
MEM_CRITICAL_MB = 400.0    # flush data + shed load (stop video) below this


# ---------------- util ----------------
def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def sanitize_run_name(raw_name: str, default: str) -> str:
    base = str(raw_name).strip().replace(" ", "_")[:40]
    if not base:
        base = default
    return "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in base)

def create_unique_run_dir(out_dir: str, run_name: str, suffix: str = "") -> str:
    # Include microseconds, then fall back to a numeric suffix if needed.
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base_name = f"{run_tag}_{run_name}{suffix}"
    idx = 0
    while True:
        suffix_part = "" if idx == 0 else f"_{idx:02d}"
        candidate = os.path.join(out_dir, f"{base_name}{suffix_part}")
        try:
            os.makedirs(candidate, exist_ok=False)
            return candidate
        except FileExistsError:
            idx += 1

def safe_int(s: str, default: int) -> int:
    try:
        return int(float(str(s).strip()))
    except Exception:
        log_warning(f"safe_int: could not parse {s!r}; using default {default}")
        return default

def safe_float(s: str, default: float) -> float:
    try:
        return float(str(s).strip())
    except Exception:
        log_warning(f"safe_float: could not parse {s!r}; using default {default}")
        return default

def window_decimate(ts, cols, tmin: float, max_pts: int = 800) -> List["np.ndarray"]:
    """Plot data from tmin on (plus one point before) as arrays of at most ~max_pts points. When
    thinning, every bin of samples becomes two points, its min and its max, so peaks and the
    envelope survive (a plot ~800 px wide can't show more; long noisy lines are slow to draw)."""
    t = np.asarray(ts, dtype=float)
    i = max(0, int(np.searchsorted(t, tmin)) - 1)
    t = t[i:]
    ys = [np.asarray(c, dtype=float)[i:] for c in cols]
    n = len(t)
    if n <= max_pts:
        return [t] + ys
    k = -(-2 * n // max_pts)                  # samples per bin, 2 points out per bin
    m = (n // k) * k
    tb = t[:m].reshape(-1, k)
    out = [np.column_stack((tb[:, 0], tb[:, -1])).ravel()]
    for y in ys:
        yb = y[:m].reshape(-1, k)
        # fmin/fmax skip NaN (target line outside holds) without warnings
        out.append(np.column_stack((np.fmin.reduce(yb, axis=1), np.fmax.reduce(yb, axis=1))).ravel())
    if m < n:
        out = [np.concatenate((o, a[m:])) for o, a in zip(out, [t] + ys)]
    return out


def linspace_int(a: int, b: int, n: int) -> List[int]:
    if n <= 1:
        return [int(round(a))]
    step = (b - a) / (n - 1)
    return [int(round(a + i * step)) for i in range(n)]


# ---------------- schedule / manual voltage steps ----------------
# A "voltage" block pauses the run until the operator has set the bench supply
# to voltage_mv and confirmed it in the dashboard:
#   {"type": "voltage", "voltage_mv": 2000,
#    "while_waiting": "zero",        # "zero": pump at 0 Hz; "hold": keep the last setpoint running
#    "settle_s": 30,                 # after confirming, wait this long before the next block
#    "window": ["08:00", "20:00"]}   # optional: only ask during these hours
# Its nominal duration is settle_s. The operator's delay is unknown in advance,
# so run ETAs are "nominal time + window waits + your confirmation delays".
SUPPLY_DEFAULTS = {"mode": "manual", "current_limit_ma": 60, "max_mv": 2500, "lead_min": 10}
MAX_PLANNED_CHANGES = 2000   # cap for voltage steps inside long or endless groups
# Pressure holds: the settling time is not known in advance; the plan counts this much per
# hold (a PQ point = this + measure_s). Syringe clearing by hand comes on top.
PRESSURE_SETTLE_GUESS_S = 20.0
PQ_COLS = ["point", "block", "frequency_hz", "target_mbar", "p_mean_mbar", "p_sd_mbar", "flow_mean", "flow_sd",
           "flow_n", "syringe_steps_per_s", "syringe_flow_ul_min", "current_mean_ma", "settled", "t_settle_s",
           "t_start_s", "t_end_s", "pos_start", "pos_end", "syringe_clears", "note"]


def block_type(t: Dict[str, Any]) -> str:
    return (t.get("type") or "constant").lower()


def block_duration_s(t: Dict[str, Any]) -> float:
    """Nominal run time of one block including its repeats (inf for an endless group)."""
    ttype = block_type(t)
    repeat = safe_int(t.get("repeat", 1), 1)
    settle_s = max(0, safe_int(t.get("settle_ms", 0), 0)) / 1000.0
    if ttype == "group":
        if repeat <= 0:
            return math.inf
        return repeat * sum(block_duration_s(b) for b in t.get("blocks", []))
    if ttype == "voltage":
        return max(0.0, safe_float(t.get("settle_s", 0), 0.0))
    if ttype == "pressure":
        return PRESSURE_SETTLE_GUESS_S if t.get("wait_settled", True) and is_pressure_target(t) else 0.0
    if ttype == "pq":
        n = len(pq_frequencies(t)) * len(t.get("pressures_mbar") or [])
        return n * (PRESSURE_SETTLE_GUESS_S + max(0.0, safe_float(t.get("measure_s", 30), 30.0)))
    repeat = max(1, repeat)
    if ttype == "constant":
        return repeat * (safe_float(t.get("duration_s", 0), 0.0) + settle_s)
    steps = max(1, safe_int(t.get("steps", 1), 1))
    if "step_duration_s" in t:
        step_dur = safe_float(t.get("step_duration_s"), 0.0)
    else:
        step_dur = safe_float(t.get("total_duration_s", 0), 0.0) / steps
    return repeat * steps * (step_dur + settle_s)


def pq_frequencies(t: Dict[str, Any]) -> List[int]:
    f = t.get("frequency_hz", t.get("frequencies_hz", []))
    return [int(x) for x in (f if isinstance(f, (list, tuple)) else [f])]


def is_pressure_target(t: Dict[str, Any]) -> bool:
    """A 'pressure' block that holds a pressure (not release/idle)."""
    return not isinstance(t.get("target_mbar", "release"), str) and t.get("target_mbar") is not None


def contains_pressure_block(blocks: List[Dict[str, Any]]) -> bool:
    for b in blocks:
        tt = block_type(b)
        if tt in ("pressure", "pq") or (tt == "group" and contains_pressure_block(b.get("blocks", []))):
            return True
    return False


def contains_voltage_block(blocks: List[Dict[str, Any]]) -> bool:
    for b in blocks:
        tt = block_type(b)
        if tt == "voltage" or (tt == "group" and contains_voltage_block(b.get("blocks", []))):
            return True
    return False


def _parse_hhmm(s: Any) -> Optional[int]:
    """'08:30' -> minutes after midnight (24:00 allowed as an end), else None."""
    try:
        h, m = str(s).strip().split(":")
        mins = int(h) * 60 + int(m)
        return mins if 0 <= int(m) < 60 and 0 <= mins <= 24 * 60 else None
    except Exception:
        return None


def voltage_window(t: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Daily (start, end) window in minutes for a voltage block, or None for any time."""
    w = t.get("window")
    if not isinstance(w, (list, tuple)) or len(w) != 2:
        return None
    a, b = _parse_hhmm(w[0]), _parse_hhmm(w[1])
    if a is None or b is None or a == b:
        return None
    return a, b


def next_in_window(dt: datetime, window: Optional[Tuple[int, int]]) -> datetime:
    """Earliest time >= dt inside the daily window (windows may run past midnight)."""
    if window is None:
        return dt
    a, b = window
    m = dt.hour * 60 + dt.minute + (dt.second + dt.microsecond / 1e6) / 60.0
    inside = (a <= m < b) if a < b else (m >= a or m < b)
    if inside:
        return dt
    opens = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=a)
    return opens if opens > dt else opens + timedelta(days=1)


def plan_schedule(tests: List[Dict[str, Any]], break_between: float = 0.0
                  ) -> Tuple[float, List[Dict[str, Any]], bool]:
    """Nominal timeline of a test sequence -> (total_s, voltage changes, truncated).

    total_s is inf if the sequence reaches an endless group. Each change has
    t_nom: the nominal running time before it (window waits and confirmation
    delays excluded), the same quantity PumpTestRunner counts in nominal_done,
    so a live ETA is now + (t_nom - nominal_done) + waits (project_changes).
    """
    changes: List[Dict[str, Any]] = []
    t = 0.0
    endless = False

    def walk(blocks, where: str, cycle: str, first: int = 1) -> bool:
        nonlocal t, endless
        for j, b in enumerate(blocks):
            if len(changes) >= MAX_PLANNED_CHANGES:
                return False
            tt = block_type(b)
            label = f"{where}{j + first}"
            if tt == "voltage":
                changes.append({
                    "index": len(changes) + 1, "t_nom": t,
                    "mv": int(round(safe_float(b.get("voltage_mv", 0), 0.0))),
                    "window": voltage_window(b),
                    "hold": str(b.get("while_waiting", "zero")).lower() == "hold",
                    "where": f"block {label}{cycle}",
                })
                t += block_duration_s(b)
            elif tt == "group":
                kids = b.get("blocks", [])
                if not kids:
                    continue
                rep = safe_int(b.get("repeat", 1), 1)
                if not contains_voltage_block(kids):
                    t += block_duration_s(b)
                    if math.isinf(t):
                        endless = True
                        return False
                    continue
                n = 0
                while rep <= 0 or n < rep:
                    n += 1
                    if not walk(kids, f"{label}.", f", cycle {n}/{rep if rep > 0 else 'inf'}"):
                        if rep <= 0:
                            endless = True
                        return False
            else:
                t += block_duration_s(b)
        return True

    # Top-level blocks one at a time: the runner pauses between them, not inside groups
    for i, b in enumerate(tests):
        if not walk([b], "", "", first=i + 1):
            break
        if i < len(tests) - 1 and break_between > 0:
            t += break_between
    total = math.inf if endless else t
    return total, changes, len(changes) >= MAX_PLANNED_CHANGES


def project_changes(changes: List[Dict[str, Any]], now: datetime, nominal_now: float, done: int
                    ) -> Tuple[List[Tuple[Dict[str, Any], datetime, bool]], float]:
    """Clock time of each change not yet done, applying window waits in order.

    Returns ([(change, due, moved_by_window)], total window wait in s). Each
    window wait pushes every later change back by the same amount.
    """
    out = []
    shift = 0.0
    for c in changes[done:]:
        due = now + timedelta(seconds=max(0.0, c["t_nom"] - nominal_now) + shift)
        opens = next_in_window(due, c["window"])
        moved = opens > due
        if moved:
            shift += (opens - due).total_seconds()
        out.append((c, opens, moved))
    return out, shift


def fmt_duration(s: Optional[float]) -> str:
    if s is None or math.isinf(s):
        return "until Stop"
    s = int(round(max(0.0, s)))
    d, r = divmod(s, 86400)
    h, r = divmod(r, 3600)
    m, sec = divmod(r, 60)
    return f"{d} d {h:02d}:{m:02d}:{sec:02d}" if d else f"{h}:{m:02d}:{sec:02d}"


def fmt_clock(dt: datetime) -> str:
    return dt.strftime("%a %d %b %H:%M")


def format_schedule(total_s: float, proj: List[Tuple[Dict[str, Any], datetime, bool]], shift_s: float,
                    now: datetime, nominal_now: float, supply: Dict[str, Any], truncated: bool,
                    heading: str) -> str:
    """Human-readable voltage-change schedule (also saved as schedule.txt)."""
    lines = [heading, ""]
    if math.isinf(total_s):
        lines.append(f"Runs until Stop (endless group). Now: {fmt_clock(now)}")
    else:
        remaining = max(0.0, total_s - nominal_now) + shift_s
        extra = " + your confirmation delays" if proj else ""
        lines.append(f"Run time left: {fmt_duration(remaining)}{extra}")
        lines.append(f"Now: {fmt_clock(now)}    Earliest end: {fmt_clock(now + timedelta(seconds=remaining))}")
    lines.append("")
    if not proj:
        lines.append("No voltage changes left in this sequence.")
        return "\n".join(lines)
    lines.append(f"Voltage changes (bench supply current limit "
                 f"{safe_float(supply.get('current_limit_ma', 0), 0.0):g} mA):")
    lines.append(f"  {'#':>4}  {'due':<18}  {'set supply to':<13}  where")
    for c, due, moved in proj:
        mark = " *" if moved else "  "
        pump = "pump holds" if c["hold"] else "pump 0 Hz"
        lines.append(f"  {c['index']:>4}  {fmt_clock(due):<16}{mark}  {c['mv'] / 1000:>8.3f} V     "
                     f"{c['where']} ({pump})")
    if truncated:
        lines.append(f"  ... list capped at {MAX_PLANNED_CHANGES} changes")
    lines.append("")
    if any(m for _, _, m in proj):
        lines.append("* moved to the start of its allowed time window.")
    lines.append("Each change waits for your confirmation; any delay pushes all later changes back")
    lines.append("by the same amount. The dashboard keeps this list up to date.")
    return "\n".join(lines)


# ---------------- serial ----------------
def serial_open(port: str, baud: int, timeout_s: float = 0.2):
    if serial is None:
        raise RuntimeError("pyserial missing. Install with: pip install pyserial")
    ser = serial.Serial(port=port, baudrate=baud, timeout=timeout_s)
    time.sleep(1.0)  # many boards reset on open
    try:
        if ser.in_waiting:
            ser.read(ser.in_waiting)
    except Exception:
        log_exception("serial_open: failed to flush input buffer")
    return ser

def serial_send_freq(ser, hz: int, protocol: str):
    hz = int(hz)
    if protocol == "freq_only":
        ser.write(f"{hz}\n".encode("utf-8"))
    else:
        ser.write(f"F {hz}\n".encode("utf-8"))
    ser.flush()

def serial_send_duty(ser, duty: float, protocol: str):
    if protocol != "freq_duty":
        return
    duty = clamp(float(duty), 0.0, 1.0)
    ser.write(f"D {duty:.4f}\n".encode("utf-8"))
    ser.flush()


# ---------------- flow backends ----------------
class FlowSensorBase:
    def open(self): ...
    def close(self): ...
    def read(self) -> Tuple[bool, Optional[float]]:
        return False, None
    def sensor_range(self) -> Optional[Tuple[float, float]]:
        return None


# One Fluigent SDK session per process, shared by the flow and the pressure sensor:
# fgt_init/fgt_close are reference-counted and every SDK call holds _FGT_LOCK.
_FGT_LOCK = threading.RLock()
_FGT_USERS = 0


def fluigent_acquire():
    global _FGT_USERS
    with _FGT_LOCK:
        if _FGT_USERS == 0:
            fgt_init()
            try:
                fgt_set_errorReportMode("None")
            except Exception:
                log_exception("Fluigent: failed to set error report mode")
        _FGT_USERS += 1


def fluigent_release():
    global _FGT_USERS
    with _FGT_LOCK:
        if _FGT_USERS <= 0:
            return
        _FGT_USERS -= 1
        if _FGT_USERS == 0:
            try:
                fgt_close()
            except Exception:
                log_exception("Fluigent: failed to close SDK")


def fluigent_list_sensors() -> List[Dict[str, Any]]:
    """Index, unit and range of every Fluigent sensor (to find the pressure channel)."""
    if not FLUIGENT_AVAILABLE or fgt_get_sensorChannelCount is None:
        raise RuntimeError("Fluigent SDK not available.")
    fluigent_acquire()
    try:
        with _FGT_LOCK:
            n = int(fgt_get_sensorChannelCount())
            out = []
            for i in range(n):
                info = {"index": i}
                try:
                    info["unit"] = str(fgt_get_sensorUnit(i))
                except Exception:
                    info["unit"] = "?"
                try:
                    r = fgt_get_sensorRange(i)
                    info["range"] = (float(r[-2]), float(r[-1]))
                except Exception:
                    info["range"] = None
                out.append(info)
        return out
    finally:
        fluigent_release()


class FluigentFlowSensor(FlowSensorBase):
    def __init__(self, channel: int = 0):
        self.channel = channel
        self._opened = False

    def open(self):
        if not FLUIGENT_AVAILABLE:
            raise RuntimeError("Fluigent SDK not available.")
        fluigent_acquire()
        self._opened = True

    def close(self):
        if self._opened:
            fluigent_release()
        self._opened = False

    def read(self):
        if not self._opened:
            return False, None
        try:
            with _FGT_LOCK:
                try:
                    err, value = fgt_get_sensorValue(self.channel, get_error=True)
                except TypeError:
                    value = fgt_get_sensorValue(self.channel)
                    err = 0
            if err != 0:
                return False, None
            return True, float(value)
        except Exception:
            log_exception_limited("fluigent_read", "FluigentFlowSensor.read: failed to read sensor value")
            return False, None

    def sensor_range(self):
        if not self._opened or fgt_get_sensorRange is None:
            return None
        try:
            with _FGT_LOCK:
                r = fgt_get_sensorRange(self.channel)
            lo, hi = float(r[-2]), float(r[-1])      # (min, max), or (error, min, max)
            return (lo, hi) if hi > lo else None
        except Exception:
            log_exception(f"Fluigent: could not read the range of sensor {self.channel}")
            return None


class PlantFlowSensor(FlowSensorBase):
    """Dummy flow on the simulated pressure line: the pump's flow at the line pressure (uL/min)."""
    def __init__(self, plant):
        self.plant = plant
        self.t0 = time.time()

    def read(self):
        t = time.time() - self.t0
        return True, self.plant.pump_flow_ul_min(self.plant.pressure()) + 1.5 * math.sin(2 * math.pi * 1.7 * t)

class DummyFlowSensor(FlowSensorBase):
    """UI testing without hardware."""
    def __init__(self):
        self.t0 = time.time()
    def open(self): pass
    def close(self): pass
    def read(self):
        t = time.time() - self.t0
        return True, 2.0 + 0.35*math.sin(2*math.pi*0.2*t) + 0.06*math.sin(2*math.pi*1.7*t)

class NetworkFlowSensor(FlowSensorBase):
    """One channel of the shared flow server (flow_server.py), read over the network.

    read() never blocks: it returns the newest sample the background client has
    received, or (False, None) if the server is unreachable, the sensor reports an
    error, or nothing new arrived within stale_s. The client reconnects on its own,
    so a server restart mid-run shows up as a gap in the data, not an exception.
    """
    def __init__(self, channel: int = 0, server: str = "localhost:8765", stale_s: float = 2.0):
        self.channel = int(channel)
        self.server = str(server).strip() or "localhost:8765"
        self.stale_s = float(stale_s)
        self._client = None
        self._had_data = False
        self._gap_since: Optional[float] = None

    def open(self):
        if not NETWORK_FLOW_AVAILABLE:
            raise RuntimeError("Network flow backend needs flow_client.py next to the dashboard and the "
                               "'websockets' package. Install with: pip install websockets")
        host, port = parse_server(self.server)
        addr = f"{host}:{port}"
        client = FlowClient(host=host, port=port, channels=[self.channel], rate="max", mode="latest")
        client.start()
        connected = client.wait_connected(3.0)
        channels = client.channels_info
        available = sorted(c["ch"] for c in channels or [] if isinstance(c.get("ch"), int))
        if channels is not None and self.channel not in available:
            client.close()
            raise RuntimeError(f"Flow server at {addr} has no channel {self.channel}; "
                               f"available channels: {available}")
        if not connected:
            error = client.last_error
            client.close(timeout=0.2)  # open() runs on the Tk thread; don't wait out a pending connect
            if channels is not None:
                raise RuntimeError(f"Flow server at {addr} did not accept the subscription to channel "
                                   f"{self.channel}: {error or 'no reply within 3 s'}")
            raise RuntimeError(f"Flow server at {addr} not reachable - is flow_server.py running? "
                               f"({error or 'no reply within 3 s'})")
        self._client = client
        self._had_data = False
        self._gap_since = None

    def close(self):
        client, self._client = self._client, None
        if client is not None:
            client.close()

    def read(self):
        client = self._client
        if client is None:
            return False, None
        connected = client.connected
        s = client.latest(self.channel) if connected else None
        fresh = (s is not None and bool(s.get("ok")) and isinstance(s.get("v"), (int, float))
                 and time.monotonic() - s["rx_mono"] <= self.stale_s)
        if not fresh:
            if self._had_data and self._gap_since is None:
                self._gap_since = time.monotonic()
                reason = ("connection to server lost" if not connected
                          else "sensor reports an error" if s is not None and not s.get("ok")
                          else "no new samples")
                log_warning(f"NetworkFlowSensor: no flow data on channel {self.channel} from "
                            f"{self.server} ({reason}); waiting for it to resume")
            return False, None
        if self._gap_since is not None:
            log_warning(f"NetworkFlowSensor: channel {self.channel} data resumed after a "
                        f"{time.monotonic() - self._gap_since:.1f} s gap")
            self._gap_since = None
        self._had_data = True
        return True, float(s["v"])

def make_flow_sensor(flow_cfg: Dict[str, Any], who: str) -> FlowSensorBase:
    """Flow backend selected by cfg['flow_sensor']; `who` prefixes error messages."""
    backend = (flow_cfg.get("backend") or ("fluigent" if FLUIGENT_AVAILABLE else "dummy")).lower()
    if backend == "fluigent":
        if not FLUIGENT_AVAILABLE:
            raise RuntimeError(f"{who}: Fluigent backend requested but SDK not available.")
        return FluigentFlowSensor(channel=int(flow_cfg.get("channel", 0)))
    if backend == "network":
        if not NETWORK_FLOW_AVAILABLE:
            raise RuntimeError(f"{who}: network flow backend requested but flow_client/websockets is not "
                               "available. Install with: pip install websockets")
        return NetworkFlowSensor(channel=int(flow_cfg.get("channel", 0)),
                                 server=str(flow_cfg.get("server") or "localhost:8765"))
    return DummyFlowSensor()


# ---------------- logging ----------------
# Registry of every open DataLogger so crash handlers / the memory watchdog can
# flush all in-flight data to disk from any thread.
_OPEN_LOGGERS: "set[DataLogger]" = set()
_OPEN_LOGGERS_LOCK = threading.Lock()

def _register_logger(lg: "DataLogger") -> None:
    with _OPEN_LOGGERS_LOCK:
        _OPEN_LOGGERS.add(lg)

def _unregister_logger(lg: "DataLogger") -> None:
    with _OPEN_LOGGERS_LOCK:
        _OPEN_LOGGERS.discard(lg)

def flush_all_loggers(sync: bool = True) -> None:
    """Best-effort flush (and optional fsync) of every open logger.

    Safe to call from crash handlers, excepthooks and the memory watchdog.
    """
    with _OPEN_LOGGERS_LOCK:
        loggers = list(_OPEN_LOGGERS)
    for lg in loggers:
        try:
            lg.flush(sync=sync)
        except Exception:
            pass  # never raise from a crash/flush path


class DataLogger:
    """Crash-safe CSV logger.

    Every row is flushed to the OS immediately, so a Python-level crash
    (including MemoryError) or a hard process kill loses no acknowledged data.
    A throttled fsync (see maybe_sync) additionally guards against power loss.

    with_voltage (runs with manual voltage steps) appends a v_set_mv column to
    events.csv and merged.csv and opens voltage_changes.csv.
    """
    VOLTAGE_CHANGE_COLS = ["index", "t_s", "requested_mv", "confirmed_mv", "asked_at", "confirmed_at",
                           "wait_s", "pump_while_waiting"]

    def __init__(self, run_dir: str, with_voltage: bool = False):
        self.run_dir = run_dir
        self.with_voltage = bool(with_voltage)
        self.events_f = None
        self.flow_f = None
        self.merged_f = None
        self.volt_f = None
        self.events_w = None
        self.flow_w = None
        self.merged_w = None
        self.volt_w = None
        self._lock = threading.Lock()
        self._last_sync = 0.0
        self._closed = False

    def _all_files(self):
        return (self.events_f, self.flow_f, self.merged_f, self.volt_f)

    def open(self):
        vcol = ["v_set_mv"] if self.with_voltage else []
        self.events_f = open(os.path.join(self.run_dir, "events.csv"), "x", newline="", encoding="utf-8")
        self.flow_f   = open(os.path.join(self.run_dir, "flow.csv"), "x", newline="", encoding="utf-8")
        self.merged_f = open(os.path.join(self.run_dir, "merged.csv"), "x", newline="", encoding="utf-8")
        # t_cmd_s / t_cmd_done_s: run clock just before and after the frequency
        # command's serial write (scheduled runs; empty in manual sessions)
        self.events_w = csv.writer(self.events_f)
        self.events_w.writerow(["t_s", "freq_set_hz", "duty_set"] + vcol + ["t_cmd_s", "t_cmd_done_s"])
        self.flow_w   = csv.writer(self.flow_f);   self.flow_w.writerow(["t_s", "flow"])
        self.merged_w = csv.writer(self.merged_f); self.merged_w.writerow(["t_s", "freq_set_hz", "duty_set", "flow"] + vcol)
        if self.with_voltage:
            self.volt_f = open(os.path.join(self.run_dir, "voltage_changes.csv"), "x", newline="", encoding="utf-8")
            self.volt_w = csv.writer(self.volt_f); self.volt_w.writerow(self.VOLTAGE_CHANGE_COLS)
        self._last_sync = time.monotonic()
        self.flush(sync=True)   # make headers durable immediately
        _register_logger(self)

    def _flush_file(self, f, sync: bool):
        if not f:
            return
        try:
            f.flush()                     # Python buffer -> OS (survives app crash)
            if sync:
                os.fsync(f.fileno())      # OS buffer -> disk (survives power loss)
        except Exception:
            log_exception_limited("datalogger_flush",
                                  f"DataLogger: failed to flush {getattr(f, 'name', '<unknown>')}")

    def flush(self, sync: bool = False):
        with self._lock:
            for f in self._all_files():
                self._flush_file(f, sync)
            if sync:
                self._last_sync = time.monotonic()

    def maybe_sync(self, interval: float = FSYNC_INTERVAL_S):
        """Throttled fsync so a power loss loses at most `interval` seconds of data."""
        if time.monotonic() - self._last_sync >= interval:
            self.flush(sync=True)

    def close(self):
        _unregister_logger(self)
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for f in self._all_files():
                try:
                    if f:
                        f.flush()
                        try:
                            os.fsync(f.fileno())
                        except Exception:
                            pass
                        f.close()
                except Exception:
                    fname = getattr(f, "name", "<unknown>")
                    log_exception(f"DataLogger.close: failed to close {fname}")
            # prevent any late writes from a straggling thread hitting a closed file
            self.events_w = self.flow_w = self.merged_w = self.volt_w = None

    def _vcol(self, v_mv: Optional[float]) -> List[str]:
        if not self.with_voltage:
            return []
        return ["" if v_mv is None else str(int(round(v_mv)))]

    def log_event(self, t_s: float, freq: int, duty: float, v_mv: Optional[float] = None,
                  t_cmd: Optional[Tuple[float, float]] = None):
        cmd = [f"{t_cmd[0]:.6f}", f"{t_cmd[1]:.6f}"] if t_cmd else ["", ""]
        with self._lock:
            if self.events_w:
                self.events_w.writerow([f"{t_s:.6f}", int(freq), f"{float(duty):.6f}"] + self._vcol(v_mv) + cmd)
                self._flush_file(self.events_f, False)

    def log_voltage_change(self, row: Dict[str, Any]):
        with self._lock:
            if self.volt_w:
                self.volt_w.writerow([row.get(k, "") for k in self.VOLTAGE_CHANGE_COLS])
                self._flush_file(self.volt_f, True)   # rare and important: make it durable now

    def log_flow(self, t_s: float, flow: float):
        with self._lock:
            if self.flow_w:
                self.flow_w.writerow([f"{t_s:.6f}", f"{float(flow):.6f}"])
                self._flush_file(self.flow_f, False)

    def log_merged(self, t_s: float, freq: int, duty: float, flow: float, v_mv: Optional[float] = None):
        with self._lock:
            if self.merged_w:
                self.merged_w.writerow([f"{t_s:.6f}", int(freq), f"{float(duty):.6f}", f"{float(flow):.6f}"]
                                       + self._vcol(v_mv))
                self._flush_file(self.merged_f, False)


# ---------------- runtime sessions ----------------
def pressure_cal_path() -> str:
    """Syringe/sensor calibration (pressure_cal.json), kept per PC next to the panel layout."""
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, "ATTO", "pressure_cal.json")


class LiveSessionBase:
    meter: Optional["PowerMeter"] = None      # PPK2 power meter when enabled
    ctl = None                                # pressure_control.PressureController when enabled
    plant = None                              # simulated line (pressure backend "dummy")

    def get_plot_data(self) -> Tuple[List[float], List[float], List[float]]:
        return [], [], []
    def stop(self): ...
    def join(self, timeout: Optional[float] = None): ...

    _hires = False

    def _release_hires(self):
        if self._hires:
            hires_timer(False)
            self._hires = False

    def overlay_info(self) -> Tuple[float, int, float]:
        """(t_s, frequency, latest flow) for the optional video overlay."""
        t0 = getattr(self, "t0", None)
        with self.lock:
            f = self.freq_set if getattr(self, "output_enabled", True) else 0
            flow = self.buf_flow[-1] if self.buf_flow else 0.0
        return (CLOCK() - t0) if t0 is not None else 0.0, f, flow

    def get_power_live(self) -> Optional[Dict[str, Any]]:
        return self.meter.get_live() if self.meter else None

    def get_power_plot_data(self) -> Tuple[List[float], List[float]]:
        return self.meter.get_plot_data() if self.meter else ([], [])

    def _make_meter(self, who: str, get_state, on_trip, on_fatal) -> Optional["PowerMeter"]:
        """PowerMeter from cfg['power'] (None when disabled). The supply voltage
        starts at supply.initial_mv; watch_ma defaults to 95 % of the bench
        supply's current limit."""
        pcfg = dict(self.cfg.get("power") or {})
        if not pcfg.get("enabled"):
            return None
        if not POWER_AVAILABLE:
            raise RuntimeError(f"{who}: power meter enabled but power_ppk2.py could not be imported.")
        supply = {**SUPPLY_DEFAULTS, **(self.cfg.get("supply") or {})}
        if pcfg.get("watch_ma") in (None, ""):
            pcfg["watch_ma"] = 0.95 * float(supply.get("current_limit_ma") or 0)
        vdd = float(supply.get("initial_mv") or 2500)
        return PowerMeter(pcfg, self.run_dir, vdd, emit=lambda k, m: self.ui_events.put((k, m)),
                          get_state=get_state, on_trip=on_trip, on_fatal=on_fatal)

    def _close_meter(self, who: str):
        if self.meter:
            try:
                self.meter.close()
            except Exception:
                log_exception(f"{who}: failed to close power meter")

    # ---- pressure control ----
    def _pump_hz(self) -> float:
        with self.lock:
            return float(self.freq_set if getattr(self, "output_enabled", True) else 0)

    def _make_pressure(self, who: str, on_fault):
        """PressureController from cfg['pressure'] (None when disabled): opens the syringe and the
        pressure sensor (a Fluigent channel sharing the SDK session with the flow sensor, or a
        flow-server channel). backend 'dummy' simulates the line, syringe and sensor. Started at
        t0 by start(); closed (pressure released) after the pump is at 0 Hz."""
        pcfg = self.cfg.get("pressure") or {}
        if not pcfg.get("enabled"):
            return None
        if not PRESSURE_AVAILABLE:
            raise RuntimeError(f"{who}: pressure control enabled but pressure_control.py could not be imported.")
        c = pc.merged_cfg(pcfg)
        sim = str(c["backend"]).lower() == "dummy"
        sensor = None
        if sim:
            m = pc.microsteps(c)
            self.plant = pc.PressurePlant(get_freq=self._pump_hz, pos_max=int(c["pos_max"]) * m,
                                          ul_per_step=0.405 / m, backlash=6 * m)
            read, rng = self.plant.read_sensor, self.plant.range
            cal = {}
        else:
            sensor = make_flow_sensor(c["sensor"], f"{who} pressure sensor")
            sensor.open()
            read = sensor.read
            rng = sensor.sensor_range() or tuple(c["sensor"].get("range_mbar") or (-312.0, 312.0))
            cal = pc.load_calibration(pressure_cal_path())
        try:
            syr = pc.make_syringe(c, self.plant)
            ctl = pc.PressureController(c, syr, read, sensor_range=rng, run_dir=self.run_dir,
                                        emit=lambda k, m: None, on_fault=on_fault, calibration=cal,
                                        pump_on=lambda: self._pump_hz() > 0)
            ctl.open()
        except Exception:
            if sensor is not None:
                sensor.close()
            raise
        ctl.sensor_obj = sensor
        if self.run_dir and cal:
            try:
                with open(os.path.join(self.run_dir, "pressure", "calibration_used.json"), "w",
                          encoding="utf-8") as f:
                    json.dump(cal, f, indent=2)
            except Exception:
                log_exception(f"{who}: failed to copy the pressure calibration into the run folder")
        self._emit("status", f"{who}: pressure control ready (syringe {syr.protocol}, position "
                             f"{syr.position()}, limit {ctl.p_max:g} mbar)")
        return ctl

    def _close_pressure(self, who: str):
        ctl = self.ctl
        if ctl is None:
            return
        try:
            ctl.close()
        except Exception:
            log_exception(f"{who}: failed to close the pressure control")
        try:
            if getattr(ctl, "sensor_obj", None) is not None:
                ctl.sensor_obj.close()
        except Exception:
            log_exception(f"{who}: failed to close the pressure sensor")
        self.ctl = None

    def _make_flow(self, who: str) -> FlowSensorBase:
        fcfg = self.cfg.get("flow_sensor", {})
        if self.plant is not None and str(fcfg.get("backend") or "").lower() == "dummy":
            return PlantFlowSensor(self.plant)
        return make_flow_sensor(fcfg, who)

    def get_pressure_status(self) -> Optional[Dict[str, Any]]:
        ctl = self.ctl
        return ctl.status() if ctl is not None else None

    def get_pressure_plot_data(self):
        """(t, p, target) arrays for the live plot; empty lists without pressure control."""
        ctl = self.ctl
        rows = list(ctl.plot_buf) if ctl is not None else []
        if not rows:
            return [], [], []
        a = np.asarray(rows, dtype=float)
        return a[:, 0], a[:, 1], a[:, 2]


class ManualSession(LiveSessionBase):
    """Manual real-time control session."""
    def __init__(self, cfg: Dict[str, Any], run_dir: Optional[str], ui_events: "queue.Queue[Tuple[str, Any]]", video_recorder: Optional["VideoRecorder"] = None):
        self.cfg = cfg
        self.run_dir = run_dir
        self.ui_events = ui_events
        self.stop_flag = threading.Event()
        self.video_recorder = video_recorder

        self.protocol = (cfg.get("protocol") or cfg.get("serial", {}).get("protocol") or "freq_only")
        self.ser = None
        self.sensor: FlowSensorBase = DummyFlowSensor()

        self.freq_set = 0
        self.duty_set = 0.0
        self.output_enabled = False

        self.lock = threading.Lock()
        self.buf_t = deque(maxlen=5000)
        self.buf_freq = deque(maxlen=5000)
        self.buf_flow = deque(maxlen=5000)

        self.logger = DataLogger(run_dir) if run_dir else None
        self.t0 = None
        self.t_sample = None
        self.t_video = None
        self.meter = None
        self._ser_lock = threading.Lock()   # the power meter's trip handler also writes to the Arduino

    def _emit(self, kind: str, msg: Any):
        self.ui_events.put((kind, msg))

    def _meter_state(self):
        with self.lock:
            return (self.freq_set if self.output_enabled else 0), None

    def _on_power_trip(self, info: Dict[str, Any]):
        self.set_output_enabled(False)
        self._emit("manual_output_off", None)
        cut = " The PPK2 output was switched off too." if info.get("power_cut") is True else ""
        self._emit("error", f"Current {info['i_ma']:.1f} mA stayed at or above {info['watch_ma']:.1f} mA for "
                            f"{info['hold_ms']:.0f} ms: the bench supply is probably limiting. "
                            f"Output switched OFF (0 Hz).{cut} Details: power/power_events.csv")

    def _on_power_fatal(self, msg: str):
        self.set_output_enabled(False)
        self._emit("manual_output_off", None)
        self._emit("error", msg + " Output switched OFF (0 Hz).")

    def _on_pressure_fault(self, msg: str):
        self.set_output_enabled(False)
        self._emit("manual_output_off", None)
        self._emit("error", f"Pressure control: {msg}. Pump output switched OFF (0 Hz).")

    def _setup_hardware(self):
        port = self.cfg.get("port", "COM22")
        baud = int(self.cfg.get("baud", 115200))
        timeout_s = float(self.cfg.get("read_timeout_s", 0.2))

        # Power meter first: if it fails, nothing else has been opened yet
        self.meter = self._make_meter("Manual", self._meter_state, self._on_power_trip, self._on_power_fatal)
        if self.meter:
            self._emit("status", "Manual: connecting power meter ...")
            self.meter.open()
        try:
            if bool(self.cfg.get("dry_run", False)):
                self.ser = None
                self._emit("status", "Manual: dry_run enabled (no serial).")
            else:
                self._emit("status", f"Manual: opening serial {port} @ {baud} ...")
                self.ser = serial_open(port, baud, timeout_s)

            self.ctl = self._make_pressure("Manual", self._on_pressure_fault)
            self.sensor = self._make_flow("Manual")
            self.sensor.open()
        except Exception:
            self._close_pressure("Manual")
            if self.ser:
                try:
                    self.ser.close()
                except Exception:
                    log_exception("Manual: failed to close serial after a setup error")
                self.ser = None
            self._close_meter("Manual")
            raise

        if self.logger:
            self.logger.open()

        self._set_freq_internal(0)
        self._set_duty_internal(0.0)

    def _teardown_hardware(self):
        try:
            if bool(self.cfg.get("send_zero_on_exit", True)):
                self._set_freq_internal(0)
                self._set_duty_internal(0.0)
        except Exception:
            log_exception("ManualSession teardown: failed to send zero on exit")
        self._close_pressure("ManualSession teardown")
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            log_exception("ManualSession teardown: failed to close serial port")
        try:
            self.sensor.close()
        except Exception:
            log_exception("ManualSession teardown: failed to close flow sensor")
        try:
            if self.logger:
                self.logger.close()
        except Exception:
            log_exception("ManualSession teardown: failed to close logger")
        self._close_meter("ManualSession teardown")

    def _set_freq_internal(self, hz: int):
        hz = int(hz)
        with self.lock:
            self.freq_set = hz
        if self.ser:
            with self._ser_lock:
                serial_send_freq(self.ser, hz, self.protocol)

    def _set_duty_internal(self, duty: float):
        duty = clamp(float(duty), 0.0, 1.0)
        with self.lock:
            self.duty_set = duty
        if self.ser:
            serial_send_duty(self.ser, duty, self.protocol)

    def set_output_enabled(self, enabled: bool):
        enabled = bool(enabled)
        with self.lock:
            self.output_enabled = enabled
            f = self.freq_set
            d = self.duty_set
        self._set_freq_internal(f if enabled else 0)
        if self.logger and self.t0 is not None:
            self.logger.log_event(CLOCK() - self.t0, f if enabled else 0, d)

    def set_frequency(self, hz: int):
        hz = max(0, int(hz))
        with self.lock:
            self.freq_set = hz
            enabled = self.output_enabled
            d = self.duty_set
        if enabled:
            self._set_freq_internal(hz)
        if self.logger and self.t0 is not None:
            self.logger.log_event(CLOCK() - self.t0, hz if enabled else 0, d)

    def set_duty(self, duty: float):
        duty = clamp(float(duty), 0.0, 1.0)
        with self.lock:
            self.duty_set = duty
            enabled = self.output_enabled
            f = self.freq_set
        if enabled:
            self._set_duty_internal(duty)
        if self.logger and self.t0 is not None:
            self.logger.log_event(CLOCK() - self.t0, f if enabled else 0, duty)

    def _sampling_loop(self):
        sr = float(self.cfg.get("flow_sensor", {}).get("sample_rate_hz", 25.0))
        sr = 10.0 if sr <= 0 else sr
        dt = 1.0 / sr
        try:
            while not self.stop_flag.is_set():
                ok, val = self.sensor.read()
                t_s = CLOCK() - self.t0
                if ok and val is not None:
                    with self.lock:
                        f = self.freq_set if self.output_enabled else 0
                        d = self.duty_set
                        self.buf_t.append(t_s)
                        self.buf_freq.append(f)
                        self.buf_flow.append(float(val))
                    if self.logger:
                        self.logger.log_flow(t_s, float(val))
                        self.logger.log_merged(t_s, f, d, float(val))
                        self.logger.maybe_sync()
                time.sleep(dt)
        except Exception:
            # Includes MemoryError: don't let the thread die silently and drop data.
            log_exception("ManualSession sampling loop crashed; flushing data to disk")
            try:
                if self.logger:
                    self.logger.flush(sync=True)
            except Exception:
                log_exception("ManualSession: failed to flush after sampling crash")
            self.stop_flag.set()
            self._emit("error", "Sampling stopped due to an error (possibly low memory). "
                                "Data collected so far has been saved to disk.")

    def start(self):
        self._setup_hardware()
        hires_timer(True)
        self._hires = True
        self.t0 = CLOCK()
        self.stop_flag.clear()
        if self.meter:
            self.meter.start(self.t0)
        if self.ctl:
            self.ctl.start(self.t0)
        self._emit("status", "Manual: connected. Sampling started.")
        self.t_sample = threading.Thread(target=self._sampling_loop, daemon=True)
        self.t_sample.start()

    def stop(self):
        self.stop_flag.set()

    def join(self, timeout: Optional[float] = None):
        try:
            if self.t_sample:
                self.t_sample.join(timeout=timeout)
        except Exception:
            log_exception("ManualSession.join: failed while joining sampling thread")
        self._teardown_hardware()
        self._release_hires()
        self._emit("status", "Manual: disconnected.")

    def get_plot_data(self):
        with self.lock:
            return list(self.buf_t), list(self.buf_freq), list(self.buf_flow)


class PumpTestRunner(LiveSessionBase):
    """Scheduled runner based on cfg['tests'] blocks (constant + sweep + group)."""
    def __init__(self, cfg: Dict[str, Any], run_dir: str, ui_events: "queue.Queue[Tuple[str, Any]]", video_recorder: Optional["VideoRecorder"] = None):
        self.cfg = cfg
        self.run_dir = run_dir
        self.ui_events = ui_events
        self.stop_flag = threading.Event()
        self.video_recorder = video_recorder

        self.protocol = (cfg.get("protocol") or cfg.get("serial", {}).get("protocol") or "freq_only")
        self.ser = None
        self.sensor: FlowSensorBase = DummyFlowSensor()

        self.freq_set = 0
        self.duty_set = 0.0
        self._step_deadline: Optional[float] = None

        self.lock = threading.Lock()
        self.buf_t = deque(maxlen=5000)
        self.buf_freq = deque(maxlen=5000)
        self.buf_flow = deque(maxlen=5000)

        # Manual bench-supply voltage steps and run progress. nominal_done counts
        # the same nominal seconds as plan_schedule, so ETAs line up with the plan.
        tests = cfg.get("tests", [])
        self.tracks_voltage = contains_voltage_block(tests)
        self.supply_cfg = {**SUPPLY_DEFAULTS, **(cfg.get("supply") or {})}
        init_mv = self.supply_cfg.get("initial_mv")
        self.v_set_mv: Optional[int] = int(init_mv) if init_mv is not None else None
        self.plan_total_s, self.plan_changes, _ = plan_schedule(
            tests, float(cfg.get("break_seconds_between_tests", 0.0)))
        self.nominal_done = 0.0
        self._nominal_step: Optional[Tuple[float, float]] = None   # (monotonic start, nominal s)
        self.changes_done = 0
        self.voltage_request: Optional[Dict[str, Any]] = None
        self._voltage_confirmed = threading.Event()
        self._voltage_value_mv: Optional[int] = None
        self.t_end: Optional[float] = None

        self.logger = DataLogger(run_dir, with_voltage=self.tracks_voltage)
        self.t0 = None
        self.t_sample = None
        self.t_run = None
        self.t_video = None
        self.meter = None
        self._ser_lock = threading.Lock()   # the power meter's trip handler also writes to the Arduino
        self._force_zero = False            # after a power trip every frequency command becomes 0 Hz
        self._last_cmd: Optional[Tuple[float, float]] = None   # (t_cmd_s, t_cmd_done_s) of the last frequency command
        # Operator pauses other than voltage steps (clearing the syringe by hand), PQ points
        self.operator_request: Optional[Dict[str, Any]] = None
        self._operator_done = threading.Event()
        self._op_index = 0
        self._acc: Optional[List[float]] = None   # flow samples of the PQ point being measured
        self.pq_count = 0
        self.syringe_clears = 0
        self._pq_file = None
        self._pq_writer = None

    def _emit(self, kind: str, msg: Any):
        self.ui_events.put((kind, msg))

    def _meter_state(self):
        with self.lock:
            return self.freq_set, self.v_set_mv

    def _power_stop(self, msg: str):
        """Power trip / meter lost: pump to 0 Hz now, end the run, tell the operator."""
        self._force_zero = True
        self.stop_flag.set()
        try:
            self._set_freq(0)
        except Exception:
            log_exception("PumpTestRunner: failed to send 0 Hz after power stop")
        self._emit("error", msg)

    def _on_power_trip(self, info: Dict[str, Any]):
        cut = " The PPK2 output was switched off too." if info.get("power_cut") is True else ""
        self._power_stop(f"Current {info['i_ma']:.1f} mA stayed at or above {info['watch_ma']:.1f} mA for "
                         f"{info['hold_ms']:.0f} ms: the bench supply is probably limiting. "
                         f"Run stopped, pump at 0 Hz.{cut} Details: power/power_events.csv")

    def _on_power_fatal(self, msg: str):
        self._power_stop(msg + " Run stopped, pump at 0 Hz.")

    def _on_pressure_fault(self, msg: str):
        self._power_stop(f"Pressure control: {msg}. Run stopped, pump at 0 Hz.")

    def _nominal_begin(self, t_start: float, dur_s: float):
        with self.lock:
            self._nominal_step = (t_start, dur_s)

    def _nominal_end(self, dur_s: float):
        with self.lock:
            self._nominal_step = None
            self.nominal_done += dur_s

    def get_progress(self) -> Dict[str, Any]:
        """Snapshot for the UI timer (thread-safe)."""
        now = CLOCK()
        with self.lock:
            nominal = self.nominal_done
            if self._nominal_step is not None:
                t_start, dur = self._nominal_step
                nominal += min(max(0.0, now - t_start), dur)
            return {
                "elapsed_s": ((self.t_end or now) - self.t0) if self.t0 is not None else 0.0,
                "nominal_done_s": nominal,
                "total_s": self.plan_total_s,
                "changes_done": self.changes_done,
                "voltage_request": dict(self.voltage_request) if self.voltage_request else None,
                "operator_request": dict(self.operator_request) if self.operator_request else None,
                "v_set_mv": self.v_set_mv,
                "finished": self.t_end is not None,
            }

    def confirm_voltage(self, mv: float) -> bool:
        """Called from the UI when the operator has set the supply. False if nothing was asked."""
        with self.lock:
            if self.voltage_request is None:
                return False
            self._voltage_value_mv = int(round(mv))
        self._voltage_confirmed.set()
        return True

    def confirm_operator(self) -> bool:
        """UI: the operator finished what the prompt asked. False if nothing was asked."""
        with self.lock:
            if self.operator_request is None:
                return False
        self._operator_done.set()
        return True

    def _ask_operator(self, kind: str, title: str, text: str) -> bool:
        """Pause until the operator presses Done (True) or the run is stopped (False)."""
        self._operator_done.clear()
        with self.lock:
            self._op_index += 1
            self.operator_request = {"kind": kind, "index": self._op_index, "title": title, "text": text,
                                     "asked_mono": CLOCK(),
                                     "asked_at": datetime.now().isoformat(timespec="seconds")}
            req = dict(self.operator_request)
        self._emit("operator_prompt", req)
        self._emit("status", f"Waiting for you: {title}")
        while not self.stop_flag.is_set() and not self._operator_done.wait(0.1):
            pass
        with self.lock:
            self.operator_request = None
        return not self.stop_flag.is_set()

    # ---- pressure blocks ----
    def _need_ctl(self):
        if self.ctl is None:
            raise RuntimeError("This block needs pressure control: tick 'Control pressure' in the "
                               "Pressure panel (or set pressure.enabled in the config).")
        return self.ctl

    def _clear_syringe(self, why: str, log_event) -> bool:
        """Pump to 0 Hz, let the pressure down, ask the operator to empty the syringe by hand,
        then position = 0 (the empty end). False if the run was stopped meanwhile."""
        ctl = self.ctl
        self._set_freq(0)
        log_event()
        ctl.release()
        t_end = CLOCK() + 20.0
        while ctl.mode == "release" and CLOCK() < t_end and not self.stop_flag.is_set():
            time.sleep(0.1)
        ctl.idle()
        try:
            ctl.syr.set_hold(False)
        except Exception:
            log_exception_limited("syringe_hold", "Clear syringe: could not release the motor")
        p = ctl.p
        ctl.note("clear_request", why)
        ok = self._ask_operator(
            "syringe_clear", "Clear the syringe by hand",
            f"{why}\nThe pump is at 0 Hz and the syringe motor is idle (line at {p:.0f} mbar). Empty the "
            f"syringe by hand back to its empty end (plunger fully in), remove bubbles, reconnect and close "
            f"the line, then press 'Done - continue'. The position is then set to 0.")
        if not ok:
            return False
        ctl.set_position(0)
        self.syringe_clears += 1
        ctl.note("syringe_cleared", str(self.syringe_clears))
        self._emit("status", f"Syringe cleared (#{self.syringe_clears}); position set to 0. Continuing.")
        return True

    def _exec_pressure(self, test, log_event):
        """{'type': 'pressure', 'target_mbar': 50 | 'release' | 'idle', 'wait_settled': true,
        'timeout_s': 120}: hold a pressure (and wait until settled), release it or stop holding."""
        ctl = self._need_ctl()
        tgt = test.get("target_mbar", "release")
        if isinstance(tgt, str) or tgt is None:
            if str(tgt).lower() == "idle":
                ctl.idle()
                return
            ctl.release()
            t_end = CLOCK() + safe_float(test.get("timeout_s", 30), 30.0)
            while ctl.mode == "release" and CLOCK() < t_end and not self.stop_flag.is_set():
                time.sleep(0.1)
            return
        target = float(tgt)
        ctl.set_target(target)
        if not test.get("wait_settled", True):
            return
        timeout = safe_float(test.get("timeout_s", 120), 120.0)
        t_start = CLOCK()
        self._nominal_begin(t_start, PRESSURE_SETTLE_GUESS_S)
        reserve = int(ctl.cfg["clear_reserve_steps"])
        while not self.stop_flag.is_set() and CLOCK() - t_start < timeout:
            if ctl.mode == "fault" or ctl.settled():
                break
            if ctl.remaining_steps(-1) < reserve:
                hz = self.freq_set
                if not self._clear_syringe("The syringe is nearly full.", log_event):
                    return
                self._set_freq(hz)
                log_event()
                ctl.set_target(target)
            time.sleep(0.05)
        if not self.stop_flag.is_set():
            self._nominal_end(PRESSURE_SETTLE_GUESS_S)
            if not ctl.settled() and ctl.mode != "fault":
                self._emit("status", f"Pressure {target:g} mbar not settled within {timeout:g} s; continuing.")

    def _open_pq_log(self):
        if self._pq_writer is None:
            path = os.path.join(self.run_dir, "pq_points.csv")
            self._pq_file = open(path, "a", newline="", encoding="utf-8")
            self._pq_writer = csv.DictWriter(self._pq_file, fieldnames=PQ_COLS, lineterminator="\n")
            if self._pq_file.tell() == 0:
                self._pq_writer.writeheader()

    def _write_pq_row(self, row: Dict[str, Any]):
        self._open_pq_log()
        self._pq_writer.writerow({k: row.get(k, "") for k in PQ_COLS})
        self._pq_file.flush()
        try:
            os.fsync(self._pq_file.fileno())
        except Exception:
            pass

    def _window_current_ma(self, t_from: float, t_to: float) -> Optional[float]:
        if not self.meter:
            return None
        ts, cur = self.meter.get_plot_data()
        vals = [i for t, i in zip(ts, cur) if t_from <= t <= t_to]
        return sum(vals) / len(vals) if vals else None

    def _exec_pq(self, test, log_event, label: str):
        """PQ curve: for each frequency, hold each outlet pressure, wait until settled, then
        average flow and pressure over measure_s -> one row in pq_points.csv per point.
        {'type': 'pq', 'frequency_hz': 600 | [400, 600], 'pressures_mbar': [0, 25, 50, ...],
         'settle_timeout_s': 120, 'measure_s': 30, 'on_timeout': 'measure' | 'skip',
         'release_after': true}
        The pump flow fills the syringe; when it would run out, the run pauses for the operator
        to clear the syringe by hand (pump at 0 Hz) and the point is repeated."""
        ctl = self._need_ctl()
        pressures = [float(p) for p in test.get("pressures_mbar") or []]
        settle_to = safe_float(test.get("settle_timeout_s", 120), 120.0)
        measure_s = max(1.0, safe_float(test.get("measure_s", 30), 30.0))
        on_timeout = str(test.get("on_timeout", "measure")).lower()
        nominal = PRESSURE_SETTLE_GUESS_S + measure_s
        for hz in pq_frequencies(test):
            if self.stop_flag.is_set():
                break
            self._set_freq(hz)
            log_event()
            for k, p_t in enumerate(pressures):
                if self.stop_flag.is_set():
                    break
                self._emit("status", f"{label}PQ {hz} Hz: point {k + 1}/{len(pressures)} at {p_t:g} mbar")
                t_start = CLOCK()
                self._nominal_begin(t_start, nominal)
                row = self._pq_point(hz, p_t, settle_to, measure_s, on_timeout, log_event, label)
                if self.stop_flag.is_set():
                    break
                self._nominal_end(nominal)
                if row is not None and row.get("note") == "above_shutoff":
                    rest = len(pressures) - k - 1
                    self._emit("status", f"{label}PQ {hz} Hz: the pump cannot hold {p_t:g} mbar (syringe had to "
                                         f"push from its empty end); skipping the {rest} higher point(s).")
                    with self.lock:
                        self.nominal_done += rest * nominal
                    break
        if not self.stop_flag.is_set():
            self._set_freq(0)
            log_event()
            if test.get("release_after", True):
                ctl.release()
                t_end = CLOCK() + 30.0
                while ctl.mode == "release" and CLOCK() < t_end and not self.stop_flag.is_set():
                    time.sleep(0.1)

    def _pq_point(self, hz: int, p_t: float, settle_to: float, measure_s: float, on_timeout: str,
                  log_event, label: str) -> Optional[Dict[str, Any]]:
        ctl = self.ctl
        reserve = int(ctl.cfg["clear_reserve_steps"])
        for attempt in range(1, 4):
            # Enough syringe left for settling + measuring? Else clear it before starting.
            need = ctl.steps_rate() * (measure_s + min(settle_to, PRESSURE_SETTLE_GUESS_S)) * 1.3
            if ctl.remaining_steps(-1) < reserve + need:
                if not self._clear_syringe(f"{ctl.remaining_steps(-1)} steps left in the syringe; the next PQ "
                                           f"point needs about {need:.0f}.", log_event):
                    return None
                self._set_freq(hz)
                log_event()
            ctl.set_target(p_t)
            t0 = CLOCK()
            full = blocked_since = None
            p_ref = 0.0
            settled = False
            while not self.stop_flag.is_set() and CLOCK() - t0 < settle_to:
                if ctl.mode == "fault":
                    return None
                if ctl.settled():
                    settled = True
                    break
                if ctl.remaining_steps(-1) < reserve:
                    full = "settling"
                    break
                # Below the target with the push blocked at the empty end: wait while the pump
                # still raises the pressure; a plateau (< 2 mbar in 5 s) means above shut-off
                if ctl.blocked in ("empty", "pump_below") and ctl.p_1s < p_t - 2 * float(ctl.cfg["tolerance_mbar"]):
                    if blocked_since is None or ctl.p_1s - p_ref >= 2.0:
                        blocked_since, p_ref = CLOCK(), ctl.p_1s
                    elif CLOCK() - blocked_since > 5.0:
                        break
                else:
                    blocked_since = None
                time.sleep(0.05)
            if self.stop_flag.is_set():
                return None
            t_settle = CLOCK() - t0
            base = {"point": self.pq_count + 1, "block": label.strip().rstrip(":"), "frequency_hz": hz,
                    "target_mbar": p_t, "settled": int(settled), "t_settle_s": f"{t_settle:.1f}",
                    "syringe_clears": self.syringe_clears}
            if blocked_since is not None:
                with self.lock:
                    last = list(self.buf_flow)[-50:]
                fm = sum(last) / len(last) if last else float("nan")
                row = {**base, "p_mean_mbar": f"{ctl.p_1s:.2f}", "flow_mean": f"{fm:.4f}" if last else "",
                       "flow_n": len(last), "pos_end": ctl.syr.position(),
                       "note": "above_shutoff"}
                self.pq_count += 1
                self._write_pq_row(row)
                self._emit("pq_point", row)
                return row
            if full:
                if not self._clear_syringe(f"The syringe filled up while {full}; the point is repeated.", log_event):
                    return None
                self._set_freq(hz)
                log_event()
                continue
            if not settled and on_timeout == "skip":
                row = {**base, "note": f"not settled in {settle_to:g} s, skipped"}
                self.pq_count += 1
                self._write_pq_row(row)
                self._emit("pq_point", row)
                return row
            # Measure
            tm0 = CLOCK() - self.t0
            pos0 = ctl.syr.position()
            with self.lock:
                self._acc = []
            t_end = CLOCK() + measure_s
            while not self.stop_flag.is_set() and CLOCK() < t_end:
                if ctl.mode == "fault":
                    break
                if ctl.remaining_steps(-1) < reserve:
                    full = "measuring"
                    break
                time.sleep(0.05)
            with self.lock:
                flows, self._acc = self._acc or [], None
            tm1 = CLOCK() - self.t0
            if self.stop_flag.is_set() or ctl.mode == "fault":
                return None
            if full:
                if not self._clear_syringe(f"The syringe filled up while {full}; the point is repeated.", log_event):
                    return None
                self._set_freq(hz)
                log_event()
                continue
            st = ctl.window_stats(tm0, tm1)
            n = len(flows)
            fm = sum(flows) / n if n else float("nan")
            fsd = math.sqrt(sum((x - fm) ** 2 for x in flows) / (n - 1)) if n > 1 else float("nan")
            cur = self._window_current_ma(tm0, tm1)
            fmt = lambda v, d=3: "" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{d}f}"
            row = {**base, "p_mean_mbar": fmt(st.get("p_mean")), "p_sd_mbar": fmt(st.get("p_sd")),
                   "flow_mean": fmt(fm, 4), "flow_sd": fmt(fsd, 4), "flow_n": n,
                   "syringe_steps_per_s": fmt(st.get("steps_per_s")),
                   "syringe_flow_ul_min": fmt(st.get("syringe_flow_ul_min"), 2),
                   "current_mean_ma": fmt(cur), "t_start_s": f"{tm0:.3f}", "t_end_s": f"{tm1:.3f}",
                   "pos_start": pos0, "pos_end": ctl.syr.position(),
                   "note": "" if settled else f"not settled in {settle_to:g} s"}
            self.pq_count += 1
            self._write_pq_row(row)
            self._emit("pq_point", row)
            return row
        self._emit("status", f"{label}PQ point {p_t:g} mbar at {hz} Hz failed 3 times; moving on.")
        return None

    def _exec_voltage(self, test, log_event):
        """Manual voltage step: (pump to 0 Hz) -> wait for the allowed window ->
        ask the operator -> wait for confirmation -> log -> settle."""
        mv = int(round(safe_float(test.get("voltage_mv", 0), 0.0)))
        hold = str(test.get("while_waiting", "zero")).lower() == "hold"
        settle_s = max(0.0, safe_float(test.get("settle_s", 0), 0.0))
        index = self.changes_done + 1
        if not hold:
            self._set_freq(0)
            log_event()

        now = datetime.now()
        opens = next_in_window(now, voltage_window(test))
        if opens > now:
            self._emit("status", f"Voltage change #{index} ({mv / 1000:.3f} V) waits for its time window; "
                                 f"asking at {fmt_clock(opens)}.")
            self._wait_until(CLOCK() + (opens - now).total_seconds())
            if self.stop_flag.is_set():
                return

        self._voltage_confirmed.clear()
        asked_wall = datetime.now()
        asked_mono = CLOCK()
        with self.lock:
            self._voltage_value_mv = None
            self.voltage_request = {
                "index": index, "mv": mv, "hold": hold, "freq_hz": self.freq_set,
                "asked_at": asked_wall.isoformat(timespec="seconds"), "asked_mono": asked_mono,
                "limit_ma": self.supply_cfg.get("current_limit_ma"),
                "max_mv": self.supply_cfg.get("max_mv"), "note": str(test.get("note", "")),
            }
            req = dict(self.voltage_request)
        self._emit("voltage_prompt", req)
        self._emit("status", f"Waiting for you: set the bench supply to {mv / 1000:.3f} V "
                             f"(voltage change #{index}).")
        while not self.stop_flag.is_set() and not self._voltage_confirmed.wait(0.1):
            pass
        with self.lock:
            self.voltage_request = None
            confirmed = self._voltage_value_mv
        if self.stop_flag.is_set() or confirmed is None:
            return

        wait_s = CLOCK() - asked_mono
        with self.lock:
            self.v_set_mv = confirmed
            self.changes_done = index
        if self.meter:
            self.meter.set_voltage(confirmed)
        self.logger.log_voltage_change({
            "index": index, "t_s": f"{CLOCK() - self.t0:.3f}", "requested_mv": mv,
            "confirmed_mv": confirmed, "asked_at": req["asked_at"],
            "confirmed_at": datetime.now().isoformat(timespec="seconds"),
            "wait_s": f"{wait_s:.1f}", "pump_while_waiting": "hold" if hold else "zero",
        })
        log_event()
        self._emit("status", f"Voltage change #{index} confirmed: {confirmed / 1000:.3f} V after "
                             f"{fmt_duration(wait_s)}" + (f"; settling {settle_s:g} s" if settle_s > 0 else ""))
        if settle_s > 0:
            t_start = CLOCK()
            self._nominal_begin(t_start, settle_s)
            self._wait_until(t_start + settle_s)
            if not self.stop_flag.is_set():
                self._nominal_end(settle_s)

    def _setup_hardware(self):
        port = self.cfg.get("port", "COM22")
        baud = int(self.cfg.get("baud", 115200))
        timeout_s = float(self.cfg.get("read_timeout_s", 0.2))

        # Power meter first: if it fails, nothing else has been opened yet
        self.meter = self._make_meter("Runner", self._meter_state, self._on_power_trip, self._on_power_fatal)
        if self.meter:
            self._emit("status", "Runner: connecting power meter ...")
            self.meter.open()
        try:
            if bool(self.cfg.get("dry_run", False)):
                self.ser = None
                self._emit("status", "Runner: dry_run enabled (no serial).")
            else:
                self._emit("status", f"Runner: opening serial {port} @ {baud} ...")
                self.ser = serial_open(port, baud, timeout_s)

            self.ctl = self._make_pressure("Runner", self._on_pressure_fault)
            self.sensor = self._make_flow("Runner")
            self.sensor.open()
        except Exception:
            self._close_pressure("Runner")
            if self.ser:
                try:
                    self.ser.close()
                except Exception:
                    log_exception("Runner: failed to close serial after a setup error")
                self.ser = None
            self._close_meter("Runner")
            raise

        self.logger.open()
        self._set_freq(0)
        self._set_duty(0.0)

    def _teardown_hardware(self):
        try:
            if bool(self.cfg.get("send_zero_on_exit", True)):
                self._set_freq(0)
                self._set_duty(0.0)
        except Exception:
            log_exception("PumpTestRunner teardown: failed to send zero on exit")
        self._close_pressure("PumpTestRunner teardown")
        if self._pq_file is not None:
            try:
                self._pq_file.close()
            except Exception:
                log_exception("PumpTestRunner teardown: failed to close pq_points.csv")
        self._close_meter("PumpTestRunner teardown")
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            log_exception("PumpTestRunner teardown: failed to close serial port")
        try:
            self.sensor.close()
        except Exception:
            log_exception("PumpTestRunner teardown: failed to close flow sensor")
        try:
            self.logger.close()
        except Exception:
            log_exception("PumpTestRunner teardown: failed to close logger")

    def _set_freq(self, hz: int):
        hz = 0 if self._force_zero else int(hz)
        with self.lock:
            self.freq_set = hz
        t_cmd = CLOCK()
        if self.ser:
            with self._ser_lock:
                serial_send_freq(self.ser, hz, self.protocol)
        if self.t0 is not None:
            self._last_cmd = (t_cmd - self.t0, CLOCK() - self.t0)

    def _set_duty(self, duty: float):
        duty = clamp(float(duty), 0.0, 1.0)
        with self.lock:
            self.duty_set = duty
        if self.ser:
            serial_send_duty(self.ser, duty, self.protocol)

    def _sampling_loop(self):
        sr = float(self.cfg.get("flow_sensor", {}).get("sample_rate_hz", 25.0))
        sr = 10.0 if sr <= 0 else sr
        dt = 1.0 / sr
        try:
            while not self.stop_flag.is_set():
                ok, val = self.sensor.read()
                t_s = CLOCK() - self.t0
                if ok and val is not None:
                    with self.lock:
                        f = self.freq_set
                        d = self.duty_set
                        v = self.v_set_mv
                        self.buf_t.append(t_s)
                        self.buf_freq.append(f)
                        self.buf_flow.append(float(val))
                        if self._acc is not None:
                            self._acc.append(float(val))
                    self.logger.log_flow(t_s, float(val))
                    self.logger.log_merged(t_s, f, d, float(val), v)
                    self.logger.maybe_sync()
                time.sleep(dt)
        except Exception:
            # Includes MemoryError: don't let the thread die silently and drop data.
            log_exception("PumpTestRunner sampling loop crashed; flushing data to disk")
            try:
                self.logger.flush(sync=True)
            except Exception:
                log_exception("PumpTestRunner: failed to flush after sampling crash")
            self.stop_flag.set()
            self._emit("error", "Sampling stopped due to an error (possibly low memory). "
                                "Data collected so far has been saved to disk.")

    def _exec_block(self, test, log_event, status_prefix: str = ""):
        """Execute one test block. Supports 'constant', 'sweep', 'voltage' (manual
        supply change, see _exec_voltage) and 'group' (a list of child blocks
        repeated 'repeat' times; repeat <= 0 loops forever until Stop)."""
        ttype = (test.get("type") or "constant").lower()
        settle_ms = max(0, int(test.get("settle_ms", 0)))

        if ttype == "voltage":
            self._exec_voltage(test, log_event)
            return
        if ttype == "pressure":
            self._exec_pressure(test, log_event)
            return
        if ttype == "pq":
            self._exec_pq(test, log_event, status_prefix)
            return

        if ttype == "group":
            sub_blocks = test.get("blocks", [])
            if not sub_blocks:
                return
            repeat = int(test.get("repeat", 1))
            infinite = repeat <= 0
            cycle = 0
            last_status_t = 0.0
            while not self.stop_flag.is_set() and (infinite or cycle < repeat):
                cycle += 1
                # Throttle status updates: short cycles (ms-scale) would flood the UI
                now = CLOCK()
                if now - last_status_t >= 0.5:
                    total = "inf" if infinite else str(repeat)
                    self._emit("status", f"{status_prefix}group cycle {cycle}/{total}")
                    last_status_t = now
                for b in sub_blocks:
                    if self.stop_flag.is_set():
                        break
                    self._exec_block(b, log_event, status_prefix)
            return

        settle_s = settle_ms / 1000.0
        repeat = max(1, int(test.get("repeat", 1)))
        for _ in range(repeat):
            if self.stop_flag.is_set():
                break

            if ttype == "constant":
                hz = int(test["frequency_hz"])
                dur = float(test["duration_s"])
                t_start = self._step_start()
                self._nominal_begin(t_start, settle_s + dur)
                self._set_freq(hz)
                self._wait_until(t_start + settle_s)
                log_event()
                self._step_deadline = t_start + settle_s + dur
                self._wait_until(self._step_deadline)
                if not self.stop_flag.is_set():
                    self._nominal_end(settle_s + dur)

            elif ttype == "sweep":
                a = int(test["start_hz"]); b = int(test["end_hz"]); n = int(test["steps"])
                if "step_duration_s" in test:
                    step_dur = float(test["step_duration_s"])
                elif "total_duration_s" in test:
                    step_dur = float(test["total_duration_s"]) / max(1, n)
                else:
                    raise ValueError("Sweep needs step_duration_s or total_duration_s")

                for hz in linspace_int(a, b, n):
                    if self.stop_flag.is_set():
                        break
                    t_start = self._step_start()
                    self._nominal_begin(t_start, settle_s + step_dur)
                    self._set_freq(hz)
                    self._wait_until(t_start + settle_s)
                    log_event()
                    self._step_deadline = t_start + settle_s + step_dur
                    self._wait_until(self._step_deadline)
                    if not self.stop_flag.is_set():
                        self._nominal_end(settle_s + step_dur)
            else:
                raise ValueError(f"Unknown block type: {ttype}")

    def _step_start(self) -> float:
        """Start time for the next step. Back-to-back steps start at the previous
        step's deadline instead of 'now', so sleep overshoot (~0.5 ms per step on
        Windows) doesn't accumulate over a long ms-scale group loop."""
        now = CLOCK()
        prev = self._step_deadline
        if prev is not None and 0.0 <= now - prev < 0.05:
            return prev
        return now

    def _wait_until(self, t_end: float):
        """Wait until the CLOCK deadline t_end (within ~0.1 ms), or until Stop."""
        while not self.stop_flag.is_set():
            remaining = t_end - CLOCK()
            if remaining <= 0:
                return
            if remaining > 0.003:
                # Cap each sleep so Stop stays responsive; wake ~2 ms early because
                # a sleep can overshoot by ~1.5 ms even with the 1 ms timer.
                time.sleep(min(0.02, remaining - 0.002))
            else:
                time.sleep(0)       # last ~3 ms: spin, yielding the GIL

    def _run_tests(self):
        tests = self.cfg.get("tests", [])
        break_between = float(self.cfg.get("break_seconds_between_tests", 0.0))

        def log_event():
            self.logger.log_event(CLOCK() - self.t0, self.freq_set, self.duty_set, self.v_set_mv,
                                  self._last_cmd)

        try:
            for i, test in enumerate(tests, 1):
                if self.stop_flag.is_set():
                    break

                ttype = (test.get("type") or "constant").lower()
                repeat = int(test.get("repeat", 1))
                self._emit("status", f"Running block {i}/{len(tests)}: {ttype} x{repeat if repeat > 0 else 'inf'}")

                self._exec_block(test, log_event, status_prefix=f"Block {i}/{len(tests)}: ")

                if self.stop_flag.is_set():
                    break

                if i < len(tests) and break_between > 0:
                    self._emit("status", f"Break {break_between:.1f}s ...")
                    t_break = CLOCK()
                    self._nominal_begin(t_break, break_between)
                    t_end = t_break + break_between
                    while CLOCK() < t_end and not self.stop_flag.is_set():
                        time.sleep(0.05)
                    if not self.stop_flag.is_set():
                        self._nominal_end(break_between)

            if not self.stop_flag.is_set():
                self._emit("status", "Test sequence complete.")
        except Exception:
            log_exception("PumpTestRunner run loop crashed; flushing data to disk")
            try:
                self.logger.flush(sync=True)
            except Exception:
                log_exception("PumpTestRunner: failed to flush after run crash")
            self._emit("error", "Test run stopped due to an error. "
                                "Data collected so far has been saved to disk.")
        finally:
            with self.lock:
                self.t_end = CLOCK()
                self.voltage_request = None
                self.operator_request = None
            self.stop_flag.set()

    def start(self, before_run: Optional[Callable[[], Any]] = None):
        """Open hardware, set t0, start sampling, call before_run (e.g. start the
        video recording on the same t0), then start the test sequence."""
        self._setup_hardware()
        hires_timer(True)
        self._hires = True
        self.t0 = CLOCK()
        self.stop_flag.clear()
        if self.meter:
            self.meter.start(self.t0)
        if self.ctl:
            self.ctl.start(self.t0)
        self.t_sample = threading.Thread(target=self._sampling_loop, daemon=True)
        self.t_run = threading.Thread(target=self._run_tests, daemon=True)
        self.t_sample.start()
        if before_run:
            try:
                before_run()
            except Exception:
                log_exception("PumpTestRunner.start: before_run hook failed (run continues)")
        self.t_run.start()

    def stop(self):
        self.stop_flag.set()

    def join(self, timeout: Optional[float] = None):
        for t in (self.t_run, self.t_sample):
            try:
                if t:
                    t.join(timeout=timeout)
            except Exception:
                log_exception("PumpTestRunner.join: failed while joining worker thread")
        self._teardown_hardware()
        self._release_hires()

    def get_plot_data(self):
        with self.lock:
            return list(self.buf_t), list(self.buf_freq), list(self.buf_flow)


# ---------------- video recording ----------------
VIDEO_DEFAULTS: Dict[str, Any] = {
    "codec": "mjpg",        # "mjpg" (AVI, quality settable), "ffv1" (lossless AVI, large), "mp4v" (old default)
    "quality": 95,          # MJPG quality 0-100
    "overlay": True,        # burn time/frequency/flow into the frames (covers the top-left corner)
    "chunk_s": 60.0,        # a new video file every chunk_s seconds (crash safety)
    "crop": None,           # [x, y, w, h]: write only this strip (e.g. the tube and the scale)
    "queue_mb": 256,        # frames waiting for the writer; beyond this frames are dropped and counted
}
VIDEO_CODECS = {"mjpg": ("MJPG", ".avi"), "ffv1": ("FFV1", ".avi"), "mp4v": ("mp4v", ".mp4")}
FRAME_COLS = ["frame_index", "t_s", "t_perf_s", "cam_pos_msec", "dropped_before", "video_file", "video_frame"]


class DummyCamera:
    """Synthetic camera for tests without hardware: a dark meniscus band that
    oscillates once per stroke and advances a little with every stroke."""
    PX_PER_STROKE, OSC_PX = 1.5, 3.0

    def __init__(self, width: int = 640, height: int = 240, fps: float = 15.0,
                 hz_cb: Optional[Callable[[], float]] = None):
        self.w, self.h, self.fps = int(width), int(height), float(fps)
        self.hz_cb = hz_cb or (lambda: 0.0)
        self._t_next = CLOCK()
        self._t_last = None
        self._phase = 0.0          # strokes done (fractional)
        self._open = True

    def isOpened(self):
        return self._open

    def read(self):
        if not self._open:
            return False, None
        delay = self._t_next - CLOCK()
        if delay > 0:
            time.sleep(delay)
        self._t_next = max(self._t_next + 1.0 / self.fps, CLOCK() - 1.0 / self.fps)
        now = CLOCK()
        if self._t_last is not None:
            self._phase += max(0.0, float(self.hz_cb() or 0.0)) * (now - self._t_last)
        self._t_last = now
        x = int(self.w * 0.2 + self.PX_PER_STROKE * self._phase
                + self.OSC_PX * math.sin(2 * math.pi * self._phase)) % self.w
        frame = np.full((self.h, self.w, 3), 200, np.uint8)
        frame[:, max(0, x - 3):x + 3] = 40
        return True, frame

    def get(self, prop):
        if CV2_AVAILABLE:
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return float(self.w)
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return float(self.h)
            if prop == cv2.CAP_PROP_FPS:
                return self.fps
        return 0.0

    def set(self, prop, value):
        return False

    def release(self):
        self._open = False


class VideoRecorder:
    """USB camera with a capture thread and a separate writer thread.

    The capture thread reads frames as fast as the camera delivers them and
    stamps each with CLOCK() right after read() returns. It feeds the preview
    (latest frame) and, while recording, a bounded queue to the writer thread,
    which encodes the video (in chunks) and writes one frames.csv row per
    written frame. Encoding can therefore never slow down capture; if the
    writer falls behind by more than queue_mb, frames are dropped and counted
    (dropped_before). The video header carries the measured frame rate.
    """

    def __init__(self):
        self.cap = None
        self.camera = None
        self.frame_width = 1920
        self.frame_height = 1080
        self._cap_lock = threading.Lock()
        self._cap_thread: Optional[threading.Thread] = None
        self._cap_stop = threading.Event()
        self._latest = None                          # (index, t, frame)
        self._latest_lock = threading.Lock()
        self._idx = 0
        self._times: deque = deque(maxlen=150)      # recent capture times, for the frame rate
        self._read_fail = 0
        # recording
        self.recording = False
        self._q: Optional["queue.Queue"] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._drop_pending = 0
        self.dropped_total = 0
        self.frames_written = 0
        self.rec_fps = None
        self.video_paths: List[str] = []
        self.info_cb: Optional[Callable[[], Tuple[float, int, float]]] = None

    # ---- camera ----
    def open(self, camera_index: Any = 0, hz_cb: Optional[Callable[[], float]] = None) -> bool:
        """Open a camera ("dummy" = synthetic test camera) and start capturing."""
        self.close()
        try:
            if str(camera_index).lower() == "dummy":
                cap = DummyCamera(hz_cb=hz_cb)
                self.camera = "dummy"
            else:
                if not CV2_AVAILABLE:
                    log_warning("VideoRecorder.open: OpenCV not available; cannot open camera.")
                    return False
                cap = cv2.VideoCapture(int(camera_index))
                self.camera = int(camera_index)
                if not cap.isOpened():
                    log_error(f"VideoRecorder.open: failed to open camera index {camera_index}")
                    cap.release()
                    return False
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if CV2_AVAILABLE else 0
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if CV2_AVAILABLE else 0
            if width > 0 and height > 0:
                self.frame_width, self.frame_height = width, height
            with self._cap_lock:
                self.cap = cap
            self._idx = 0
            self._times.clear()
            self._cap_stop.clear()
            self._cap_thread = threading.Thread(target=self._capture_loop, name="camera-capture", daemon=True)
            self._cap_thread.start()
            return True
        except Exception:
            log_exception(f"VideoRecorder.open: exception opening camera {camera_index}")
            with self._cap_lock:
                self.cap = None
            return False

    def _capture_loop(self):
        while not self._cap_stop.is_set():
            with self._cap_lock:
                cap = self.cap
            if cap is None:
                return
            try:
                ok, frame = cap.read()
                t = CLOCK()
            except Exception:
                log_exception_limited("camera_read", "Camera read failed")
                ok, frame = False, None
            if not ok or frame is None:
                self._read_fail += 1
                if self._read_fail in (50, 500):
                    log_warning(f"Camera: {self._read_fail} failed reads in a row")
                time.sleep(0.01)
                continue
            self._read_fail = 0
            self._idx += 1
            idx = self._idx
            self.frame_height, self.frame_width = frame.shape[:2]
            self._times.append(t)
            with self._latest_lock:
                self._latest = (idx, t, frame)
            if self.recording and self._q is not None:
                try:
                    pos = float(cap.get(cv2.CAP_PROP_POS_MSEC)) if CV2_AVAILABLE else 0.0
                except Exception:
                    pos = 0.0
                try:
                    self._q.put_nowait((idx, t, pos, frame, self._drop_pending))
                    self._drop_pending = 0
                except queue.Full:
                    self._drop_pending += 1
                    self.dropped_total += 1

    def measured_fps(self) -> Optional[float]:
        """Capture rate over the recent frames (None until ~1 s of frames)."""
        ts = list(self._times)
        if len(ts) < 3 or ts[-1] - ts[0] < 1.0:
            return None
        return (len(ts) - 1) / (ts[-1] - ts[0])

    def close(self):
        """Stop recording and capturing, release the camera."""
        self.stop_recording()
        self._cap_stop.set()
        if self._cap_thread and self._cap_thread is not threading.current_thread():
            self._cap_thread.join(timeout=2.0)
        self._cap_thread = None
        with self._cap_lock:
            cap, self.cap = self.cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                log_exception("VideoRecorder.close: failed to release camera")
        with self._latest_lock:
            self._latest = None

    def is_opened(self) -> bool:
        with self._cap_lock:
            return self.cap is not None and self.cap.isOpened()

    def read_frame(self, fresh: bool = False) -> Optional["np.ndarray"]:
        """Latest captured frame (a copy). The camera itself is only read by the
        capture thread; `fresh` is kept for compatibility."""
        with self._latest_lock:
            return self._latest[2].copy() if self._latest is not None else None

    # ---- recording ----
    def start_recording(self, run_dir: str, t0: float, video_cfg: Optional[Dict[str, Any]] = None,
                        info_cb: Optional[Callable[[], Tuple[float, int, float]]] = None) -> bool:
        """Record into run_dir (video_partNNNN.<ext> + frames.csv), times relative to t0 (CLOCK)."""
        if not self.is_opened():
            log_error("VideoRecorder.start_recording: camera not opened")
            return False
        if not CV2_AVAILABLE:
            log_error("VideoRecorder.start_recording: OpenCV not available")
            return False
        self.stop_recording()
        cfg = {**VIDEO_DEFAULTS, **(video_cfg or {})}
        codec = str(cfg.get("codec") or "mjpg").lower()
        if codec not in VIDEO_CODECS:
            log_warning(f"VideoRecorder: unknown codec {codec!r}; using mjpg")
            codec = "mjpg"
        # The header needs the real frame rate: wait (up to 3 s) for a measurement
        t_wait = CLOCK() + 3.0
        while self.measured_fps() is None and CLOCK() < t_wait:
            time.sleep(0.05)
        fps = self.measured_fps()
        if fps is None:
            fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0) or 15.0
            log_warning(f"VideoRecorder: frame rate not measured yet; header uses {fps:.2f} fps")
        with self._latest_lock:
            if self._latest is None:
                log_error("VideoRecorder.start_recording: no frames from the camera")
                return False
            h, w = self._latest[2].shape[:2]
        crop = cfg.get("crop")
        if crop:
            try:
                x, y, cw, ch = (int(v) for v in crop)
                if not (0 <= x < w and 0 <= y < h and cw > 0 and ch > 0):
                    raise ValueError
                cw, ch = min(cw, w - x), min(ch, h - y)
                crop = (x, y, cw, ch)
            except Exception:
                log_warning(f"VideoRecorder: invalid crop {crop!r} for {w}x{h}; recording the full frame")
                crop = None
        size = (crop[2], crop[3]) if crop else (w, h)
        frame_bytes = size[0] * size[1] * 3 + w * h * 3
        maxsize = max(8, int(float(cfg.get("queue_mb") or 256) * 2**20 / max(1, frame_bytes)))
        try:
            frames_f = open(os.path.join(run_dir, "frames.csv"), "x", newline="", encoding="utf-8")
        except Exception:
            log_exception(f"VideoRecorder: cannot create frames.csv in {run_dir}")
            return False
        self.rec_fps = fps
        self.frames_written = 0
        self.dropped_total = 0
        self._drop_pending = 0
        self.video_paths = []
        self.info_cb = info_cb
        self._q = queue.Queue(maxsize=maxsize)
        params = {"run_dir": run_dir, "t0": t0, "fps": fps, "codec": codec, "size": size, "crop": crop,
                  "quality": int(cfg.get("quality") or 95), "overlay": bool(cfg.get("overlay")),
                  "chunk_s": max(1.0, float(cfg.get("chunk_s") or 60.0)), "frames_f": frames_f,
                  "queue_max": maxsize}
        self._writer_thread = threading.Thread(target=self._writer_loop, args=(self._q, params),
                                               name="video-writer", daemon=True)
        self._writer_thread.start()
        self.recording = True
        return True

    def _open_chunk(self, p, n: int):
        fourcc, ext = VIDEO_CODECS[p["codec"]]
        path = os.path.join(p["run_dir"], f"video_part{n:04d}{ext}")
        args = (path, cv2.VideoWriter_fourcc(*fourcc), p["fps"], p["size"])
        writer = None
        if p["codec"] == "mjpg":
            # OpenCV's own MJPEG writer honours the quality setting
            writer = cv2.VideoWriter(path, cv2.CAP_OPENCV_MJPEG, *args[1:])
            if not writer.isOpened():
                writer.release()
                writer = None
        if writer is None:
            writer = cv2.VideoWriter(*args)
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"cannot open video writer {path} ({p['codec']})")
        if p["codec"] == "mjpg":
            writer.set(cv2.VIDEOWRITER_PROP_QUALITY, p["quality"])
        self.video_paths.append(path)
        return writer, os.path.basename(path)

    def _writer_loop(self, q: "queue.Queue", p: Dict[str, Any]):
        frames_f = p["frames_f"]
        w = csv.writer(frames_f, lineterminator="\n")
        w.writerow(FRAME_COLS)
        writer, name, chunk_n, chunk_frame, chunk_t0 = None, "", 0, 0, None
        gaps, t_prev, t_first, t_last = [], None, None, None
        last_flush = CLOCK()
        try:
            while True:
                item = q.get()
                if item is None:
                    break
                idx, t, pos, frame, dropped = item
                if writer is None or (t - chunk_t0) >= p["chunk_s"]:
                    if writer is not None:
                        writer.release()
                    chunk_n += 1
                    writer, name = self._open_chunk(p, chunk_n)
                    chunk_frame, chunk_t0 = 0, t
                img = frame
                if p["crop"]:
                    x, y, cw, ch = p["crop"]
                    img = img[y:y + ch, x:x + cw]
                if p["overlay"]:
                    img = img.copy()           # never draw on the frame the preview shares
                    ts, fq, flow = self.info_cb() if self.info_cb else (t - p["t0"], 0, 0.0)
                    self.add_overlay(img, t - p["t0"], fq, flow)
                writer.write(np.ascontiguousarray(img))
                w.writerow([idx, f"{t - p['t0']:.6f}", f"{t:.6f}", f"{pos:.3f}", dropped, name, chunk_frame])
                chunk_frame += 1
                self.frames_written += 1
                if t_prev is not None:
                    gaps.append(t - t_prev)
                t_prev = t
                t_first = t if t_first is None else t_first
                t_last = t
                if CLOCK() - last_flush >= 1.0:
                    frames_f.flush()
                    last_flush = CLOCK()
        except Exception:
            log_exception("VideoRecorder writer failed; recording stopped")
            self.recording = False
        finally:
            if writer is not None:
                writer.release()
            try:
                frames_f.flush()
                os.fsync(frames_f.fileno())
                frames_f.close()
            except Exception:
                pass
            try:
                n = self.frames_written
                g = np.array(gaps) if gaps else np.zeros(0)
                fps = (n - 1) / (t_last - t_first) if n > 1 and t_last > t_first else None
                summary = {
                    "frames_written": n, "frames_dropped": self.dropped_total,
                    "fps_header": p["fps"], "fps_measured": fps,
                    "interval_median_s": float(np.median(g)) if g.size else None,
                    "interval_max_s": float(g.max()) if g.size else None,
                    "gaps_over_1p5_intervals": int((g > 1.5 * np.median(g)).sum()) if g.size else 0,
                    "codec": p["codec"], "quality": p["quality"], "size": list(p["size"]),
                    "crop": list(p["crop"]) if p["crop"] else None, "overlay": p["overlay"],
                    "chunk_s": p["chunk_s"], "queue_max_frames": p["queue_max"],
                    "files": [os.path.basename(x) for x in self.video_paths],
                    "clock": "perf_counter; t_s = t_perf_s - run t0",
                }
                with open(os.path.join(p["run_dir"], "video_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
            except Exception:
                log_exception("VideoRecorder: could not write video_summary.json")

    def stop_recording(self):
        """Stop recording; the writer finishes the queued frames first."""
        if not self.recording and self._writer_thread is None:
            return
        self.recording = False
        q, th = self._q, self._writer_thread
        if q is not None:
            q.put(None)
        if th is not None and th is not threading.current_thread():
            th.join(timeout=30.0)
        self._q = None
        self._writer_thread = None

    def is_recording(self) -> bool:
        return self.recording

    def add_overlay(self, frame, t_s: float, freq: float, flow: float) -> "np.ndarray":
        """Burn time/frequency/flow into the top-left corner (in place)."""
        try:
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 5), (250, 90), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, f"Time: {t_s:.2f}s", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Freq: {freq:.0f} Hz", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Flow: {flow:.4f}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            return frame
        except Exception:
            log_exception_limited("video_overlay_error", "VideoRecorder.add_overlay: failed to add overlay")
            return frame


# ---------------- UI ----------------
class DashboardApp:
    def __init__(self, initial_config_path: Optional[str] = None):
        self.root = tk.Tk()
        self.root.title("Pump Testing Suite Dashboard (v2)")
        self._apply_ui_scaling()
        self._set_default_geometry()

        self.ui_events: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self.cfg: Dict[str, Any] = {"tests": []}

        self.active_stream: Optional[LiveSessionBase] = None
        self.run_dir: Optional[str] = None
        self.cfg_path: Optional[str] = None

        self._build_layout()
        self._build_plot()

        # Live plot every 150 ms (blitted, see _update_plot); skipped while another tab is shown
        self.root.after(150, self._plot_tick)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        if initial_config_path:
            try:
                self.load_config(initial_config_path)
            except Exception as e:
                log_exception(f"DashboardApp.__init__: failed to load config {initial_config_path}")
                messagebox.showwarning("Load config", f"Could not load config:\n{e}")

        self._mem_critical_warned = False
        self._poll_ui_events()
        if PSUTIL_AVAILABLE:
            self._poll_memory()

    # ---- layout ----
    def _build_layout(self):
        """Toolbar + progress row on top, status bar at the bottom, and three
        columns: left sidebar | tabbed plots | right sidebar. Sidebar panels
        are managed by ui_panels.PanelManager (View menu)."""
        self.panels = PanelManager(self.root)
        self.panels.load()
        geo = self.panels.window_geometry()
        if geo:
            self.root.geometry(geo)

        self.var_mode = tk.StringVar(value="scheduled")
        menubar = tk.Menu(self.root)
        fm = tk.Menu(menubar, tearoff=False)
        fm.add_command(label="Load config...", command=self.pick_and_load_config)
        fm.add_command(label="Save config as...", command=self.save_config_as)
        fm.add_separator()
        fm.add_command(label="Exit", command=self.on_close)
        menubar.add_cascade(label="File", menu=fm)

        # Toolbar: run buttons + status (always visible)
        btns = ttk.Frame(self.root)
        btns.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(6, 2))

        self.btn_start = ttk.Button(btns, text="Start Run", command=self.start_scheduled_run)
        self.btn_stop  = ttk.Button(btns, text="Stop", command=self.stop_active, state=tk.DISABLED)
        self.btn_manual_connect = ttk.Button(btns, text="Connect Manual", command=self.manual_connect)
        self.btn_manual_disconnect = ttk.Button(btns, text="Disconnect Manual", command=self.manual_disconnect, state=tk.DISABLED)

        self.btn_start.pack(side=tk.LEFT, padx=4)
        self.btn_stop.pack(side=tk.LEFT, padx=4)
        self.btn_manual_connect.pack(side=tk.LEFT, padx=(18, 4))
        self.btn_manual_disconnect.pack(side=tk.LEFT, padx=4)
        ttk.Label(btns, text="Mode:").pack(side=tk.LEFT, padx=(18, 2))
        ttk.Radiobutton(btns, text="Scheduled", variable=self.var_mode, value="scheduled",
                        command=self._update_mode_ui).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(btns, text="Manual", variable=self.var_mode, value="manual",
                        command=self._update_mode_ui).pack(side=tk.LEFT, padx=2)

        self.var_status = tk.StringVar(value="Idle.")
        ttk.Label(btns, textvariable=self.var_status).pack(side=tk.LEFT, padx=12)

        self._build_progress_row()
        self._build_status_bar()

        # Three columns; the sidebars hide themselves when they hold no visible panel
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=(0, 4))
        self.panels.paned = main
        left = self.panels.add_dock("left", main)
        left.canvas.configure(width=430)
        self.center_tabs = ttk.Notebook(main)
        right = self.panels.add_dock("right", main)
        right.canvas.configure(width=420)
        main.add(left, weight=0)
        main.add(self.center_tabs, weight=1)
        main.add(right, weight=0)
        self.panels.set_columns([left, self.center_tabs, right])

        self.plot_frame = ttk.Frame(self.center_tabs)
        self.center_tabs.add(self.plot_frame, text="Live plots")
        self._build_pq_tab()
        # Redrawing a plot nobody sees costs CPU (and GIL time the drive timing needs)
        self.center_tabs.bind("<<NotebookTabChanged>>", self._on_center_tab)

        # Video state is shared by the camera panels
        self.video_recorder = None
        self.video_recording = False
        self.var_video_enabled = tk.BooleanVar(value=False)
        self.var_video_camera_index = tk.StringVar(value="0")
        self.var_video_status = tk.StringVar(value="Camera: Not connected")

        P = self.panels
        P.add_panel("run", "Run details & storage", "left", self._build_run_details)
        P.add_panel("hardware", "Hardware", "left", self._build_hardware)
        P.add_panel("config", "Config builder", "left", self._build_config_builder)
        P.add_panel("manual", "Manual control", "right", self._build_manual_panel)
        P.add_panel("power", "Power meter (PPK2)", "right", self._build_power_panel)
        P.add_panel("pressure", "Pressure control (syringe)", "right", self._build_pressure_panel)
        P.add_panel("camera", "Camera", "right",
                    lambda body: self._build_video_controls(body).pack(side=tk.TOP, fill=tk.X))
        P.add_panel("preview", "Camera preview", "right",
                    lambda body: self._build_video_preview(body).pack(side=tk.TOP, fill=tk.BOTH, expand=True))

        P.add_preset("Run", {"run": "open", "hardware": "collapsed", "config": "open", "manual": "collapsed",
                             "power": "collapsed", "pressure": "collapsed", "camera": "collapsed", "preview": "open"})
        P.add_preset("Setup", {"run": "open", "hardware": "open", "config": "collapsed", "manual": "collapsed",
                               "power": "open", "pressure": "open", "camera": "open", "preview": "open"})
        P.add_preset("Manual", {"run": "collapsed", "hardware": "open", "config": "hidden", "manual": "open",
                                "power": "open", "pressure": "collapsed", "camera": "collapsed", "preview": "open"})
        P.add_preset("PQ", {"run": "collapsed", "hardware": "collapsed", "config": "open", "manual": "open",
                            "power": "collapsed", "pressure": "open", "camera": "hidden", "preview": "hidden"})
        P.add_preset("Video", {"run": "collapsed", "hardware": "collapsed", "config": "collapsed",
                               "manual": "collapsed", "power": "hidden", "pressure": "hidden", "camera": "open",
                               "preview": "open"})
        P.add_preset("Plots only", {k: "hidden" for k in ("run", "hardware", "config", "manual", "power",
                                                           "pressure", "camera", "preview")})
        P.add_preset("Compact", {"config": "open"})
        P.add_preset("All open", {k: "open" for k in ("run", "hardware", "config", "manual", "power",
                                                       "pressure", "camera", "preview")})
        P.build_view_menu(menubar)
        self.root.config(menu=menubar)

        self._update_mode_ui()
        self._refresh_manual_duty_state()

        # Start video preview if camera is connected
        self._init_video()
        P.restore(default_preset="Run")

    def _build_run_details(self, meta):
        self.var_test_name = tk.StringVar(value="Pump Test")
        self.var_operator  = tk.StringVar(value="")
        self.var_pump_id   = tk.StringVar(value="")
        self.var_out_dir   = tk.StringVar(value=os.path.abspath("./pump_runs"))
        self.var_log_manual = tk.BooleanVar(value=False)
        self.txt_desc      = tk.Text(meta, height=3, width=40)
        self._default_test_name = self.var_test_name.get()
        self._default_operator = self.var_operator.get()
        self._default_desc = ""
        meta.columnconfigure(1, weight=1)

        ttk.Label(meta, text="Name:").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(meta, textvariable=self.var_test_name).grid(row=0, column=1, columnspan=2, sticky="we", padx=4, pady=2)

        ttk.Label(meta, text="Operator:").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        op_row = ttk.Frame(meta)
        op_row.grid(row=1, column=1, columnspan=2, sticky="w", padx=4, pady=2)
        ttk.Entry(op_row, textvariable=self.var_operator, width=14).pack(side=tk.LEFT)
        ttk.Label(op_row, text="Pump ID:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(op_row, textvariable=self.var_pump_id, width=14).pack(side=tk.LEFT)

        ttk.Label(meta, text="Output folder:").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(meta, textvariable=self.var_out_dir).grid(row=2, column=1, sticky="we", padx=4, pady=2)
        ttk.Button(meta, text="Browse...", command=self.pick_out_dir).grid(row=2, column=2, padx=4, pady=2)

        ttk.Label(meta, text="Description:").grid(row=3, column=0, sticky="nw", padx=4, pady=2)
        self.txt_desc.grid(row=3, column=1, columnspan=2, sticky="we", padx=4, pady=2)

        ttk.Checkbutton(meta, text="Log manual sessions to disk", variable=self.var_log_manual)\
            .grid(row=4, column=1, sticky="w", padx=4, pady=(2, 4))

    def _build_hardware(self, hw):
        self.var_cfg_file = tk.StringVar(value="")
        self.var_port     = tk.StringVar(value="COM22")
        self.var_baud     = tk.StringVar(value="115200")
        self.var_protocol = tk.StringVar(value="freq_only")

        self.var_sensor_backend = tk.StringVar(value="fluigent" if FLUIGENT_AVAILABLE else "dummy")
        self.var_sensor_channel = tk.StringVar(value="0")
        self.var_flow_server    = tk.StringVar(value="localhost:8765")
        self.var_sample_rate    = tk.StringVar(value="25")
        self.var_break_between  = tk.StringVar(value="0")
        hw.columnconfigure(1, weight=1)

        ttk.Label(hw, text="Config file:").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(hw, textvariable=self.var_cfg_file).grid(row=0, column=1, columnspan=2, sticky="we", padx=4, pady=2)
        cf = ttk.Frame(hw)
        cf.grid(row=0, column=3, sticky="e")
        ttk.Button(cf, text="Load", command=self.pick_and_load_config, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Button(cf, text="Save As", command=self.save_config_as, width=8).pack(side=tk.LEFT, padx=2)

        ttk.Label(hw, text="Serial port:").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(hw, textvariable=self.var_port, width=10).grid(row=1, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(hw, text="Baud:").grid(row=1, column=2, sticky="e", padx=4, pady=2)
        ttk.Entry(hw, textvariable=self.var_baud, width=9).grid(row=1, column=3, sticky="w", padx=4, pady=2)

        ttk.Label(hw, text="Protocol:").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        proto = ttk.Combobox(hw, state="readonly", width=10, values=["freq_only", "freq_duty"], textvariable=self.var_protocol)
        proto.grid(row=2, column=1, sticky="w", padx=4, pady=2)
        proto.bind("<<ComboboxSelected>>", lambda *_: self._refresh_manual_duty_state())

        ttk.Label(hw, text="Flow backend:").grid(row=3, column=0, sticky="w", padx=4, pady=2)
        ttk.Combobox(hw, state="readonly", width=10, values=["fluigent", "dummy", "network"], textvariable=self.var_sensor_backend)\
            .grid(row=3, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(hw, text="Channel:").grid(row=3, column=2, sticky="e", padx=4, pady=2)
        ttk.Entry(hw, textvariable=self.var_sensor_channel, width=9).grid(row=3, column=3, sticky="w", padx=4, pady=2)

        ttk.Label(hw, text="Sample rate (Hz):").grid(row=4, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(hw, textvariable=self.var_sample_rate, width=10).grid(row=4, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(hw, text="Break (s):").grid(row=4, column=2, sticky="e", padx=4, pady=2)
        ttk.Entry(hw, textvariable=self.var_break_between, width=9).grid(row=4, column=3, sticky="w", padx=4, pady=2)

        ttk.Label(hw, text="Flow server:").grid(row=5, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(hw, textvariable=self.var_flow_server).grid(row=5, column=1, columnspan=3, sticky="we", padx=4, pady=2)

    # ---- status bar: device lights + memory ----
    def _build_status_bar(self):
        bar = tk.Frame(self.root, bd=1, relief=tk.SUNKEN)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        self._status_lights: Dict[str, tk.Label] = {}
        for key in ("arduino", "flow", "ppk2", "pressure", "camera"):
            lbl = tk.Label(bar, text=f"● {key}", fg="#888888", padx=8)
            lbl.pack(side=tk.LEFT)
            self._status_lights[key] = lbl
        self.var_mem = tk.StringVar(value="")
        tk.Label(bar, textvariable=self.var_mem, padx=8).pack(side=tk.RIGHT)

    def _set_light(self, key: str, state: str, text: str):
        colours = {"ok": "#1a7f37", "bad": "#c62828", "off": "#888888", "busy": "#b26a00"}
        lbl = self._status_lights.get(key)
        if lbl is not None:
            lbl.configure(text=f"● {text}", fg=colours.get(state, "#888888"))

    def _update_status_bar(self):
        s = self.active_stream
        if s is not None and getattr(s, "ser", None) is not None:
            self._set_light("arduino", "ok", f"Arduino {s.cfg.get('port', '')}")
        elif s is not None:
            self._set_light("arduino", "busy", "Arduino (dry run)")
        else:
            self._set_light("arduino", "off", "Arduino")
        backend = self.var_sensor_backend.get()
        if s is not None and getattr(s, "t0", None) is not None:
            with s.lock:
                last = s.buf_t[-1] if s.buf_t else None
            fresh = last is not None and (CLOCK() - s.t0 - last) < 2.0
            self._set_light("flow", "ok" if fresh else "bad", f"Flow {backend}" + ("" if fresh else " (no data)"))
        else:
            self._set_light("flow", "off", f"Flow {backend}")
        live = s.get_power_live() if s is not None else None
        if live is None:
            self._set_light("ppk2", "off", "PPK2")
        else:
            st = live.get("state", "?")
            self._set_light("ppk2", "ok" if st == "measuring" else "bad" if st in ("tripped", "lost") else "busy",
                            f"PPK2 {st}")
        pst = s.get_pressure_status() if s is not None else None
        if pst is None:
            self._set_light("pressure", "off", "Pressure")
        else:
            bad = pst["mode"] == "fault" or pst["stale"]
            busy = pst["mode"] in ("hold", "release") and not pst["settled"]
            p = pst["p"]
            self._set_light("pressure", "bad" if bad else "busy" if busy else "ok",
                            "Pressure " + ("FAULT" if pst["mode"] == "fault" else
                                           "no reading" if pst["stale"] or p != p else f"{p:.1f} mbar"))
        vr = self.video_recorder
        if vr is not None and vr.is_opened():
            fps = vr.measured_fps()
            txt = f"Camera {fps:.1f} fps" if fps else "Camera"
            self._set_light("camera", "busy" if vr.is_recording() else "ok", txt + (" · REC" if vr.is_recording() else ""))
        else:
            self._set_light("camera", "off", "Camera")
        if PSUTIL_AVAILABLE:
            try:
                rss = psutil.Process().memory_info().rss / 2**20
                free = psutil.virtual_memory().available / 2**20
                self.var_mem.set(f"Dashboard RAM {rss:.0f} MB · free {free / 1024:.1f} GB")
            except Exception:
                pass

    # ---- run timer + manual voltage steps ----
    def _build_progress_row(self):
        prog = ttk.Frame(self.root)
        prog.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))
        self._progress_row = prog
        self.pbar = ttk.Progressbar(prog, mode="determinate", maximum=1000, length=220)
        self.pbar.pack(side=tk.LEFT, padx=4)
        self.var_progress = tk.StringVar(value="")
        ttk.Label(prog, textvariable=self.var_progress).pack(side=tk.LEFT, padx=8)
        ttk.Button(prog, text="Voltage schedule...", command=self.show_voltage_schedule)\
            .pack(side=tk.RIGHT, padx=4)
        self.var_next_change = tk.StringVar(value="")
        self.lbl_next_change = tk.Label(prog, textvariable=self.var_next_change)
        self.lbl_next_change.pack(side=tk.RIGHT, padx=8)
        self._next_change_fg = self.lbl_next_change.cget("fg")

        # Shown only while a run waits for the operator to change the supply voltage
        bg = "#ffd966"
        self.voltage_banner = tk.Frame(self.root, bg=bg, bd=2, relief=tk.RIDGE)
        self.var_vprompt_title = tk.StringVar(value="")
        self.var_vprompt_info = tk.StringVar(value="")
        self.var_vprompt_read = tk.StringVar(value="")
        tk.Label(self.voltage_banner, textvariable=self.var_vprompt_title, bg=bg,
                 font=("Segoe UI", 14, "bold")).pack(side=tk.TOP, anchor="w", padx=10, pady=(6, 0))
        tk.Label(self.voltage_banner, textvariable=self.var_vprompt_info, bg=bg, justify=tk.LEFT)\
            .pack(side=tk.TOP, anchor="w", padx=10)
        row = tk.Frame(self.voltage_banner, bg=bg)
        row.pack(side=tk.TOP, anchor="w", padx=10, pady=(4, 8))
        tk.Label(row, text="Voltage read on the supply (V):", bg=bg).pack(side=tk.LEFT)
        self.ent_vprompt = ttk.Entry(row, textvariable=self.var_vprompt_read, width=10)
        self.ent_vprompt.pack(side=tk.LEFT, padx=6)
        self.ent_vprompt.bind("<Return>", lambda _e: self.confirm_voltage_change())
        ttk.Button(row, text="Confirm voltage set", command=self.confirm_voltage_change).pack(side=tk.LEFT, padx=6)
        self._vprompt_index: Optional[int] = None
        self._last_bell = 0.0
        self._last_status_json = 0.0

        # Shown while a run waits for another operator action (clearing the syringe by hand)
        bg2 = "#f8cbad"
        self.operator_banner = tk.Frame(self.root, bg=bg2, bd=2, relief=tk.RIDGE)
        self.var_oprompt_title = tk.StringVar(value="")
        self.var_oprompt_info = tk.StringVar(value="")
        tk.Label(self.operator_banner, textvariable=self.var_oprompt_title, bg=bg2,
                 font=("Segoe UI", 14, "bold")).pack(side=tk.TOP, anchor="w", padx=10, pady=(6, 0))
        tk.Label(self.operator_banner, textvariable=self.var_oprompt_info, bg=bg2, justify=tk.LEFT,
                 wraplength=1100).pack(side=tk.TOP, anchor="w", padx=10)
        orow = tk.Frame(self.operator_banner, bg=bg2)
        orow.pack(side=tk.TOP, anchor="w", padx=10, pady=(4, 8))
        ttk.Button(orow, text="Done - continue", command=self.confirm_operator_prompt).pack(side=tk.LEFT)
        self._oprompt_index: Optional[int] = None

        self.root.after(1000, self._tick_progress)

    def _supply_cfg(self) -> Dict[str, Any]:
        """Bench supply settings: config file values, overridden by the power panel fields."""
        s = {**SUPPLY_DEFAULTS, **(self.cfg.get("supply") or {})}
        if hasattr(self, "var_supply_v"):
            v = self.var_supply_v.get().strip().replace(",", ".")
            if v:
                s["initial_mv"] = int(round(safe_float(v, 2.5) * 1000))
            s["current_limit_ma"] = safe_float(self.var_supply_limit.get(), float(s.get("current_limit_ma") or 60))
        return s

    def _power_cfg(self) -> Dict[str, Any]:
        p = {**POWER_DEFAULTS, **(self.cfg.get("power") or {})}
        if hasattr(self, "var_pw_enabled"):
            watch = self.var_pw_watch.get().strip()
            p.update(enabled=bool(self.var_pw_enabled.get()), backend=self.var_pw_backend.get().strip() or "ppk2",
                     port=self.var_pw_port.get().strip() or "auto",
                     bin_ms=max(1.0, safe_float(self.var_pw_bin.get(), 10.0)),
                     watch_ma=None if watch == "" else safe_float(watch, 0.0))
        return p

    # ---- power meter panel ----
    def _build_power_panel(self, parent):
        p = {**POWER_DEFAULTS}
        s = {**SUPPLY_DEFAULTS}
        self.var_pw_enabled = tk.BooleanVar(value=False)
        self.var_pw_backend = tk.StringVar(value=str(p.get("backend", "ppk2")))
        self.var_pw_port = tk.StringVar(value=str(p.get("port", "auto")))
        self.var_pw_bin = tk.StringVar(value=str(p.get("bin_ms", 10)))
        self.var_pw_watch = tk.StringVar(value="")
        self.var_supply_v = tk.StringVar(value="2.500")
        self.var_supply_limit = tk.StringVar(value=str(s.get("current_limit_ma", 60)))
        self.var_pw_live = tk.StringVar(value="Power meter off." if POWER_AVAILABLE
                                        else "Power meter unavailable (power_ppk2.py / numpy missing).")

        r0 = ttk.Frame(parent); r0.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(6, 2))
        ttk.Checkbutton(r0, text="Measure power", variable=self.var_pw_enabled,
                        state=tk.NORMAL if POWER_AVAILABLE else tk.DISABLED).pack(side=tk.LEFT)
        ttk.Label(r0, text="Backend:").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Combobox(r0, state="readonly", width=7, values=["ppk2", "dummy"],
                     textvariable=self.var_pw_backend).pack(side=tk.LEFT)
        ttk.Label(r0, text="Port:").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Entry(r0, textvariable=self.var_pw_port, width=8).pack(side=tk.LEFT)
        ttk.Button(r0, text="Detect", command=self._detect_ppk2).pack(side=tk.LEFT, padx=4)
        r0b = ttk.Frame(parent); r0b.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2)
        ttk.Label(r0b, text="Bin (ms):").pack(side=tk.LEFT)
        ttk.Entry(r0b, textvariable=self.var_pw_bin, width=5).pack(side=tk.LEFT, padx=4)
        ttk.Label(r0b, text="Supply voltage at start (V):").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Entry(r0b, textvariable=self.var_supply_v, width=7).pack(side=tk.LEFT, padx=4)

        r1 = ttk.Frame(parent); r1.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2)
        ttk.Label(r1, text="Bench limit (mA):").pack(side=tk.LEFT)
        ttk.Entry(r1, textvariable=self.var_supply_limit, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(r1, text="Stop at (mA):").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Entry(r1, textvariable=self.var_pw_watch, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(r1, text="(blank = 95 % of limit)").pack(side=tk.LEFT, padx=2)

        r2 = ttk.Frame(parent); r2.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(2, 6))
        self.lbl_pw_live = tk.Label(r2, textvariable=self.var_pw_live, anchor="w", wraplength=380,
                                    justify=tk.LEFT)
        self.lbl_pw_live.pack(side=tk.LEFT)
        self._pw_live_fg = self.lbl_pw_live.cget("fg")

        # Missing packages (e.g. ppk2-api): install into exactly the Python running this window
        self._pip_running = False
        self.btn_pw_install = ttk.Button(r2, text="Install missing packages", command=self._install_packages)
        self._refresh_missing_packages()

    def _refresh_missing_packages(self):
        missing = missing_feature_packages()
        if missing:
            self.btn_pw_install.pack(side=tk.RIGHT, padx=4)
            names = ", ".join(f"{pip} ({why})" for _, pip, why in missing)
            self.var_pw_live.set(f"Not installed for this Python ({sys.version.split()[0]}): {names}. "
                                 f"Click 'Install missing packages'.")
            self.lbl_pw_live.configure(fg="#b00000")
        else:
            self.btn_pw_install.pack_forget()
        return missing

    def _install_packages(self):
        """pip install requirements.txt (+ the bundled Fluigent SDK if missing) into
        sys.executable, i.e. the Python this dashboard runs on, in the background."""
        if self._pip_running:
            return
        self._pip_running = True
        self.btn_pw_install.configure(state=tk.DISABLED)
        self.var_pw_live.set(f"Installing packages into {sys.executable} ... (can take a few minutes)")
        here = os.path.dirname(os.path.abspath(__file__))
        cmds = [[sys.executable, "-m", "pip", "install", "-r", os.path.join(here, "requirements.txt")]]
        if importlib.util.find_spec("Fluigent") is None and os.path.isdir(FLUIGENT_SDK_DIR):
            cmds.append([sys.executable, "-m", "pip", "install", FLUIGENT_SDK_DIR])

        def work():
            ok, out = True, []
            for cmd in cmds:
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    out.append((r.stdout or "") + (r.stderr or ""))
                    ok = ok and r.returncode == 0
                except Exception as e:
                    out.append(f"{' '.join(cmd)}: {e!r}")
                    ok = False
            self.ui_events.put(("pip_done", (ok, "\n".join(out))))

        threading.Thread(target=work, name="pip-install", daemon=True).start()

    def _on_pip_done(self, ok: bool, out: str):
        self._pip_running = False
        self.btn_pw_install.configure(state=tk.NORMAL)
        missing = self._refresh_missing_packages()
        if ok and not missing:
            self.var_pw_live.set("Packages installed.")
            self.lbl_pw_live.configure(fg=self._pw_live_fg)
            messagebox.showinfo("Packages installed",
                                f"Installed into {sys.executable}.\n\nThe PPK2 power meter works right away. "
                                f"Restart the dashboard to enable the network flow backend / Fluigent sensor "
                                f"if they were missing.")
        else:
            tail = "\n".join(out.strip().splitlines()[-15:])
            messagebox.showerror("Install failed", f"pip reported a problem (Python: {sys.executable}).\n\n{tail}")

    def _detect_ppk2(self):
        ports = find_ppk2_ports() if POWER_AVAILABLE else []
        if ports:
            self.var_pw_port.set(ports[0])
            self.var_pw_live.set(f"PPK2 found on {', '.join(ports)}.")
        else:
            self.var_pw_live.set("No PPK2 found (plugged in, switched on, nRF Connect app closed?).")

    def _update_power_ui(self):
        s = self.active_stream
        live = s.get_power_live() if s is not None else None
        if live is None:
            # Session ended: keep the last reading on screen, marked as such
            if getattr(self, "_pw_last_txt", None):
                self.var_pw_live.set("Stopped. Last reading: " + self._pw_last_txt)
                self._pw_last_txt = None
                self.lbl_pw_live.configure(fg=self._pw_live_fg)
            return None
        state = live.get("state", "?")
        if "i_ma" in live:
            txt = (f"{live['i_ma']:.3f} mA (peak {live['i_max_ma']:.1f}) · {live['p_mw']:.2f} mW · "
                   f"{live['v_mv'] / 1000:.3f} V · {live['energy_j']:.3f} J so far · port {live.get('port')}")
            if live.get("lost"):
                txt += f" · {live['lost']} samples lost"
        else:
            txt = f"port {live.get('port')}: waiting for data"
        bad = state in ("tripped", "lost", "reconnecting")
        txt = ("" if state == "measuring" else f"{state.upper()} · ") + txt
        self.var_pw_live.set(txt)
        self._pw_last_txt = txt
        self.lbl_pw_live.configure(fg="#b00000" if bad else self._pw_live_fg)
        return live

    # ---- pressure control panel ----
    def _pressure_cfg(self) -> Dict[str, Any]:
        """Pressure control settings: config file values, overridden by the panel fields."""
        p = dict(self.cfg.get("pressure") or {})
        if hasattr(self, "var_pr_enabled"):
            sensor = dict(p.get("sensor") or {})
            sensor.update(backend=self.var_pr_sensor.get().strip() or "fluigent",
                          channel=safe_int(self.var_pr_channel.get(), 1))
            if sensor["backend"] == "network" and not sensor.get("server"):
                sensor["server"] = self.var_flow_server.get().strip() or "localhost:8765"
            p.update(enabled=bool(self.var_pr_enabled.get()) and PRESSURE_AVAILABLE,
                     backend=self.var_pr_backend.get().strip() or "arduino",
                     port=self.var_pr_port.get().strip() or "COM8",
                     firmware=self.var_pr_fw.get().strip() or "auto",
                     limit_mbar=safe_float(self.var_pr_limit.get(), 300.0),
                     min_mbar=safe_float(self.var_pr_min.get(), -30.0),
                     microsteps=max(1, safe_int(self.var_pr_micro.get(), 1)), sensor=sensor)
        return p

    def _cal_summary(self) -> str:
        if not PRESSURE_AVAILABLE:
            return ""
        cal = pc.load_calibration(pressure_cal_path())
        if not cal:
            return "Not calibrated yet (Calibrate...)."
        g1, g2 = cal.get("gain_push_mbar_step"), cal.get("gain_pull_mbar_step")
        txt = f"Calibrated {str(cal.get('saved_at', '?'))[:16].replace('T', ' ')}"
        if g1 and g2:
            txt += f": push {g1:.2f} / pull {g2:.2f} mbar/step"
        if cal.get("backlash_steps") is not None:
            txt += f", backlash {cal['backlash_steps']:g}"
        if cal.get("sensor_offset_mbar") is not None:
            txt += f", zero {cal['sensor_offset_mbar']:+.2f} mbar"
        return txt

    def _build_pressure_panel(self, parent):
        d = pc.PRESSURE_DEFAULTS if PRESSURE_AVAILABLE else {"sensor": {}}
        self.var_pr_enabled = tk.BooleanVar(value=False)
        self.var_pr_backend = tk.StringVar(value="arduino")
        self.var_pr_port = tk.StringVar(value=str(d.get("port", "COM8")))
        self.var_pr_fw = tk.StringVar(value="auto")
        self.var_pr_sensor = tk.StringVar(value="fluigent" if FLUIGENT_AVAILABLE else "network")
        self.var_pr_channel = tk.StringVar(value=str(d["sensor"].get("channel", 1)))
        self.var_pr_limit = tk.StringVar(value=str(d.get("limit_mbar", 300)))
        self.var_pr_micro = tk.StringVar(value=str(d.get("microsteps", 1)))
        self.var_pr_min = tk.StringVar(value=str(d.get("min_mbar", -30)))
        self.var_pr_target = tk.StringVar(value="50")
        self.var_pr_jog = tk.StringVar(value="5")
        self.var_pr_jog_unit = tk.StringVar(value="full steps")
        self.var_pr_nolimits = tk.BooleanVar(value=False)
        self.var_pr_speed = tk.StringVar(value="20")
        self._pr_hold_dir = 0
        self.var_pr_live = tk.StringVar(value="Pressure control off." if PRESSURE_AVAILABLE
                                        else "Pressure control unavailable (pressure_control.py missing).")
        self.var_pr_cal = tk.StringVar(value=self._cal_summary())
        state = tk.NORMAL if PRESSURE_AVAILABLE else tk.DISABLED

        r0 = ttk.Frame(parent); r0.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(6, 2))
        ttk.Checkbutton(r0, text="Control pressure", variable=self.var_pr_enabled, state=state).pack(side=tk.LEFT)
        ttk.Label(r0, text="Syringe:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Combobox(r0, state="readonly", width=7, values=["arduino", "dummy"],
                     textvariable=self.var_pr_backend).pack(side=tk.LEFT)
        ttk.Label(r0, text="Port:").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Entry(r0, textvariable=self.var_pr_port, width=6).pack(side=tk.LEFT)
        ttk.Label(r0, text="FW:").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Combobox(r0, state="readonly", width=4, values=["auto", "v1", "v2"],
                     textvariable=self.var_pr_fw).pack(side=tk.LEFT)
        r1 = ttk.Frame(parent); r1.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2)
        ttk.Label(r1, text="P sensor:").pack(side=tk.LEFT)
        ttk.Combobox(r1, state="readonly", width=8, values=["fluigent", "network"],
                     textvariable=self.var_pr_sensor).pack(side=tk.LEFT, padx=2)
        ttk.Label(r1, text="Ch:").pack(side=tk.LEFT, padx=(6, 2))
        ttk.Entry(r1, textvariable=self.var_pr_channel, width=3).pack(side=tk.LEFT)
        ttk.Button(r1, text="Find", width=5, command=self._find_pressure_sensor).pack(side=tk.LEFT, padx=4)
        ttk.Label(r1, text="Min/Max (mbar):").pack(side=tk.LEFT, padx=(4, 2))
        ttk.Entry(r1, textvariable=self.var_pr_min, width=5).pack(side=tk.LEFT)
        ttk.Entry(r1, textvariable=self.var_pr_limit, width=5).pack(side=tk.LEFT, padx=(2, 0))
        r1b = ttk.Frame(parent); r1b.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2)
        ttk.Label(r1b, text="Microsteps:").pack(side=tk.LEFT)
        ttk.Combobox(r1b, state="readonly", width=4, values=["1", "2", "4", "8", "16", "32"],
                     textvariable=self.var_pr_micro).pack(side=tk.LEFT, padx=2)
        ttk.Label(r1b, text="(driver setting; A4988 with MS1-3 high = 16. Ranges stay in full steps)")\
            .pack(side=tk.LEFT, padx=4)

        r2 = ttk.Frame(parent); r2.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2)
        self.lbl_pr_live = tk.Label(r2, textvariable=self.var_pr_live, anchor="w", wraplength=380, justify=tk.LEFT)
        self.lbl_pr_live.pack(side=tk.LEFT)
        self._pr_live_fg = self.lbl_pr_live.cget("fg")
        self._pr_last_txt: Optional[str] = None

        r3 = ttk.Frame(parent); r3.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2)
        ttk.Label(r3, text="Target (mbar):").pack(side=tk.LEFT)
        ent = ttk.Entry(r3, textvariable=self.var_pr_target, width=6)
        ent.pack(side=tk.LEFT, padx=4)
        ent.bind("<Return>", lambda _e: self._pr_hold())
        ttk.Button(r3, text="Hold", width=6, command=self._pr_hold).pack(side=tk.LEFT, padx=2)
        ttk.Button(r3, text="Release", width=8, command=self._pr_release).pack(side=tk.LEFT, padx=2)
        ttk.Button(r3, text="Stop", width=5, command=self._pr_idle).pack(side=tk.LEFT, padx=2)
        r4 = ttk.Frame(parent); r4.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2)
        ttk.Label(r4, text="Jog:").pack(side=tk.LEFT)
        ttk.Entry(r4, textvariable=self.var_pr_jog, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Combobox(r4, state="readonly", width=9, values=["full steps", "uL", "counts"],
                     textvariable=self.var_pr_jog_unit).pack(side=tk.LEFT, padx=2)
        ttk.Button(r4, text="Push (+P)", command=lambda: self._pr_jog(+1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(r4, text="Pull (-P)", command=lambda: self._pr_jog(-1)).pack(side=tk.LEFT, padx=2)
        r4b = ttk.Frame(parent); r4b.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2)
        ttk.Checkbutton(r4b, text="No limits", variable=self.var_pr_nolimits).pack(side=tk.LEFT)
        ttk.Label(r4b, text="Hold to move (v2), full steps/s:").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Entry(r4b, textvariable=self.var_pr_speed, width=5).pack(side=tk.LEFT)
        for text, d in (("Pull (hold)", -1), ("Push (hold)", +1)):
            b = ttk.Button(r4b, text=text)
            b.pack(side=tk.LEFT, padx=2)
            b.bind("<ButtonPress-1>", lambda _e, d=d: self._pr_hold_start(d))
            b.bind("<ButtonRelease-1>", lambda _e: self._pr_hold_stop())
        ttk.Button(r4b, text="Reset fault", command=self._pr_reset).pack(side=tk.LEFT, padx=(8, 2))
        r5 = ttk.Frame(parent); r5.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2)
        ttk.Button(r5, text="Zero sensor (vented)", command=self._pr_zero_sensor).pack(side=tk.LEFT, padx=2)
        ttk.Button(r5, text="Syringe emptied: pos 0", command=self._pr_set_empty).pack(side=tk.LEFT, padx=2)
        r6 = ttk.Frame(parent); r6.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(2, 6))
        ttk.Button(r6, text="Calibrate...", command=self.open_pressure_calibration, state=state)\
            .pack(side=tk.LEFT, padx=2)
        ttk.Label(r6, textvariable=self.var_pr_cal, wraplength=290, justify=tk.LEFT).pack(side=tk.LEFT, padx=6)
        self._cal_win = None
        self._cal_busy = False
        self._cal_result: Dict[str, Any] = {}

    def _find_pressure_sensor(self):
        try:
            sensors = fluigent_list_sensors()
        except Exception as e:
            messagebox.showinfo("Find pressure sensor", f"Could not list the Fluigent sensors: {e}\n\n"
                                "With the 'network' backend, flow_server.py prints its channels and units.")
            return
        lines = [f"channel {x['index']}: {x['unit']}" + (f", range {x['range'][0]:g}..{x['range'][1]:g}"
                                                         if x.get("range") else "") for x in sensors]
        mbar = [x for x in sensors if "bar" in str(x.get("unit", "")).lower()]
        if mbar:
            self.var_pr_channel.set(str(mbar[0]["index"]))
            lines.append(f"\nPressure channel set to {mbar[0]['index']}.")
        messagebox.showinfo("Fluigent sensors", "\n".join(lines) or "No Fluigent sensors found.")

    def _ctl_for_ui(self, allow_run: bool = False):
        s = self.active_stream
        ctl = getattr(s, "ctl", None) if s is not None else None
        if ctl is None:
            messagebox.showinfo("Pressure control", "No pressure control is running. Tick 'Control pressure', "
                                                    "then Connect Manual (or start a run).")
            return None
        if isinstance(s, PumpTestRunner) and not allow_run:
            messagebox.showinfo("Pressure control", "The run controls the pressure now. Stop the run first.")
            return None
        return ctl

    def _pressure_bg(self, label: str, fn):
        """Run a blocking controller action off the Tk thread; report through the UI queue."""
        def work():
            try:
                res = fn()
                self.ui_events.put(("status", f"{label}: {res}" if isinstance(res, str) else f"{label}: done"))
            except Exception as e:
                self.ui_events.put(("error", f"{label} failed: {e}"))
        threading.Thread(target=work, name="pressure-action", daemon=True).start()

    def _pr_hold(self):
        ctl = self._ctl_for_ui()
        if ctl is not None:
            try:
                ctl.set_target(safe_float(self.var_pr_target.get().replace(",", "."), 0.0))
            except Exception as e:
                messagebox.showerror("Hold pressure", str(e))

    def _pr_release(self):
        ctl = self._ctl_for_ui()
        if ctl is not None:
            ctl.release()

    def _pr_idle(self):
        ctl = self._ctl_for_ui()
        if ctl is not None:
            ctl.idle()

    def _pr_reset(self):
        ctl = self._ctl_for_ui()
        if ctl is not None:
            ctl.reset_fault()

    def _pr_jog(self, sign: int):
        ctl = self._ctl_for_ui()
        if ctl is None:
            return
        value = abs(safe_float(self.var_pr_jog.get().replace(",", "."), 0.0))
        unit = self.var_pr_jog_unit.get()
        if unit == "full steps":
            counts = value * ctl.m
        elif unit == "uL":
            ulpc = ctl.cfg.get("ul_per_step") or (1000.0 / 2470 / ctl.m)    # per motor count
            counts = value / float(ulpc)
        else:
            counts = value
        steps = sign * int(round(counts))
        if steps:
            free = bool(self.var_pr_nolimits.get())
            self._pressure_bg(f"Jog {steps:+d} counts{' (no limits)' if free else ''}",
                              lambda: (ctl.jog(steps, unlimited=free), f"position {ctl.syr.position()}")[1])

    def _pr_hold_start(self, d: int):
        """Hold-to-move: run while the button is pressed (v2 firmware), resent every 0.5 s
        because the firmware stops by itself 2 s after the last command."""
        ctl = self._ctl_for_ui()
        if ctl is None:
            return
        sps = d * abs(safe_float(self.var_pr_speed.get(), 20.0)) * ctl.m
        try:
            ctl.manual_velocity(sps)
        except Exception as e:
            messagebox.showerror("Hold to move", str(e))
            return
        self._pr_hold_dir = d

        def keepalive():
            if self._pr_hold_dir == d and getattr(self.active_stream, "ctl", None) is ctl:
                try:
                    ctl.manual_velocity(sps)
                except Exception:
                    self._pr_hold_dir = 0
                    return
                self.root.after(500, keepalive)
        self.root.after(500, keepalive)

    def _pr_hold_stop(self):
        if not self._pr_hold_dir:
            return
        self._pr_hold_dir = 0
        ctl = getattr(self.active_stream, "ctl", None)
        if ctl is not None:
            try:
                ctl.manual_velocity(0)
            except Exception:
                log_exception("Hold to move: stop failed")

    def _pr_zero_sensor(self):
        ctl = self._ctl_for_ui()
        if ctl is None or not messagebox.askokcancel(
                "Zero pressure sensor", "Open the line to atmosphere first (pump off, nothing pushing).\n\n"
                                        "The mean reading over 3 s becomes the zero. Zero now?"):
            return

        def work():
            off = ctl.zero_sensor(3.0)
            self._save_cal_values({"sensor_offset_mbar": round(off, 4)})
            return f"offset {off:+.2f} mbar (saved)"
        self._pressure_bg("Zero pressure sensor", work)

    def _pr_set_empty(self):
        ctl = self._ctl_for_ui()
        if ctl is None or not messagebox.askokcancel(
                "Syringe emptied", "Is the plunger at the empty end (fully in)?\n\nThe syringe position "
                                   "becomes 0; the controller only pulls from there."):
            return
        self._pressure_bg("Syringe position", lambda: (ctl.set_position(0), "set to 0")[1])

    def _save_cal_values(self, values: Dict[str, Any]):
        path = pressure_cal_path()
        cal = pc.load_calibration(path)
        cal.update(values)
        cal["saved_at"] = iso_now()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pc.save_calibration(path, cal)
        self.ui_events.put(("cal_saved", None))

    def _update_pressure_ui(self):
        s = self.active_stream
        st = s.get_pressure_status() if s is not None else None
        if st is None:
            if self._pr_last_txt:
                self.var_pr_live.set("Stopped. Last: " + self._pr_last_txt)
                self.lbl_pr_live.configure(fg=self._pr_live_fg)
                self._pr_last_txt = None
            return
        p, p1 = st["p"], st["p_1s"]
        txt = "no reading" if p != p else f"P {p:.1f} mbar (1 s mean {p1:.1f})"
        if st["mode"] == "hold":
            txt += f" -> {st['target']:g} mbar, " + ("settled" if st["settled"] else "settling")
        else:
            txt += f" · {st['mode']}"
        m = st.get("microsteps") or 1
        pos_txt = f"{st['pos']} ({st['pos'] / m:.0f} full steps)" if m > 1 else f"{st['pos']}"
        txt += (f"\nSyringe {st['protocol']}: position {pos_txt}, {st['remaining_pull']} steps to full"
                f"{' · push blocked (empty end)' if st.get('blocked') == 'empty' else ''}"
                f"{' · below target: waiting for the pump' if st.get('blocked') == 'pump_below' else ''}"
                f"\nGain push {st['gain_push_mbar_step']:.2f} / pull {st['gain_pull_mbar_step']:.2f} mbar/step · "
                f"backlash {st['backlash_steps']:.0f} · drift {st['drift_mbar_s']:+.1f} mbar/s"
                f"\nAllowed {st['p_min']:g}..{st['p_max']:.0f} mbar (max: {st['limit_source']})")
        res = st.get("resolution_mbar")
        if res and res > 3 * safe_float(self._pressure_cfg().get("tolerance_mbar", 2.0), 2.0):
            txt += (f"\nCoarse: one step = {res:.0f} mbar, holds saw-tooth +-{res / 2:.0f} mbar "
                    f"(microstepping or an air pocket would help)")
        if st["mode"] == "fault":
            txt = f"FAULT: {st['fault']}\n" + txt
        self.var_pr_live.set(txt)
        self._pr_last_txt = txt
        self.lbl_pr_live.configure(fg="#b00000" if st["mode"] == "fault" or st["stale"] else self._pr_live_fg)

    def open_pressure_calibration(self):
        if self._cal_win is not None and self._cal_win.winfo_exists():
            self._cal_win.lift()
            return
        w = tk.Toplevel(self.root)
        w.title("Pressure calibration")
        self._cal_win = w
        self.var_cal_top = tk.StringVar(value="60")
        intro = ("Needs a Manual session with 'Control pressure' ticked and the pump output OFF. "
                 "Results apply to the running controller at once; 'Save' keeps them for later sessions "
                 f"({pressure_cal_path()}).")
        ttk.Label(w, text=intro, wraplength=560, justify=tk.LEFT).pack(side=tk.TOP, anchor="w", padx=10, pady=(8, 4))
        steps = [
            ("zero", "1. Zero sensor", "Line open to atmosphere: the mean reading becomes 0 mbar."),
            ("pos0", "2. Syringe emptied: pos 0", "Plunger fully in. Then give it room to push: Output ON "
                                                  "(~400 Hz), Hold 0 mbar for 20-30 s, Release, Output OFF, Stop."),
            ("sweep", "3. Gain + backlash sweep", "Pump OFF, line liquid-filled: takes up the slack 1-4 steps at "
                                                  "a time, pushes up to the span, then back to the start (never "
                                                  "below it: sucking pulls liquid through the pump). Valveless "
                                                  "pump: clamp its inlet tube for the sweep. Optional: holds "
                                                  "measure the gain themselves."),
            ("leak", "4. Leak test (60 s)", "Brings the line to about 2/3 of the span, stops the syringe and "
                                            "fits the decay."),
            ("step", "5. Step response", "Holds 1/4, 1/2 and 1 x the span, reports settling time, then releases."),
        ]
        for key, label, hint in steps:
            row = ttk.Frame(w); row.pack(side=tk.TOP, fill=tk.X, padx=10, pady=2)
            ttk.Button(row, text=label, width=24, command=lambda k=key: self._cal_step(k)).pack(side=tk.LEFT)
            ttk.Label(row, text=hint, wraplength=380, justify=tk.LEFT).pack(side=tk.LEFT, padx=6)
        row = ttk.Frame(w); row.pack(side=tk.TOP, fill=tk.X, padx=10, pady=4)
        ttk.Label(row, text="Span (mbar):").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.var_cal_top, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Save calibration", command=self._cal_save).pack(side=tk.RIGHT, padx=2)
        self.txt_cal = tk.Text(w, height=12, width=78)
        self.txt_cal.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(4, 10))

    def _cal_log(self, msg: str):
        if self._cal_win is not None and self._cal_win.winfo_exists():
            self.txt_cal.insert(tk.END, msg + "\n")
            self.txt_cal.see(tk.END)

    def _cal_step(self, key: str):
        s = self.active_stream
        if not isinstance(s, ManualSession):
            messagebox.showinfo("Calibration", "Connect a Manual session with 'Control pressure' ticked first.",
                                parent=self._cal_win)
            return
        ctl = self._ctl_for_ui()
        if ctl is None:
            return
        if s._pump_hz() > 0:
            messagebox.showinfo("Calibration", "Switch the pump output OFF first.", parent=self._cal_win)
            return
        if self._cal_busy:
            messagebox.showinfo("Calibration", "A calibration step is still running.", parent=self._cal_win)
            return
        span = max(5.0, safe_float(self.var_cal_top.get(), 60.0))
        log = lambda m: self.ui_events.put(("cal_log", m))

        def work():
            self._cal_busy = True
            try:
                if key == "zero":
                    off = ctl.zero_sensor(3.0)
                    self._cal_result["sensor_offset_mbar"] = round(off, 4)
                    log(f"Sensor zero: offset {off:+.3f} mbar")
                elif key == "pos0":
                    ctl.set_position(0)
                    log("Syringe position set to 0 (empty end)")
                elif key == "sweep":
                    log(f"Sweep to {span:g} mbar ...")
                    r = pc.calib_gain_sweep(ctl, p_top=span, progress=log)
                    ctl.apply_calibration(r)
                    self._cal_result.update({k: v for k, v in r.items() if k != "points"})
                    self._cal_result["sweep_points"] = r["points"]
                    bl = (f"{r['backlash_steps']:g}" if r.get("backlash_steps") is not None
                          else f"at least {r['backlash_min_steps']:g}" if r.get("backlash_min_steps") else "?")
                    log(f"Gain push {r.get('gain_push_mbar_step')} / pull {r.get('gain_pull_mbar_step')} mbar/step, "
                        f"backlash {bl} steps, slack before the first push {r.get('slack_push_steps')} "
                        f"({len(r['points'])} moves)")
                    g = max(x for x in (r.get("gain_push_mbar_step"), r.get("gain_pull_mbar_step"), 0) if x)
                    if g > 3 * safe_float(self._pressure_cfg().get("tolerance_mbar", 2.0), 2.0):
                        log(f"NOTE: one step changes the pressure by ~{g:.0f} mbar, so a hold saw-tooths about "
                            f"+-{g / 2:.0f} mbar around the target. For finer holds: microstepping on the "
                            f"driver, or a small air pocket in the line.")
                elif key == "leak":
                    r = pc.calib_leak_test(ctl, p_test=round(2 * span / 3), seconds=60.0, progress=log)
                    self._cal_result.update(r)
                    log(f"Leak: {r['leak_mbar_s']:+.4f} mbar/s from {r['leak_p_start_mbar']:.1f} mbar")
                elif key == "step":
                    r = pc.calib_step_response(ctl, targets=(round(span / 4), round(span / 2), span), progress=log)
                    self._cal_result.update(r)
                    for x in r["step_response"]:
                        log(f"  {x['target']:g} mbar: {'settled' if x['settled'] else 'NOT settled'} in "
                            f"{x['t_settle_s']} s, overshoot {x['overshoot_mbar']} mbar")
            except Exception as e:
                log(f"FAILED: {e}")
            finally:
                self._cal_busy = False
        threading.Thread(target=work, name="pressure-calibration", daemon=True).start()

    def _cal_save(self):
        if not self._cal_result:
            messagebox.showinfo("Calibration", "Nothing measured yet.", parent=self._cal_win)
            return
        vals = dict(self._cal_result)
        vals["port"] = self.var_pr_port.get().strip()
        vals["microsteps"] = max(1, safe_int(self.var_pr_micro.get(), 1))   # gains/backlash are per motor count
        try:
            self._save_cal_values(vals)
            self._cal_log(f"Saved to {pressure_cal_path()}")
        except Exception as e:
            messagebox.showerror("Calibration", f"Could not save: {e}", parent=self._cal_win)

    def _active_runner(self) -> Optional["PumpTestRunner"]:
        s = self.active_stream
        return s if isinstance(s, PumpTestRunner) else None

    def _tick_progress(self):
        try:
            self._update_progress_ui()
        except Exception:
            log_exception_limited("progress_tick", "Run timer: progress update failed")
        try:
            self._update_power_ui()
        except Exception:
            log_exception_limited("power_tick", "Power meter: live update failed")
        try:
            self._update_pressure_ui()
        except Exception:
            log_exception_limited("pressure_tick", "Pressure control: live update failed")
        try:
            self._update_status_bar()
        except Exception:
            log_exception_limited("status_bar", "Status bar update failed")
        self.root.after(1000, self._tick_progress)

    def _update_progress_ui(self):
        now = datetime.now()
        runner = self._active_runner()
        lead_s = 60.0 * safe_float(self._supply_cfg().get("lead_min", 10), 10.0)

        if runner is None:
            self._show_voltage_banner(None)
            self._show_operator_banner(None)
            tests = self.cfg.get("tests", [])
            if not tests:
                self.var_progress.set("")
                self.var_next_change.set("")
                self.pbar["value"] = 0
                return
            total, changes, _ = plan_schedule(tests, safe_float(self.var_break_between.get(), 0.0))
            proj, shift = project_changes(changes, now, 0.0, 0)
            self.pbar["value"] = 0
            if math.isinf(total):
                self.var_progress.set("Planned: runs until Stop (endless group)")
            else:
                extra = " + your confirmations" if changes else ""
                self.var_progress.set(f"Planned: {fmt_duration(total + shift)}{extra} · "
                                      f"ends ~{fmt_clock(now + timedelta(seconds=total + shift))} if started now")
            self.var_next_change.set(f"{len(changes)} voltage change(s) planned" if changes else "")
            self.lbl_next_change.configure(fg=self._next_change_fg)
            return

        p = runner.get_progress()
        total, nominal, elapsed = p["total_s"], p["nominal_done_s"], p["elapsed_s"]
        proj, shift = project_changes(runner.plan_changes, now, nominal, p["changes_done"])
        req = p["voltage_request"]

        if p["finished"]:
            self.var_progress.set(f"Finished · ran {fmt_duration(elapsed)}")
            self.pbar["value"] = 1000 if not math.isinf(total) else 0
        elif math.isinf(total):
            self.var_progress.set(f"Elapsed {fmt_duration(elapsed)} · runs until Stop")
            self.pbar["value"] = 0
        else:
            remaining = max(0.0, total - nominal) + shift
            extra = " + your confirmations" if proj else ""
            frac = min(1.0, nominal / total) if total > 0 else 1.0
            self.var_progress.set(f"Elapsed {fmt_duration(elapsed)} · Remaining {fmt_duration(remaining)}{extra} · "
                                  f"Ends ~{fmt_clock(now + timedelta(seconds=remaining))} · {100 * frac:.0f}%")
            self.pbar["value"] = int(1000 * frac)

        if req:
            self.var_next_change.set(f"Waiting for you: voltage change #{req['index']}")
            self.lbl_next_change.configure(fg="#b00000")
        elif proj and not p["finished"]:
            c, due, _moved = proj[0]
            until = (due - now).total_seconds()
            self.var_next_change.set(f"Next voltage change #{c['index']} → {c['mv'] / 1000:.3f} V "
                                     f"at {fmt_clock(due)} (in {fmt_duration(until)})")
            self.lbl_next_change.configure(fg="#b00000" if until <= lead_s else self._next_change_fg)
        else:
            self.var_next_change.set("")
            self.lbl_next_change.configure(fg=self._next_change_fg)

        self._show_voltage_banner(req)
        self._show_operator_banner(p.get("operator_request"))
        if (req or p.get("operator_request")) and time.monotonic() - self._last_bell >= 60.0:
            self._last_bell = time.monotonic()
            self.root.bell()

        if self.run_dir and time.monotonic() - self._last_status_json >= 10.0:
            self._last_status_json = time.monotonic()
            self._write_status_json(self.run_dir, now, p, proj, shift)

    def _show_voltage_banner(self, req: Optional[Dict[str, Any]]):
        if not req:
            if self._vprompt_index is not None:
                self.voltage_banner.pack_forget()
                self._vprompt_index = None
            return
        waited = CLOCK() - req["asked_mono"]
        pump = (f"The pump keeps running at {req['freq_hz']} Hz" if req["hold"]
                else "The pump is at 0 Hz")
        self.var_vprompt_info.set(
            f"Bench supply current limit: {safe_float(req.get('limit_ma') or 0, 0.0):g} mA. {pump} until you confirm. "
            f"Asked at {req['asked_at'][11:16]} ({fmt_duration(waited)} ago)."
            + (f"\nNote: {req['note']}" if req.get("note") else ""))
        if self._vprompt_index == req["index"]:
            return
        # New request: show the banner, prefill the requested value, get attention
        self._vprompt_index = req["index"]
        self.var_vprompt_title.set(f"VOLTAGE CHANGE #{req['index']}: set the bench supply to {req['mv'] / 1000:.3f} V")
        self.var_vprompt_read.set(f"{req['mv'] / 1000:.3f}")
        self.voltage_banner.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 6), after=self._progress_row)
        self._last_bell = time.monotonic()
        self.root.bell()
        try:
            self.root.deiconify()
            self.root.lift()
            self.ent_vprompt.focus_set()
            self.ent_vprompt.select_range(0, tk.END)
        except Exception:
            log_exception_limited("vprompt_focus", "Voltage prompt: could not raise the window")

    def _show_operator_banner(self, req: Optional[Dict[str, Any]]):
        if not req:
            if self._oprompt_index is not None:
                self.operator_banner.pack_forget()
                self._oprompt_index = None
            return
        waited = CLOCK() - req["asked_mono"]
        self.var_oprompt_info.set(f"{req['text']}\nAsked at {req['asked_at'][11:16]} ({fmt_duration(waited)} ago).")
        if self._oprompt_index == req["index"]:
            return
        self._oprompt_index = req["index"]
        self.var_oprompt_title.set(f"WAITING FOR YOU: {req['title']}")
        self.operator_banner.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 6), after=self._progress_row)
        self._last_bell = time.monotonic()
        self.root.bell()
        try:
            self.root.deiconify()
            self.root.lift()
        except Exception:
            log_exception_limited("oprompt_raise", "Operator prompt: could not raise the window")

    def confirm_operator_prompt(self):
        runner = self._active_runner()
        if runner is None or not runner.confirm_operator():
            self._show_operator_banner(None)
            return
        self.var_status.set("Confirmed; continuing ...")

    def confirm_voltage_change(self):
        runner = self._active_runner()
        req = runner.get_progress()["voltage_request"] if runner else None
        if not req:
            return
        try:
            read_mv = float(self.var_vprompt_read.get().strip().replace(",", ".")) * 1000.0
        except ValueError:
            messagebox.showerror("Voltage", "Type the voltage shown on the supply, in volts (e.g. 2.0).")
            return
        max_mv = safe_float(req.get("max_mv") or SUPPLY_DEFAULTS["max_mv"], SUPPLY_DEFAULTS["max_mv"])
        if read_mv > max_mv:
            messagebox.showerror("Voltage too high",
                                 f"{read_mv / 1000:.3f} V is above the {max_mv / 1000:.3f} V maximum for the pump "
                                 f"(supply.max_mv in the config). Lower the supply first.")
            return
        if abs(read_mv - req["mv"]) > 50.0 and not messagebox.askyesno(
                "Voltage differs",
                f"This step asks for {req['mv'] / 1000:.3f} V, but you entered {read_mv / 1000:.3f} V.\n\n"
                f"Log {read_mv / 1000:.3f} V and continue?"):
            return
        if runner.confirm_voltage(read_mv):
            self._show_voltage_banner(None)

    def _write_status_json(self, run_dir: str, now: datetime, p: Dict[str, Any],
                           proj: List[Tuple[Dict[str, Any], datetime, bool]], shift: float):
        """Heartbeat file so a long run can be checked from another PC (atomic replace)."""
        total, nominal = p["total_s"], p["nominal_done_s"]
        remaining = None if math.isinf(total) else max(0.0, total - nominal) + shift
        req = p["voltage_request"]
        doc = {
            "updated": now.isoformat(timespec="seconds"),
            "status": self.var_status.get(),
            "finished": p["finished"],
            "elapsed_s": round(p["elapsed_s"], 1),
            "nominal_done_s": round(nominal, 1),
            "nominal_total_s": None if math.isinf(total) else round(total, 1),
            "remaining_s": None if remaining is None else round(remaining, 1),
            "earliest_end": None if remaining is None else (now + timedelta(seconds=remaining)).isoformat(timespec="seconds"),
            "v_set_mv": p["v_set_mv"],
            "waiting_for_voltage": None if not req else {"index": req["index"], "mv": req["mv"], "asked_at": req["asked_at"]},
            "next_voltage_changes": [{"index": c["index"], "mv": c["mv"], "due": due.isoformat(timespec="seconds")}
                                     for c, due, _ in proj[:10]],
            "power": self.active_stream.get_power_live() if self.active_stream is not None else None,
        }
        path = os.path.join(run_dir, "status.json")
        try:
            with open(path + ".tmp", "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2)
            os.replace(path + ".tmp", path)
        except Exception:
            log_exception_limited("status_json", f"Could not write {path}")

    def _schedule_text(self) -> str:
        now = datetime.now()
        runner = self._active_runner()
        if runner is not None:
            p = runner.get_progress()
            proj, shift = project_changes(runner.plan_changes, now, p["nominal_done_s"], p["changes_done"])
            return format_schedule(runner.plan_total_s, proj, shift, now, p["nominal_done_s"],
                                   runner.supply_cfg, False, "Voltage schedule (live, from now)")
        total, changes, truncated = plan_schedule(self.cfg.get("tests", []),
                                                  safe_float(self.var_break_between.get(), 0.0))
        proj, shift = project_changes(changes, now, 0.0, 0)
        return format_schedule(total, proj, shift, now, 0.0, self._supply_cfg(), truncated,
                               "Voltage schedule (if the run starts now)")

    def show_voltage_schedule(self):
        win = tk.Toplevel(self.root)
        win.title("Voltage schedule")
        win.geometry("760x480")
        txt = tk.Text(win, wrap=tk.NONE, font=("Consolas", 10))
        txt.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)

        def fill():
            txt.configure(state=tk.NORMAL)
            txt.delete("1.0", tk.END)
            txt.insert("1.0", self._schedule_text())
            txt.configure(state=tk.DISABLED)

        def copy_all():
            self.root.clipboard_clear()
            self.root.clipboard_append(txt.get("1.0", tk.END))

        btns = ttk.Frame(win)
        btns.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 6))
        ttk.Button(btns, text="Refresh", command=fill).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Copy", command=copy_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(btns, text="Close", command=win.destroy).pack(side=tk.RIGHT, padx=2)
        fill()

    # ---- video controls ----
    def _build_video_controls(self, parent):
        """Build video recording controls."""
        video_frame = ttk.Frame(parent)
        self.video_controls_frame = video_frame
        
        row1 = ttk.Frame(video_frame)
        row1.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        
        ttk.Checkbutton(row1, text="Enable video recording", variable=self.var_video_enabled)\
            .pack(side=tk.LEFT, padx=(0, 8))
        
        ttk.Label(row1, text="Camera index:").pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.var_video_camera_index, width=6).pack(side=tk.LEFT, padx=4)
        
        row1b = ttk.Frame(video_frame)
        row1b.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 4))
        self.btn_video_connect = ttk.Button(row1b, text="Connect Camera", command=self._connect_camera)
        self.btn_video_connect.pack(side=tk.LEFT)
        
        self.btn_video_disconnect = ttk.Button(row1b, text="Disconnect", command=self._disconnect_camera, state=tk.DISABLED)
        self.btn_video_disconnect.pack(side=tk.LEFT, padx=4)
        ttk.Label(row1b, text="(index 'dummy' = test camera)").pack(side=tk.LEFT, padx=8)
        
        row_opts = ttk.Frame(video_frame)
        row_opts.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 4))
        self.var_video_overlay = tk.BooleanVar(value=bool(VIDEO_DEFAULTS["overlay"]))
        self.var_video_codec = tk.StringVar(value=str(VIDEO_DEFAULTS["codec"]))
        ttk.Checkbutton(row_opts, text="Overlay (time/freq/flow, top-left)", variable=self.var_video_overlay)\
            .pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(row_opts, text="Codec:").pack(side=tk.LEFT)
        ttk.Combobox(row_opts, state="readonly", width=6, values=list(VIDEO_CODECS),
                     textvariable=self.var_video_codec).pack(side=tk.LEFT, padx=4)

        row2 = ttk.Frame(video_frame)
        row2.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 4))

        ttk.Label(row2, textvariable=self.var_video_status).pack(side=tk.LEFT)
        ttk.Label(row2, text="  |  Recording:").pack(side=tk.LEFT)

        # Store video label for updating recording status
        self.var_video_rec_status = tk.StringVar(value="off")
        self.lbl_video_rec = ttk.Label(row2, textvariable=self.var_video_rec_status)
        self.lbl_video_rec.pack(side=tk.LEFT, padx=4)
        self.root.after(1000, self._tick_video_status)
        return video_frame

    def _tick_video_status(self):
        """Once a second: camera frame rate, frames written and dropped."""
        try:
            vr = self.video_recorder
            if vr is not None and vr.is_opened():
                fps = vr.measured_fps()
                fps_txt = f"{fps:.2f} fps" if fps else "measuring fps"
                if vr.is_recording():
                    self.var_video_rec_status.set(f"on · {vr.frames_written} frames · {vr.dropped_total} dropped "
                                                  f"· camera {fps_txt} (header {vr.rec_fps:.2f})")
                else:
                    self.var_video_rec_status.set(f"off · camera {fps_txt}")
        except Exception:
            log_exception_limited("video_status", "Video status update failed")
        self.root.after(1000, self._tick_video_status)

    def _camera_index(self):
        """Camera index from the panel: an integer, or "dummy" for the test camera."""
        raw = self.var_video_camera_index.get().strip()
        return "dummy" if raw.lower() == "dummy" else safe_int(raw, 0)

    def _drive_hz(self) -> float:
        """Current drive frequency (moves the dummy camera's meniscus)."""
        s = self.active_stream
        try:
            return float(s.overlay_info()[1]) if s is not None else 0.0
        except Exception:
            return 0.0

    def _init_video(self):
        """Initialize video recorder."""
        if not CV2_AVAILABLE:
            self.var_video_status.set("Camera: OpenCV not available (pip install opencv-python)")
            log_warning("Video init: OpenCV not available; camera disabled.")
            return
        
        # Create video recorder instance
        self.video_recorder = VideoRecorder()
        
        # Try to auto-connect to camera 0
        camera_idx = self._camera_index()
        if self.video_recorder.open(camera_idx, hz_cb=self._drive_hz):
            self.var_video_status.set(f"Camera: Connected (index {camera_idx})")
            self.btn_video_connect.config(state=tk.DISABLED)
            self.btn_video_disconnect.config(state=tk.NORMAL)
            # Start video preview
            self._start_video_preview()
        else:
            self.var_video_status.set("Camera: Not connected (click Connect to try)")
            self.btn_video_connect.config(state=tk.NORMAL)
            self.btn_video_disconnect.config(state=tk.DISABLED)
            log_warning(f"Video init: could not open camera index {camera_idx}")

    def _connect_camera(self):
        """Connect to USB camera."""
        if not CV2_AVAILABLE:
            messagebox.showwarning("Camera", "OpenCV is not installed. Run: pip install opencv-python")
            log_warning("Connect camera: OpenCV not installed.")
            return
        
        if self.video_recorder is None:
            self.video_recorder = VideoRecorder()
        
        camera_idx = self._camera_index()
        if self.video_recorder.open(camera_idx, hz_cb=self._drive_hz):
            self.var_video_status.set(f"Camera: Connected (index {camera_idx})")
            self.btn_video_connect.config(state=tk.DISABLED)
            self.btn_video_disconnect.config(state=tk.NORMAL)
            # Start video preview
            self._start_video_preview()
        else:
            self.var_video_status.set(f"Camera: Failed to connect (index {camera_idx})")
            log_error(f"Connect camera: failed to open camera index {camera_idx}")
            messagebox.showerror("Camera", f"Could not open camera at index {camera_idx}. Check USB connection.")

    def _disconnect_camera(self):
        """Disconnect from USB camera."""
        # Stop video preview first
        self._stop_video_preview()
        
        if self.video_recorder:
            # Stop recording if active
            if self.video_recorder.is_recording():
                self.video_recorder.stop_recording()
                self.var_video_rec_status.set("off")
            
            self.video_recorder.close()
            self.video_recorder = None
        
        self.var_video_status.set("Camera: Disconnected")
        self.btn_video_connect.config(state=tk.NORMAL)
        self.btn_video_disconnect.config(state=tk.DISABLED)
        
        # Reset preview label
        self.video_preview_label.configure(image="", text="Camera not connected\nConnect camera to see preview")

    def _video_cfg(self) -> Dict[str, Any]:
        """Video settings: the config's "video" section with the panel's overlay/codec."""
        v = {**VIDEO_DEFAULTS, **(self.cfg.get("video") or {})}
        if hasattr(self, "var_video_overlay"):
            v["overlay"] = bool(self.var_video_overlay.get())
            v["codec"] = self.var_video_codec.get().strip() or v["codec"]
        return v

    def _start_video_recording(self, stream: Optional[LiveSessionBase] = None):
        """Start recording on the session's clock (t0), into the run folder."""
        if not self.var_video_enabled.get():
            return False

        if not self.video_recorder or not self.video_recorder.is_opened():
            # Camera not connected, disable video for this run
            self.var_video_enabled.set(False)
            log_warning("Video recording: camera not connected; disabling video for this run.")
            return False

        if self.video_recorder.is_recording():
            return True

        if not self.run_dir:
            out_dir = ensure_dir(self.var_out_dir.get().strip())
            run_name = sanitize_run_name(self.var_test_name.get(), "video")
            self.run_dir = create_unique_run_dir(out_dir, run_name, "_VIDEO")

        t0 = getattr(stream, "t0", None)
        info_cb = stream.overlay_info if stream is not None else None
        if self.video_recorder.start_recording(self.run_dir, t0 if t0 is not None else CLOCK(),
                                               self._video_cfg(), info_cb):
            self.var_video_rec_status.set("recording")
            return True
        else:
            log_error(f"Video recording: failed to start recording in {self.run_dir}")
            messagebox.showerror("Video", "Failed to start video recording")
            return False

    def _stop_video_recording(self):
        """Stop video recording."""
        if self.video_recorder and self.video_recorder.is_recording():
            self.video_recorder.stop_recording()
            self.var_video_rec_status.set("off")

    def _build_video_preview(self, parent):
        """Build video preview panel to show real-time camera feed."""
        self.video_preview_frame = ttk.Frame(parent)

        topbar = ttk.Frame(self.video_preview_frame)
        topbar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(4, 0))
        self.btn_video_popout = ttk.Button(topbar, text="Pop Out", command=self._toggle_video_popout)
        self.btn_video_popout.pack(side=tk.RIGHT)

        # Placeholder for video preview
        self.video_preview_label = ttk.Label(self.video_preview_frame, text="Camera not connected\nConnect camera to see preview", anchor="center")
        self.video_preview_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Pop-out window (created on demand, e.g. for a secondary screen)
        self.video_popout = None
        self.video_popout_label = None

        # Start video preview update
        self._video_preview_running = False
        return self.video_preview_frame

    def _toggle_video_popout(self):
        """Open or close the resizable pop-out camera window."""
        if self.video_popout is not None and self.video_popout.winfo_exists():
            self._close_video_popout()
            return

        win = tk.Toplevel(self.root)
        win.title("Camera View")
        win.geometry("800x620")
        win.minsize(320, 240)
        win.configure(bg="black")
        win.protocol("WM_DELETE_WINDOW", self._close_video_popout)

        label = tk.Label(win, bg="black", fg="white", anchor="center",
                         text="Camera not connected\nConnect camera to see the feed")
        label.pack(fill=tk.BOTH, expand=True)

        self.video_popout = win
        self.video_popout_label = label
        self.btn_video_popout.config(text="Close Pop-out")

    def _close_video_popout(self):
        """Close the pop-out camera window if open."""
        if self.video_popout is not None:
            try:
                self.video_popout.destroy()
            except Exception:
                pass
        self.video_popout = None
        self.video_popout_label = None
        if hasattr(self, "btn_video_popout"):
            self.btn_video_popout.config(text="Pop Out")

    def _start_video_preview(self):
        """Start updating the video preview in the UI."""
        if self._video_preview_running:
            return
        
        self._video_preview_running = True
        self._update_video_preview()

    def _stop_video_preview(self):
        """Stop updating the video preview in the UI."""
        self._video_preview_running = False
        if hasattr(self, "video_preview_label"):
            self.video_preview_label.image = None
        if getattr(self, "video_popout_label", None) is not None and self.video_popout.winfo_exists():
            self.video_popout_label.configure(image="", text="Camera not connected\nConnect camera to see the feed")
            self.video_popout_label.image = None

    def _update_video_preview(self):
        """Update the video preview with the latest camera frame."""
        if not self._video_preview_running:
            return
        
        if self.video_recorder and self.video_recorder.is_opened():
            # Latest frame from the capture thread (only that thread reads the camera)
            frame = self.video_recorder.read_frame()
            
            if frame is not None:
                # Convert frame to RGB for display
                try:
                    from PIL import Image, ImageTk
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w = frame_rgb.shape[:2]

                    # Resize frame to fit in preview (max 240px height)
                    max_size = 240
                    scale = min(max_size / h, max_size / w)
                    new_h = int(h * scale)
                    new_w = int(w * scale)
                    small = cv2.resize(frame_rgb, (new_w, new_h))

                    # Convert to PIL Image and then to PhotoImage
                    pil_image = Image.fromarray(small)
                    photo = ImageTk.PhotoImage(pil_image)

                    self.video_preview_label.configure(image=photo, text="")
                    self.video_preview_label.image = photo  # Keep reference to prevent GC

                    # Feed the pop-out window at its own (larger) size
                    if self.video_popout_label is not None:
                        if not self.video_popout.winfo_exists():
                            self._close_video_popout()
                        else:
                            pw = self.video_popout_label.winfo_width()
                            ph = self.video_popout_label.winfo_height()
                            if pw > 20 and ph > 20:
                                pscale = min(pw / w, ph / h)
                                big = cv2.resize(frame_rgb, (max(int(w * pscale), 1), max(int(h * pscale), 1)))
                                pphoto = ImageTk.PhotoImage(Image.fromarray(big))
                                self.video_popout_label.configure(image=pphoto, text="")
                                self.video_popout_label.image = pphoto
                except ImportError:
                    self.video_preview_label.configure(text="PIL not available for preview")
                    log_warning_limited("video_preview_pil", "Video preview: PIL not available; preview disabled.")
                except Exception:
                    log_exception_limited("video_preview_error", "Video preview: failed to update frame")
        
        # Schedule next update (~15 fps). Lower than the recording rate on purpose:
        # the preview rebuilds a PhotoImage each tick, so a slower cadence roughly
        # halves that allocation churn without affecting the recorded video.
        self.root.after(66, self._update_video_preview)

    def _apply_ui_scaling(self):
        """Scale UI based on screen size to avoid scroll-only layouts."""
        try:
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            base_w, base_h = 1250, 840
            scale = min(sw / base_w, sh / base_h)
            scale = clamp(scale, 0.85, 1.15)
            self.root.tk.call("tk", "scaling", scale)
        except Exception:
            log_exception("UI scaling: failed to apply scaling")

    def _set_default_geometry(self):
        """Set a sensible starting size based on screen resolution."""
        try:
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            w = min(1700, int(sw * 0.95))
            h = min(1000, int(sh * 0.92))
            x = max(0, (sw - w) // 2)
            y = max(0, (sh - h) // 2)
            self.root.geometry(f"{w}x{h}+{x}+{y}")
            self.root.minsize(900, 640)
        except Exception:
            log_exception("UI geometry: failed to set default geometry")
            self.root.geometry("1250x840")

    def _build_config_builder(self, parent):
        listf = ttk.LabelFrame(parent, text="Test blocks")
        listf.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # exportselection=False: otherwise selecting text in an editor Entry
        # silently clears the list selection and "Apply changes" has no target.
        self.lst_blocks = tk.Listbox(listf, width=62, height=16, selectmode=tk.EXTENDED, exportselection=False)
        self.lst_blocks.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.lst_blocks.bind("<<ListboxSelect>>", self._on_block_select)
        self.lst_blocks.bind("<Escape>", lambda _e: self.lst_blocks.selection_clear(0, tk.END))
        # Maps listbox row -> path into cfg["tests"]: (i,) for a top-level
        # block or group header, (i, j) for child j inside group i.
        self._row_paths: List[tuple] = []

        actions = ttk.Frame(listf)
        actions.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 6))
        ttk.Button(actions, text="Add Constant", command=self.add_constant_block).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions, text="Add Sweep", command=self.add_sweep_block).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions, text="Add Group", command=self.add_group_block).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions, text="Add Voltage Step", command=self.add_voltage_block).pack(side=tk.LEFT, padx=2)
        edit_row = ttk.Frame(listf)
        edit_row.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 6))
        ttk.Button(edit_row, text="Duplicate", command=self.duplicate_selected_block).pack(side=tk.LEFT, padx=2)
        ttk.Button(edit_row, text="Remove", command=self.remove_selected_block).pack(side=tk.LEFT, padx=2)
        ttk.Button(edit_row, text="Move Up", command=lambda: self.move_selected_block(-1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(edit_row, text="Move Down", command=lambda: self.move_selected_block(+1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(edit_row, text="Clear All", command=self.clear_all_blocks).pack(side=tk.RIGHT, padx=2)

        actions2 = ttk.Frame(listf)
        actions2.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 6))
        ttk.Button(actions2, text="Group Selected", command=self.group_selected_blocks).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions2, text="Ungroup", command=self.ungroup_selected_block).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions2, text="Add PQ Curve", command=self.add_pq_block).pack(side=tk.LEFT, padx=(12, 2))
        ttk.Button(actions2, text="Add Pressure", command=self.add_pressure_block).pack(side=tk.LEFT, padx=2)
        ttk.Label(listf, wraplength=380, justify=tk.LEFT,
                  text="A group repeats its blocks back-to-back (Repeat 0 = loop until Stop). "
                       "With a group row selected, Add Constant/Sweep/Voltage adds inside the group; "
                       "Esc clears the selection so Add appends at the end. "
                       "A voltage step pauses the run until you set the bench supply and confirm.")\
            .pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 6))

        editf = ttk.LabelFrame(parent, text="Edit selected block")
        editf.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))

        self.var_edit_type = tk.StringVar(value="constant")
        self.var_edit_repeat = tk.StringVar(value="1")
        self.var_edit_settle_ms = tk.StringVar(value="0")

        row0 = ttk.Frame(editf); row0.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        ttk.Label(row0, text="Type:").pack(side=tk.LEFT)
        self.cmb_edit_type = ttk.Combobox(row0, state="readonly", width=12, values=["constant", "sweep", "voltage"],
                                          textvariable=self.var_edit_type)
        self.cmb_edit_type.pack(side=tk.LEFT, padx=6)
        ttk.Label(row0, text="Repeat:").pack(side=tk.LEFT, padx=(14, 2))
        ttk.Entry(row0, textvariable=self.var_edit_repeat, width=8).pack(side=tk.LEFT)
        ttk.Label(row0, text="Settle ms:").pack(side=tk.LEFT, padx=(14, 2))
        ttk.Entry(row0, textvariable=self.var_edit_settle_ms, width=8).pack(side=tk.LEFT)

        # Constant editor
        self.var_edit_freq = tk.StringVar(value="500")
        self.var_edit_dur  = tk.StringVar(value="10")

        self.edit_constant = ttk.Frame(editf)
        ttk.Label(self.edit_constant, text="Frequency (Hz):").pack(side=tk.LEFT)
        ttk.Entry(self.edit_constant, textvariable=self.var_edit_freq, width=10).pack(side=tk.LEFT, padx=6)
        ttk.Label(self.edit_constant, text="Duration (s):").pack(side=tk.LEFT, padx=(14, 2))
        ttk.Entry(self.edit_constant, textvariable=self.var_edit_dur, width=10).pack(side=tk.LEFT, padx=6)

        # Sweep editor
        self.var_edit_start = tk.StringVar(value="100")
        self.var_edit_end   = tk.StringVar(value="1000")
        self.var_edit_steps = tk.StringVar(value="10")
        self.var_edit_step_dur = tk.StringVar(value="1.5")
        self.var_edit_total_dur = tk.StringVar(value="")

        self.edit_sweep = ttk.Frame(editf)
        row1 = ttk.Frame(self.edit_sweep); row1.pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Label(row1, text="Start Hz:").pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.var_edit_start, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Label(row1, text="End Hz:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(row1, textvariable=self.var_edit_end, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Label(row1, text="Steps:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(row1, textvariable=self.var_edit_steps, width=8).pack(side=tk.LEFT, padx=4)

        row2 = ttk.Frame(self.edit_sweep); row2.pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Label(row2, text="Step duration (s):").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.var_edit_step_dur, width=10).pack(side=tk.LEFT, padx=6)
        ttk.Label(row2, text="OR Total duration (s):").pack(side=tk.LEFT, padx=(14, 2))
        ttk.Entry(row2, textvariable=self.var_edit_total_dur, width=10).pack(side=tk.LEFT, padx=6)

        # Group editor (children are edited as individual rows in the list)
        self.edit_group = ttk.Frame(editf)
        ttk.Label(self.edit_group,
                  text="Group: child blocks run in order, repeated 'Repeat' times (0 = loop until Stop).")\
            .pack(side=tk.LEFT)

        # Voltage-step editor (Repeat / Settle ms above don't apply)
        self.var_edit_volt = tk.StringVar(value="2.500")
        self.var_edit_vwait = tk.StringVar(value="zero")
        self.var_edit_vsettle = tk.StringVar(value="30")
        self.var_edit_vfrom = tk.StringVar(value="")
        self.var_edit_vto = tk.StringVar(value="")
        self.edit_voltage = ttk.Frame(editf)
        vrow1 = ttk.Frame(self.edit_voltage); vrow1.pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Label(vrow1, text="Supply voltage (V):").pack(side=tk.LEFT)
        ttk.Entry(vrow1, textvariable=self.var_edit_volt, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Label(vrow1, text="Pump while waiting:").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Combobox(vrow1, state="readonly", width=6, values=["zero", "hold"],
                     textvariable=self.var_edit_vwait).pack(side=tk.LEFT, padx=4)
        ttk.Label(vrow1, text="Settle after (s):").pack(side=tk.LEFT, padx=(10, 2))
        ttk.Entry(vrow1, textvariable=self.var_edit_vsettle, width=6).pack(side=tk.LEFT, padx=4)
        vrow2 = ttk.Frame(self.edit_voltage); vrow2.pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Label(vrow2, text="Only ask between (HH:MM, blank = any time):").pack(side=tk.LEFT)
        ttk.Entry(vrow2, textvariable=self.var_edit_vfrom, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(vrow2, text="and").pack(side=tk.LEFT)
        ttk.Entry(vrow2, textvariable=self.var_edit_vto, width=6).pack(side=tk.LEFT, padx=4)

        # Pressure editor (hold / release / idle)
        self.var_edit_ptarget = tk.StringVar(value="50")
        self.var_edit_pwait = tk.BooleanVar(value=True)
        self.var_edit_ptimeout = tk.StringVar(value="120")
        self.edit_pressure = ttk.Frame(editf)
        ttk.Label(self.edit_pressure, text="Target (mbar, or release / idle):").pack(side=tk.LEFT)
        ttk.Entry(self.edit_pressure, textvariable=self.var_edit_ptarget, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(self.edit_pressure, text="wait settled, max (s):",
                        variable=self.var_edit_pwait).pack(side=tk.LEFT, padx=(8, 2))
        ttk.Entry(self.edit_pressure, textvariable=self.var_edit_ptimeout, width=6).pack(side=tk.LEFT)

        # PQ curve editor
        self.var_edit_pqfreqs = tk.StringVar(value="600")
        self.var_edit_pqps = tk.StringVar(value="0, 25, 50, 100, 150, 200")
        self.var_edit_pqsettle = tk.StringVar(value="120")
        self.var_edit_pqmeasure = tk.StringVar(value="30")
        self.var_edit_pqtimeout = tk.StringVar(value="measure")
        self.var_edit_pqrelease = tk.BooleanVar(value=True)
        self.edit_pq = ttk.Frame(editf)
        q1 = ttk.Frame(self.edit_pq); q1.pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Label(q1, text="Frequencies (Hz):").pack(side=tk.LEFT)
        ttk.Entry(q1, textvariable=self.var_edit_pqfreqs, width=16).pack(side=tk.LEFT, padx=4)
        ttk.Label(q1, text="Measure (s):").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Entry(q1, textvariable=self.var_edit_pqmeasure, width=5).pack(side=tk.LEFT)
        q2 = ttk.Frame(self.edit_pq); q2.pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Label(q2, text="Pressures (mbar):").pack(side=tk.LEFT)
        ttk.Entry(q2, textvariable=self.var_edit_pqps, width=34).pack(side=tk.LEFT, padx=4)
        q3 = ttk.Frame(self.edit_pq); q3.pack(side=tk.TOP, fill=tk.X, pady=2)
        ttk.Label(q3, text="Settle max (s):").pack(side=tk.LEFT)
        ttk.Entry(q3, textvariable=self.var_edit_pqsettle, width=5).pack(side=tk.LEFT, padx=4)
        ttk.Label(q3, text="if not settled:").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Combobox(q3, state="readonly", width=8, values=["measure", "skip"],
                     textvariable=self.var_edit_pqtimeout).pack(side=tk.LEFT)
        ttk.Checkbutton(q3, text="release after", variable=self.var_edit_pqrelease).pack(side=tk.LEFT, padx=8)

        self.btn_apply_edit = ttk.Button(editf, text="Apply changes", command=self.apply_edit_to_selected)
        self.btn_apply_edit.pack(side=tk.TOP, padx=6, pady=(6, 8))

        self.var_edit_type.trace_add("write", lambda *_: self._refresh_editor_visibility())
        self._refresh_editor_visibility()

    def _refresh_editor_visibility(self):
        ttype = self.var_edit_type.get().strip().lower()
        self.edit_constant.pack_forget()
        self.edit_sweep.pack_forget()
        self.edit_group.pack_forget()
        self.edit_voltage.pack_forget()
        self.edit_pressure.pack_forget()
        self.edit_pq.pack_forget()
        frame = {"constant": self.edit_constant, "pressure": self.edit_pressure, "pq": self.edit_pq,
                 "group": self.edit_group, "voltage": self.edit_voltage}.get(ttype, self.edit_sweep)
        frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=2, before=self.btn_apply_edit)

    def _build_manual_panel(self, parent):
        self.var_manual_enabled = tk.BooleanVar(value=False)
        self.var_manual_freq = tk.StringVar(value="500")
        self.var_manual_duty = tk.StringVar(value="0.30")

        row = ttk.Frame(parent)
        row.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Checkbutton(row, text="Output ON", variable=self.var_manual_enabled, command=self._manual_toggle_output)\
            .pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(row, text="Frequency (Hz):").pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=self.var_manual_freq, width=10).pack(side=tk.LEFT, padx=6)
        ttk.Button(row, text="Set freq", command=self._manual_set_freq).pack(side=tk.LEFT, padx=4)

        row2 = ttk.Frame(parent)
        row2.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 6))
        ttk.Label(row2, text="Duty (0..1):").pack(side=tk.LEFT)
        self.ent_manual_duty = ttk.Entry(row2, textvariable=self.var_manual_duty, width=8)
        self.ent_manual_duty.pack(side=tk.LEFT, padx=6)
        self.btn_set_duty = ttk.Button(row2, text="Set duty", command=self._manual_set_duty)
        self.btn_set_duty.pack(side=tk.LEFT, padx=4)

        info = ttk.Frame(parent)
        info.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))
        self.var_manual_info = tk.StringVar(value="Manual session not connected.")
        ttk.Label(info, textvariable=self.var_manual_info, wraplength=380, justify=tk.LEFT).pack(side=tk.LEFT)

    def _build_plot(self):
        self.fig = plt.figure(figsize=(8.2, 5.2))
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)

        # The data lines are "animated": a full draw leaves them out, they are drawn on top of the
        # cached background each frame (blitting, _update_plot)
        self.line_freq, = self.ax1.plot([], [], lw=1, animated=True)
        self.line_flow, = self.ax2.plot([], [], lw=1, animated=True)
        # Supply current from the PPK2 on a second y-axis of the frequency plot (hidden without a meter)
        self.ax1_cur = self.ax1.twinx()
        self.line_cur, = self.ax1_cur.plot([], [], lw=0.8, color="tab:red", animated=True)
        self.ax1_cur.set_ylabel("mA", color="tab:red")
        self.ax1_cur.tick_params(axis="y", labelcolor="tab:red")
        self.ax1_cur.set_visible(False)
        # Line pressure (and the held target) on a second y-axis of the flow plot
        self.ax2_p = self.ax2.twinx()
        self.line_p, = self.ax2_p.plot([], [], lw=0.9, color="tab:green", animated=True)
        self.line_ptgt, = self.ax2_p.plot([], [], lw=0.8, ls="--", color="darkgreen", animated=True)
        self._plot_lines = [self.line_freq, self.line_cur, self.line_flow, self.line_p, self.line_ptgt]
        self._plot_bg = None
        self.ax2_p.set_ylabel("mbar", color="tab:green")
        self.ax2_p.tick_params(axis="y", labelcolor="tab:green")
        self.ax2_p.set_visible(False)

        self.ax1.set_title("Frequency setpoint (Hz)")
        self.ax1.set_ylabel("Hz")
        self.ax2.set_title("Flow-rate (sensor units)")
        self.ax2.set_xlabel("t (s)")
        self.ax2.set_ylabel("Flow")

        self.ax1.grid(True, alpha=0.25)
        self.ax2.grid(True, alpha=0.25)

        # Zoom controls frame
        zoom_frame = ttk.Frame(self.plot_frame)
        zoom_frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(4, 2))
        
        # Time window controls
        ttk.Label(zoom_frame, text="Time window (s):").pack(side=tk.LEFT, padx=(0, 4))
        self.var_time_window = tk.StringVar(value="30")
        ttk.Entry(zoom_frame, textvariable=self.var_time_window, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Button(zoom_frame, text="Zoom In", command=self._zoom_time_in).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Zoom Out", command=self._zoom_time_out).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Reset", command=self._zoom_time_reset).pack(side=tk.LEFT, padx=2)
        zoom_frame = ttk.Frame(self.plot_frame)          # second row: Y ranges
        zoom_frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 2))
        
        
        # Y-axis range for frequency plot
        ttk.Label(zoom_frame, text="Freq Y min:").pack(side=tk.LEFT)
        self.var_freq_ymin = tk.StringVar(value="")
        ttk.Entry(zoom_frame, textvariable=self.var_freq_ymin, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(zoom_frame, text="max:").pack(side=tk.LEFT)
        self.var_freq_ymax = tk.StringVar(value="")
        ttk.Entry(zoom_frame, textvariable=self.var_freq_ymax, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Set Freq Y", command=self._set_freq_y_range).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(zoom_frame, text="  |  ").pack(side=tk.LEFT)
        
        # Y-axis range for flow plot
        ttk.Label(zoom_frame, text="Flow Y min:").pack(side=tk.LEFT)
        self.var_flow_ymin = tk.StringVar(value="")
        ttk.Entry(zoom_frame, textvariable=self.var_flow_ymin, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(zoom_frame, text="max:").pack(side=tk.LEFT)
        self.var_flow_ymax = tk.StringVar(value="")
        ttk.Entry(zoom_frame, textvariable=self.var_flow_ymax, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Set Flow Y", command=self._set_flow_y_range).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="Auto Y", command=self._auto_y_range).pack(side=tk.LEFT, padx=2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Custom zoom/pan with mouse wheel on plots
        self.canvas.mpl_connect('scroll_event', self._on_scroll_event)
        # After every full draw (limits changed, resize, Y range set): new background, lines on top
        self.canvas.mpl_connect('draw_event', self._on_plot_draw)
        
        # Track if user has manually set y-axis ranges
        self._user_freq_y_range = None
        self._user_flow_y_range = None
        self._auto_y_enabled = True

    # ---- zoom controls ----
    def _get_time_window(self) -> float:
        """Get current time window in seconds."""
        try:
            return float(self.var_time_window.get())
        except ValueError:
            return 30.0

    def _set_time_window(self, window: float):
        """Set the time window variable."""
        self.var_time_window.set(str(int(window)))

    def _zoom_time_in(self):
        """Zoom in on time axis (decrease time window)."""
        current = self._get_time_window()
        new_window = max(5.0, current * 0.7)
        self._set_time_window(new_window)

    def _zoom_time_out(self):
        """Zoom out on time axis (increase time window)."""
        current = self._get_time_window()
        new_window = min(600.0, current * 1.5)
        self._set_time_window(new_window)

    def _zoom_time_reset(self):
        """Reset time window to default."""
        self._set_time_window(30.0)

    def _set_freq_y_range(self):
        """Set custom Y range for frequency plot."""
        try:
            ymin = safe_float(self.var_freq_ymin.get(), None)
            ymax = safe_float(self.var_freq_ymax.get(), None)
            if ymin is not None and ymax is not None:
                self._user_freq_y_range = (ymin, ymax)
                self._auto_y_enabled = False
                self.ax1.set_ylim(ymin, ymax)
                self.canvas.draw()
        except Exception:
            log_exception("Plot: failed to set frequency Y range")

    def _set_flow_y_range(self):
        """Set custom Y range for flow plot."""
        try:
            ymin = safe_float(self.var_flow_ymin.get(), None)
            ymax = safe_float(self.var_flow_ymax.get(), None)
            if ymin is not None and ymax is not None:
                self._user_flow_y_range = (ymin, ymax)
                self._auto_y_enabled = False
                self.ax2.set_ylim(ymin, ymax)
                self.canvas.draw()
        except Exception:
            log_exception("Plot: failed to set flow Y range")

    def _auto_y_range(self):
        """Enable auto Y range for both plots."""
        self._user_freq_y_range = None
        self._user_flow_y_range = None
        self._auto_y_enabled = True

    def _on_scroll_event(self, event):
        """Handle mouse scroll for zooming time axis."""
        if event.inaxes:
            base_scale = 1.1
            if event.button == 'scrollup':
                self._zoom_time_in()
            elif event.button == 'scrolldown':
                self._zoom_time_out()

    # ---- plot update ----
    def _plot_tick(self):
        try:
            if self.center_tabs.select() == str(self.plot_frame):
                self._update_plot()
        except Exception:
            log_exception_limited("plot_tick", "Plot update failed")
        self.root.after(150, self._plot_tick)

    def _on_plot_draw(self, _evt=None):
        self._plot_bg = self.canvas.copy_from_bbox(self.fig.bbox)
        self._draw_plot_lines()

    def _draw_plot_lines(self):
        for ln in self._plot_lines:
            if ln.axes.get_visible():
                ln.axes.draw_artist(ln)

    @staticmethod
    def _fit_ylim(cur, vals, min_pad: float):
        """Y-limits for vals: the current ones while the data stays inside and fills at least
        40 % of them (so most frames need no full redraw), else new ones with 10 % padding."""
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if not len(vals):
            return cur
        lo, hi = float(vals.min()), float(vals.max())
        pad = max(0.1 * (hi - lo), min_pad)
        new = (lo - pad, hi + pad)
        if cur is not None and cur[0] <= lo and hi <= cur[1] and (new[1] - new[0]) >= 0.4 * (cur[1] - cur[0]):
            return cur
        return new

    def _update_plot(self, _frame=None):
        """Redraw the live plot. The x-window advances in steps of 1/5 of its width and the
        y-limits only change when the data leaves them, so most frames just restore the cached
        background and draw the lines (a few ms) instead of a full redraw (~100+ ms holding the
        GIL, which the drive timing needs)."""
        if not self.active_stream:
            return
        ts, freqs, flows = self.active_stream.get_plot_data()
        if not ts:
            return
        window = self._get_time_window()
        step = max(1.0, window / 5.0)
        x_right = max(window, (math.floor(ts[-1] / step) + 1) * step)
        tmin = x_right - window
        # Only the visible window, thinned: the buffers hold far more points than the plot shows
        ts, freqs, flows = window_decimate(ts, [freqs, flows], tmin)
        self.line_freq.set_data(ts, freqs)
        self.line_flow.set_data(ts, flows)
        lims = {"x": (tmin, x_right)}

        pts, pps, ptg = self.active_stream.get_pressure_plot_data()
        p_on = len(pts) > 0
        if p_on:
            pts, pps, ptg = window_decimate(pts, [pps, ptg], tmin)
            self.line_p.set_data(pts, pps)
            self.line_ptgt.set_data(pts, ptg)
            lims["p"] = self._fit_ylim(self._plot_lims_cur("p"), np.concatenate((pps, ptg)), 1.0)
        lims["p_on"] = p_on

        pt, pi = self.active_stream.get_power_plot_data()
        cur_on = len(pt) > 0
        if cur_on:
            pt, pi = window_decimate(pt, [pi], tmin)
            self.line_cur.set_data(pt, pi)
            lims["cur"] = self._fit_ylim(self._plot_lims_cur("cur"), pi, 0.1)
        lims["cur_on"] = cur_on

        lims["freq"] = self._user_freq_y_range or self._fit_ylim(self._plot_lims_cur("freq"), freqs, 1.0)
        lims["flow"] = self._user_flow_y_range or self._fit_ylim(self._plot_lims_cur("flow"), flows, 0.1)

        if lims != getattr(self, "_plot_lims", None) or self._plot_bg is None:
            self._plot_lims = lims
            self.ax1.set_xlim(*lims["x"])
            self.ax2.set_xlim(*lims["x"])
            self.ax1.set_ylim(*lims["freq"])
            self.ax2.set_ylim(*lims["flow"])
            self.ax2_p.set_visible(lims["p_on"])
            self.ax2.set_title("Flow-rate (sensor units)" + (" and pressure (mbar, green)" if lims["p_on"] else ""))
            if lims["p_on"] and lims.get("p"):
                self.ax2_p.set_ylim(*lims["p"])
            self.ax1_cur.set_visible(lims["cur_on"])
            self.ax1.set_title("Frequency setpoint (Hz)" + (" and supply current (mA, red)" if lims["cur_on"] else ""))
            if lims["cur_on"] and lims.get("cur"):
                self.ax1_cur.set_ylim(*lims["cur"])
            self.canvas.draw()                  # -> _on_plot_draw: new background + lines
        else:
            self.canvas.restore_region(self._plot_bg)
            self._draw_plot_lines()
            self.canvas.blit(self.fig.bbox)

    def _plot_lims_cur(self, key: str):
        return (getattr(self, "_plot_lims", None) or {}).get(key)

    def _on_center_tab(self, _evt=None):
        # Back on the live tab: the cached background may be stale (resized while hidden)
        if self.center_tabs.select() == str(self.plot_frame):
            self._plot_bg = None

    # ---- PQ curve tab ----
    def _build_pq_tab(self):
        from matplotlib.figure import Figure
        frame = ttk.Frame(self.center_tabs)
        self.center_tabs.add(frame, text="PQ curve")
        bar = ttk.Frame(frame)
        bar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(4, 2))
        ttk.Button(bar, text="Open pq_points.csv...", command=self._pq_open_csv).pack(side=tk.LEFT, padx=2)
        ttk.Button(bar, text="Clear", command=self._pq_clear).pack(side=tk.LEFT, padx=2)
        self.var_pq_info = tk.StringVar(value="Points appear here while a run with a PQ block measures.")
        ttk.Label(bar, textvariable=self.var_pq_info).pack(side=tk.LEFT, padx=10)
        self.pq_fig = Figure(figsize=(7.0, 4.5))
        self.pq_ax = self.pq_fig.add_subplot(111)
        self.pq_canvas = FigureCanvasTkAgg(self.pq_fig, master=frame)
        self.pq_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.pq_rows: List[Dict[str, Any]] = []
        self._redraw_pq()

    def _pq_clear(self):
        self.pq_rows = []
        self.var_pq_info.set("")
        self._redraw_pq()

    def _pq_open_csv(self):
        p = filedialog.askopenfilename(title="Open pq_points.csv", filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if not p:
            return
        try:
            with open(p, newline="", encoding="utf-8") as f:
                self.pq_rows = list(csv.DictReader(f))
        except Exception as e:
            messagebox.showerror("PQ curve", f"Could not read {p}: {e}")
            return
        self.var_pq_info.set(f"{len(self.pq_rows)} points from {p}")
        self._redraw_pq()

    def _add_pq_point(self, row: Dict[str, Any]):
        self.pq_rows.append(row)
        note = f" ({row['note']})" if row.get("note") else ""
        self.var_pq_info.set(f"{len(self.pq_rows)} points · last: {row.get('frequency_hz')} Hz, "
                             f"{row.get('p_mean_mbar') or row.get('target_mbar')} mbar, flow {row.get('flow_mean') or '-'}"
                             + note)
        self._redraw_pq()

    def _redraw_pq(self):
        def num(r, k):
            try:
                v = float(r.get(k, ""))
                return v if v == v else None
            except (TypeError, ValueError):
                return None
        ax = self.pq_ax
        ax.clear()
        curves: Dict[int, list] = {}
        shutoff: Dict[int, list] = {}
        for r in self.pq_rows:
            p, q, hz = num(r, "p_mean_mbar"), num(r, "flow_mean"), num(r, "frequency_hz")
            if p is None or q is None or hz is None:
                continue
            if r.get("note") == "above_shutoff":
                shutoff.setdefault(int(hz), []).append((p, q))
                continue
            curves.setdefault(int(hz), []).append((p, q, num(r, "p_sd_mbar") or 0.0, num(r, "flow_sd") or 0.0))
        for hz in sorted(set(curves) | set(shutoff)):
            pts = sorted(curves.get(hz, []))
            colour = None
            if pts:
                eb = ax.errorbar([a[0] for a in pts], [a[1] for a in pts], xerr=[a[2] for a in pts],
                                 yerr=[a[3] for a in pts], marker="o", ms=4, capsize=2, lw=1, label=f"{hz} Hz")
                colour = eb[0].get_color()
            if hz in shutoff:
                # Pressure plateau with the syringe unable to push: the pump's shut-off point
                ax.plot([a[0] for a in shutoff[hz]], [a[1] for a in shutoff[hz]], "x", ms=9, mew=2, color=colour,
                        label=f"{hz} Hz shut-off (plateau)")
        ax.set_xlabel("Outlet pressure (mbar)")
        ax.set_ylabel("Flow (flow-sensor units)")
        ax.set_title("PQ curve")
        ax.grid(True, alpha=0.25)
        if curves:
            ax.legend(fontsize=8)
        self.pq_canvas.draw_idle()

    # ---- config IO ----
    def pick_out_dir(self):
        d = filedialog.askdirectory(title="Choose output directory")
        if d:
            self.var_out_dir.set(d)

    def pick_and_load_config(self):
        p = filedialog.askopenfilename(title="Select config JSON", filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if p:
            self.load_config(p)

    def load_config(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            self.cfg = json.load(f)
        self.cfg_path = path
        self.var_cfg_file.set(path)

        self.var_port.set(str(self.cfg.get("port", self.var_port.get())))
        self.var_baud.set(str(self.cfg.get("baud", self.var_baud.get())))
        self.var_break_between.set(str(self.cfg.get("break_seconds_between_tests", self.var_break_between.get())))

        serial_cfg = self.cfg.get("serial", {})
        if isinstance(serial_cfg, dict):
            self.var_protocol.set(serial_cfg.get("protocol", self.var_protocol.get()))

        flow_cfg = self.cfg.get("flow_sensor", {})
        if isinstance(flow_cfg, dict):
            self.var_sensor_backend.set(flow_cfg.get("backend", self.var_sensor_backend.get()))
            self.var_sensor_channel.set(str(flow_cfg.get("channel", self.var_sensor_channel.get())))
            self.var_flow_server.set(str(flow_cfg.get("server", self.var_flow_server.get())))
            self.var_sample_rate.set(str(flow_cfg.get("sample_rate_hz", self.var_sample_rate.get())))

        video = self.cfg.get("video")
        if isinstance(video, dict) and hasattr(self, "var_video_overlay"):
            if "overlay" in video:
                self.var_video_overlay.set(bool(video["overlay"]))
            if video.get("codec") in VIDEO_CODECS:
                self.var_video_codec.set(video["codec"])

        supply = self.cfg.get("supply")
        if isinstance(supply, dict):
            if supply.get("initial_mv") is not None:
                self.var_supply_v.set(f"{safe_float(supply['initial_mv'], 2500.0) / 1000:.3f}")
            if supply.get("current_limit_ma") is not None:
                self.var_supply_limit.set(str(supply["current_limit_ma"]))
        power = self.cfg.get("power")
        if isinstance(power, dict):
            self.var_pw_enabled.set(bool(power.get("enabled", False)) and POWER_AVAILABLE)
            self.var_pw_backend.set(str(power.get("backend") or "ppk2"))
            self.var_pw_port.set(str(power.get("port") or "auto"))
            self.var_pw_bin.set(str(power.get("bin_ms", 10)))
            self.var_pw_watch.set("" if power.get("watch_ma") in (None, "") else str(power["watch_ma"]))

        pressure = self.cfg.get("pressure")
        if isinstance(pressure, dict) and hasattr(self, "var_pr_enabled"):
            self.var_pr_enabled.set(bool(pressure.get("enabled", False)) and PRESSURE_AVAILABLE)
            self.var_pr_backend.set(str(pressure.get("backend") or "arduino"))
            self.var_pr_port.set(str(pressure.get("port") or self.var_pr_port.get()))
            self.var_pr_fw.set(str(pressure.get("firmware") or "auto"))
            self.var_pr_limit.set(str(pressure.get("limit_mbar", self.var_pr_limit.get())))
            self.var_pr_min.set(str(pressure.get("min_mbar", self.var_pr_min.get())))
            sensor = pressure.get("sensor") if isinstance(pressure.get("sensor"), dict) else {}
            self.var_pr_sensor.set(str(sensor.get("backend") or self.var_pr_sensor.get()))
            self.var_pr_channel.set(str(sensor.get("channel", self.var_pr_channel.get())))
            self.var_pr_micro.set(str(pressure.get("microsteps", self.var_pr_micro.get())))

        meta = self.cfg.get("meta", {})
        if not isinstance(meta, dict):
            meta = {}
        self.var_test_name.set(meta.get("name", self._default_test_name))
        self.var_operator.set(meta.get("operator", self._default_operator))
        self.var_pump_id.set(str(meta.get("pump_id") or ""))
        self.txt_desc.delete("1.0", tk.END)
        self.txt_desc.insert("1.0", meta.get("description", self._default_desc))

        self.cfg.setdefault("tests", [])
        self.refresh_blocks_list()
        self.var_status.set(f"Loaded config: {os.path.basename(path)}")
        self._refresh_manual_duty_state()

    def build_config_from_ui(self) -> Dict[str, Any]:
        """The panel's values on top of the loaded config: keys the panel doesn't
        show (e.g. meta.tube_id_mm, analysis notes) are kept, panel values win."""
        built = self._config_from_panel()
        old_meta = self.cfg.get("meta") if isinstance(self.cfg.get("meta"), dict) else {}
        per_run = ("created", "output_dir", "run_dir", "clock")
        built["meta"] = {**{k: v for k, v in old_meta.items() if k not in per_run}, **built["meta"]}
        extra = {k: v for k, v in self.cfg.items() if k not in built}
        return {**extra, **built}

    def _config_from_panel(self) -> Dict[str, Any]:
        return {
            "port": self.var_port.get().strip(),
            "baud": safe_int(self.var_baud.get(), 115200),
            "read_timeout_s": 0.2,
            "start_time": None,
            "end_time": None,
            "break_seconds_between_tests": safe_float(self.var_break_between.get(), 0.0),
            "send_zero_on_exit": True,
            "tests": self.cfg.get("tests", []),
            "meta": {
                "name": self.var_test_name.get().strip(),
                "description": self.txt_desc.get("1.0", tk.END).strip(),
                "operator": self.var_operator.get().strip(),
                "pump_id": self.var_pump_id.get().strip(),
                "created": iso_now(),
                "output_dir": self.var_out_dir.get().strip(),
            },
            "require_pump_id": bool(self.cfg.get("require_pump_id", False)),
            "serial": {"protocol": self.var_protocol.get().strip()},
            "flow_sensor": {
                "backend": self.var_sensor_backend.get().strip(),
                "channel": safe_int(self.var_sensor_channel.get(), 0),
                "server": self.var_flow_server.get().strip(),
                "sample_rate_hz": safe_float(self.var_sample_rate.get(), 25.0),
            },
            # Bench supply (voltage steps, power calculation) and the PPK2 power meter
            "supply": self._supply_cfg(),
            "power": self._power_cfg(),
            "pressure": self._pressure_cfg(),
            "video": self._video_cfg(),
        }

    def save_config_as(self):
        p = filedialog.asksaveasfilename(title="Save config as", defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not p:
            return
        cfg = self.build_config_from_ui()
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        self.cfg = cfg
        self.cfg_path = p
        self.var_cfg_file.set(p)
        self.refresh_blocks_list()
        self.var_status.set(f"Saved config: {os.path.basename(p)}")

    # ---- config builder list ----
    @staticmethod
    def _is_group(t) -> bool:
        return (t.get("type") or "").lower() == "group"

    @staticmethod
    def _block_duration_s(t) -> float:
        """Nominal run time of one block including its repeats (inf for an endless group)."""
        return block_duration_s(t)

    @staticmethod
    def _describe_block(t) -> str:
        ttype = (t.get("type") or "constant").lower()
        if ttype == "constant":
            return f"constant: {t.get('frequency_hz')} Hz for {t.get('duration_s')} s (x{t.get('repeat', 1)})"
        if ttype == "voltage":
            mv = safe_float(t.get("voltage_mv", 0), 0.0)
            pump = "pump holds" if str(t.get("while_waiting", "zero")).lower() == "hold" else "pump 0 Hz"
            w = t.get("window")
            when = f", ask {w[0]}-{w[1]}" if voltage_window(t) else ""
            return (f"VOLTAGE STEP → {mv / 1000:.3f} V (wait for you, {pump}, "
                    f"settle {safe_float(t.get('settle_s', 0), 0.0):g} s{when})")
        if ttype == "pressure":
            tgt = t.get("target_mbar", "release")
            if not is_pressure_target(t):
                return f"PRESSURE: {str(tgt).lower()}"
            wait = f", wait until settled (max {safe_float(t.get('timeout_s', 120), 120.0):g} s)" \
                if t.get("wait_settled", True) else ""
            return f"PRESSURE: hold {safe_float(tgt, 0.0):g} mbar{wait}"
        if ttype == "pq":
            ps = t.get("pressures_mbar") or []
            fr = ", ".join(str(f) for f in pq_frequencies(t))
            rng = f"{min(ps):g}..{max(ps):g} mbar" if ps else "no pressures"
            return (f"PQ CURVE: {fr} Hz x {len(ps)} pressures ({rng}), measure {t.get('measure_s', 30)} s, "
                    f"settle max {t.get('settle_timeout_s', 120)} s")
        if ttype == "group":
            children = t.get("blocks", [])
            repeat = safe_int(t.get("repeat", 1), 1)
            reps = "loop until Stop" if repeat <= 0 else f"x{repeat}"
            cycle_s = sum(DashboardApp._block_duration_s(b) for b in children)
            return f"GROUP ({reps}): {len(children)} blocks, {cycle_s:g} s per cycle"
        sdur = t.get("step_duration_s", None)
        tdur = t.get("total_duration_s", None)
        d = f"{sdur}s/step" if sdur is not None else f"{tdur}s total"
        return f"sweep: {t.get('start_hz')}→{t.get('end_hz')} Hz, {t.get('steps')} steps, {d} (x{t.get('repeat', 1)})"

    def refresh_blocks_list(self):
        self.lst_blocks.delete(0, tk.END)
        self._row_paths = []
        volt_bg = "#fff2cc"
        for i, t in enumerate(self.cfg.get("tests", [])):
            self.lst_blocks.insert(tk.END, f"{i + 1:02d}. {self._describe_block(t)}")
            self._row_paths.append((i,))
            if block_type(t) == "voltage":
                self.lst_blocks.itemconfig(tk.END, background=volt_bg)
            if block_type(t) in ("pressure", "pq"):
                self.lst_blocks.itemconfig(tk.END, background="#e2f0d9")
            if self._is_group(t):
                self.lst_blocks.itemconfig(tk.END, background="#e3ebfa")
                for j, child in enumerate(t.get("blocks", [])):
                    self.lst_blocks.insert(tk.END, f"        {i + 1}.{j + 1}  {self._describe_block(child)}")
                    self.lst_blocks.itemconfig(tk.END, background=volt_bg if block_type(child) == "voltage"
                                               else "#f3f6fc")
                    self._row_paths.append((i, j))

    def _selected_paths(self) -> List[tuple]:
        rows = self.lst_blocks.curselection()
        return [self._row_paths[int(r)] for r in rows if int(r) < len(self._row_paths)]

    def _block_at(self, path: tuple) -> Dict[str, Any]:
        container, idx = self._container_of(path)
        return container[idx]

    def _container_of(self, path: tuple) -> Tuple[List[Dict[str, Any]], int]:
        """Return (list, index) such that list[index] is the block at path."""
        tests = self.cfg.setdefault("tests", [])
        if len(path) == 1:
            return tests, path[0]
        return tests[path[0]].setdefault("blocks", []), path[1]

    def _select_path(self, path: tuple):
        self.lst_blocks.selection_clear(0, tk.END)
        if path in self._row_paths:
            row = self._row_paths.index(path)
            self.lst_blocks.selection_set(row)
            self.lst_blocks.activate(row)
            self.lst_blocks.see(row)
            self._on_block_select()

    def _on_block_select(self, _evt=None):
        paths = self._selected_paths()
        if not paths:
            return
        t = self._block_at(paths[0])
        ttype = (t.get("type") or "constant").lower()

        # A group can't be turned into a constant/sweep (its children would be lost)
        self.cmb_edit_type.configure(values=["group"] if ttype == "group"
                                     else ["constant", "sweep", "voltage", "pressure", "pq"])
        self.var_edit_type.set(ttype)
        self.var_edit_repeat.set(str(t.get("repeat", 1)))
        self.var_edit_settle_ms.set(str(t.get("settle_ms", 0)))

        if ttype == "constant":
            self.var_edit_freq.set(str(t.get("frequency_hz", 500)))
            self.var_edit_dur.set(str(t.get("duration_s", 10)))
        elif ttype == "voltage":
            self.var_edit_volt.set(f"{safe_float(t.get('voltage_mv', 2500), 2500.0) / 1000:.3f}")
            self.var_edit_vwait.set("hold" if str(t.get("while_waiting", "zero")).lower() == "hold" else "zero")
            self.var_edit_vsettle.set(str(t.get("settle_s", 0)))
            w = t.get("window") if voltage_window(t) else None
            self.var_edit_vfrom.set(str(w[0]) if w else "")
            self.var_edit_vto.set(str(w[1]) if w else "")
        elif ttype == "pressure":
            tgt = t.get("target_mbar", "release")
            self.var_edit_ptarget.set(str(tgt) if not is_pressure_target(t) else f"{safe_float(tgt, 0.0):g}")
            self.var_edit_pwait.set(bool(t.get("wait_settled", True)))
            self.var_edit_ptimeout.set(str(t.get("timeout_s", 120)))
        elif ttype == "pq":
            self.var_edit_pqfreqs.set(", ".join(str(f) for f in pq_frequencies(t)))
            self.var_edit_pqps.set(", ".join(f"{float(p):g}" for p in t.get("pressures_mbar") or []))
            self.var_edit_pqsettle.set(str(t.get("settle_timeout_s", 120)))
            self.var_edit_pqmeasure.set(str(t.get("measure_s", 30)))
            self.var_edit_pqtimeout.set(str(t.get("on_timeout", "measure")))
            self.var_edit_pqrelease.set(bool(t.get("release_after", True)))
        elif ttype != "group":
            self.var_edit_start.set(str(t.get("start_hz", 100)))
            self.var_edit_end.set(str(t.get("end_hz", 1000)))
            self.var_edit_steps.set(str(t.get("steps", 10)))
            self.var_edit_step_dur.set(str(t.get("step_duration_s", "")))
            self.var_edit_total_dur.set(str(t.get("total_duration_s", "")))

        self._refresh_editor_visibility()

    def apply_edit_to_selected(self):
        paths = self._selected_paths()
        if not paths:
            messagebox.showinfo("Edit", "Select a block to edit.")
            return
        path = paths[0]

        ttype = self.var_edit_type.get().strip().lower()
        repeat = max(1, safe_int(self.var_edit_repeat.get(), 1))
        settle = max(0, safe_int(self.var_edit_settle_ms.get(), 0))

        if ttype == "group":
            grp = self._block_at(path)
            if self._is_group(grp):
                grp["repeat"] = max(0, safe_int(self.var_edit_repeat.get(), 1))
            self.refresh_blocks_list()
            self._select_path(path)
            return

        if ttype == "constant":
            block = {
                "type": "constant",
                "frequency_hz": safe_int(self.var_edit_freq.get(), 500),
                "duration_s": safe_float(self.var_edit_dur.get(), 10.0),
                "repeat": repeat,
                "settle_ms": settle,
            }
        elif ttype == "voltage":
            mv = int(round(safe_float(self.var_edit_volt.get().replace(",", "."), 2.5) * 1000))
            max_mv = safe_float(self._supply_cfg().get("max_mv"), SUPPLY_DEFAULTS["max_mv"])
            if mv > max_mv:
                messagebox.showerror("Voltage step", f"{mv / 1000:.3f} V is above the {max_mv / 1000:.3f} V "
                                                     f"maximum (supply.max_mv in the config).")
                return
            block = {
                "type": "voltage",
                "voltage_mv": mv,
                "while_waiting": self.var_edit_vwait.get().strip() or "zero",
                "settle_s": max(0.0, safe_float(self.var_edit_vsettle.get(), 0.0)),
            }
            frm, to = self.var_edit_vfrom.get().strip(), self.var_edit_vto.get().strip()
            if frm or to:
                if _parse_hhmm(frm) is None or _parse_hhmm(to) is None or _parse_hhmm(frm) == _parse_hhmm(to):
                    messagebox.showerror("Voltage step", "Time window: give both times as HH:MM "
                                                         "(e.g. 08:00 and 20:00), or leave both blank.")
                    return
                block["window"] = [frm, to]
            old = self._block_at(path)
            if block_type(old) == "voltage" and old.get("note"):
                block["note"] = old["note"]
        elif ttype == "pressure":
            raw = self.var_edit_ptarget.get().strip().lower().replace(",", ".")
            if raw in ("release", "idle"):
                block = {"type": "pressure", "target_mbar": raw}
            else:
                try:
                    tgt = float(raw)
                except ValueError:
                    messagebox.showerror("Pressure block", "Target: a pressure in mbar, 'release' or 'idle'.")
                    return
                block = {"type": "pressure", "target_mbar": tgt, "wait_settled": bool(self.var_edit_pwait.get()),
                         "timeout_s": max(1.0, safe_float(self.var_edit_ptimeout.get(), 120.0))}
        elif ttype == "pq":
            try:
                freqs = [int(float(x)) for x in self.var_edit_pqfreqs.get().replace(";", ",").split(",") if x.strip()]
                ps = [float(x) for x in self.var_edit_pqps.get().replace(";", ",").split(",") if x.strip()]
            except ValueError:
                messagebox.showerror("PQ block", "Frequencies and pressures: numbers separated by commas.")
                return
            if not freqs or not ps:
                messagebox.showerror("PQ block", "Give at least one frequency and one pressure.")
                return
            block = {"type": "pq", "frequency_hz": freqs if len(freqs) > 1 else freqs[0], "pressures_mbar": ps,
                     "settle_timeout_s": max(5.0, safe_float(self.var_edit_pqsettle.get(), 120.0)),
                     "measure_s": max(1.0, safe_float(self.var_edit_pqmeasure.get(), 30.0)),
                     "on_timeout": self.var_edit_pqtimeout.get().strip() or "measure",
                     "release_after": bool(self.var_edit_pqrelease.get())}
        else:
            step_dur = self.var_edit_step_dur.get().strip()
            total_dur = self.var_edit_total_dur.get().strip()
            block = {
                "type": "sweep",
                "start_hz": safe_int(self.var_edit_start.get(), 100),
                "end_hz": safe_int(self.var_edit_end.get(), 1000),
                "steps": max(1, safe_int(self.var_edit_steps.get(), 10)),
                "repeat": repeat,
                "settle_ms": settle,
            }
            if step_dur != "":
                block["step_duration_s"] = safe_float(step_dur, 1.0)
            elif total_dur != "":
                block["total_duration_s"] = safe_float(total_dur, 10.0)
            else:
                block["step_duration_s"] = 1.0

        container, idx = self._container_of(path)
        if self._is_group(container[idx]):
            return
        container[idx] = block
        self.refresh_blocks_list()
        self._select_path(path)

    def _insert_new_block(self, block: Dict[str, Any]):
        """Add a block inside the selected group (header or child row selected),
        otherwise at the end of the sequence."""
        tests = self.cfg.setdefault("tests", [])
        paths = self._selected_paths()
        new_path = None
        if paths:
            p = paths[0]
            if len(p) == 2:
                tests[p[0]]["blocks"].insert(p[1] + 1, block)
                new_path = (p[0], p[1] + 1)
            elif self._is_group(tests[p[0]]):
                children = tests[p[0]].setdefault("blocks", [])
                children.append(block)
                new_path = (p[0], len(children) - 1)
        if new_path is None:
            tests.append(block)
            new_path = (len(tests) - 1,)
        self.refresh_blocks_list()
        self._select_path(new_path)

    def add_constant_block(self):
        self._insert_new_block({"type": "constant", "frequency_hz": 500, "duration_s": 10, "repeat": 1, "settle_ms": 0})

    def add_sweep_block(self):
        self._insert_new_block({"type": "sweep", "start_hz": 100, "end_hz": 1000, "steps": 10,
                                "step_duration_s": 1.5, "repeat": 1, "settle_ms": 0})

    def add_voltage_block(self):
        self._insert_new_block({"type": "voltage", "voltage_mv": 2500, "while_waiting": "zero", "settle_s": 30})

    def add_pq_block(self):
        self._insert_new_block({"type": "pq", "frequency_hz": 600, "pressures_mbar": [0, 25, 50, 100, 150, 200],
                                "settle_timeout_s": 120, "measure_s": 30, "on_timeout": "measure",
                                "release_after": True})

    def add_pressure_block(self):
        self._insert_new_block({"type": "pressure", "target_mbar": 50, "wait_settled": True, "timeout_s": 120})

    def add_group_block(self):
        tests = self.cfg.setdefault("tests", [])
        tests.append({
            "type": "group",
            "repeat": 10,
            "blocks": [
                {"type": "constant", "frequency_hz": 500, "duration_s": 0.1, "repeat": 1, "settle_ms": 0},
                {"type": "constant", "frequency_hz": 0, "duration_s": 0.1, "repeat": 1, "settle_ms": 0},
            ],
        })
        self.refresh_blocks_list()
        self._select_path((len(tests) - 1,))

    def group_selected_blocks(self):
        paths = self._selected_paths()
        if not paths:
            messagebox.showinfo("Group", "Select one or more blocks to group (Ctrl/Shift+click to select several).")
            return
        if any(len(p) != 1 or self._is_group(self._block_at(p)) for p in paths):
            messagebox.showinfo("Group", "Only top-level constant/sweep blocks can be grouped "
                                         "(groups cannot be nested).")
            return
        tests = self.cfg["tests"]
        idxs = sorted(p[0] for p in paths)
        children = [tests[i] for i in idxs]
        for i in reversed(idxs):
            tests.pop(i)
        tests.insert(idxs[0], {"type": "group", "repeat": 10, "blocks": children})
        self.refresh_blocks_list()
        self._select_path((idxs[0],))

    def ungroup_selected_block(self):
        paths = self._selected_paths()
        tests = self.cfg.get("tests", [])
        if not paths or not self._is_group(tests[paths[0][0]]):
            messagebox.showinfo("Ungroup", "Select a group (or a block inside it) to ungroup.")
            return
        gi = paths[0][0]
        tests[gi:gi + 1] = tests[gi].get("blocks", [])
        self.refresh_blocks_list()
        if gi < len(tests):
            self._select_path((gi,))

    def remove_selected_block(self):
        paths = self._selected_paths()
        if not paths:
            return
        tests = self.cfg["tests"]
        top = sorted({p[0] for p in paths if len(p) == 1}, reverse=True)
        # Children first (highest index first) so the remaining paths stay valid
        for gi, ci in sorted((p for p in paths if len(p) == 2), reverse=True):
            if gi not in top:
                tests[gi]["blocks"].pop(ci)
        for i in top:
            tests.pop(i)
        self.refresh_blocks_list()

    def duplicate_selected_block(self):
        paths = self._selected_paths()
        if not paths:
            return
        container, idx = self._container_of(paths[0])
        container.insert(idx + 1, copy.deepcopy(container[idx]))
        self.refresh_blocks_list()
        self._select_path(paths[0][:-1] + (idx + 1,))

    def move_selected_block(self, direction: int):
        paths = self._selected_paths()
        if not paths:
            return
        container, idx = self._container_of(paths[0])
        new_idx = idx + int(direction)
        if new_idx < 0 or new_idx >= len(container):
            return
        container[idx], container[new_idx] = container[new_idx], container[idx]
        self.refresh_blocks_list()
        self._select_path(paths[0][:-1] + (new_idx,))

    def clear_all_blocks(self):
        if messagebox.askyesno("Clear", "Clear all test blocks?"):
            self.cfg["tests"] = []
            self.refresh_blocks_list()

    # ---- mode / runtime ----
    def _update_mode_ui(self):
        if self.active_stream is not None:
            return
        mode = self.var_mode.get()
        if mode == "scheduled":
            self.btn_start.config(state=tk.NORMAL)
            self.var_manual_info.set("Switch to Manual mode for real-time ON/OFF + frequency.")
        else:
            self.btn_start.config(state=tk.DISABLED)
            self.var_manual_info.set("Manual session not connected.")

    def _refresh_manual_duty_state(self):
        enabled = (self.var_protocol.get().strip() == "freq_duty")
        state = tk.NORMAL if enabled else tk.DISABLED
        self.ent_manual_duty.configure(state=state)
        self.btn_set_duty.configure(state=state)

    def start_scheduled_run(self):
        if self.active_stream is not None:
            return
        cfg = self.build_config_from_ui()
        cfg["protocol"] = cfg.get("serial", {}).get("protocol", "freq_only")

        if not cfg.get("tests"):
            messagebox.showerror("No tests", "No test blocks. Add at least one block in the Config builder.")
            return

        if cfg.get("require_pump_id") and not cfg["meta"].get("pump_id"):
            messagebox.showerror("Pump ID", "This config requires a pump ID. Type it in the Pump ID field "
                                            "(Run details) and start again.")
            return

        if contains_pressure_block(cfg["tests"]) and not (cfg.get("pressure") or {}).get("enabled"):
            messagebox.showerror("Pressure control", "The sequence has pressure / PQ blocks: tick 'Control "
                                                     "pressure' in the Pressure panel first.")
            return

        supply_now = {**SUPPLY_DEFAULTS, **(cfg.get("supply") or {})}
        if safe_float(supply_now.get("initial_mv") or 0, 0.0) > safe_float(supply_now.get("max_mv"), 2500.0):
            messagebox.showerror("Supply voltage", f"Supply voltage at start ({supply_now['initial_mv'] / 1000:.3f} V) "
                                                   f"is above the {supply_now['max_mv'] / 1000:.3f} V maximum.")
            return

        # Voltage steps: refuse voltages above the pump's maximum, then show the
        # schedule so the operator can set alarms before anything starts.
        schedule_txt = None
        if contains_voltage_block(cfg["tests"]):
            supply = {**SUPPLY_DEFAULTS, **(cfg.get("supply") or {})}
            total, changes, truncated = plan_schedule(cfg["tests"], float(cfg.get("break_seconds_between_tests", 0.0)))
            max_mv = safe_float(supply.get("max_mv"), SUPPLY_DEFAULTS["max_mv"])
            too_high = sorted({c["mv"] for c in changes if c["mv"] > max_mv})
            if too_high:
                messagebox.showerror("Voltage too high",
                                     f"Voltage step(s) above the {max_mv / 1000:.3f} V maximum: "
                                     + ", ".join(f"{mv / 1000:.3f} V" for mv in too_high)
                                     + "\n\nFix the blocks (or supply.max_mv in the config) first.")
                return
            now = datetime.now()
            proj, shift = project_changes(changes, now, 0.0, 0)
            schedule_txt = format_schedule(total, proj, shift, now, 0.0, supply, truncated,
                                           f"Voltage schedule for run '{self.var_test_name.get().strip()}'")
            preview = "\n".join(schedule_txt.splitlines()[:28])
            if len(schedule_txt.splitlines()) > 28:
                preview += "\n  ... (full list: 'Voltage schedule...' button, and schedule.txt in the run folder)"
            if not messagebox.askokcancel(
                    "Voltage schedule - set your alarms",
                    preview + "\n\nThe run pauses at each change until you confirm the new voltage. Start the run?"):
                return

        out_dir = ensure_dir(self.var_out_dir.get().strip())
        run_name = sanitize_run_name(self.var_test_name.get(), "run")
        run_dir = create_unique_run_dir(out_dir, run_name)
        self.run_dir = run_dir
        if schedule_txt is not None:
            try:
                with open(os.path.join(run_dir, "schedule.txt"), "x", encoding="utf-8") as f:
                    f.write(schedule_txt + "\n")
            except Exception:
                log_exception("Start scheduled run: failed to write schedule.txt")

        cfg.setdefault("meta", {})
        cfg["meta"]["output_dir"] = out_dir
        cfg["meta"]["run_dir"] = run_dir
        cfg["meta"]["clock"] = clock_info()
        LOG.info(f"Run clock: {cfg['meta']['clock']}")

        with open(os.path.join(run_dir, "meta.json"), "x", encoding="utf-8") as f:
            json.dump(cfg.get("meta", {}), f, indent=2)
        with open(os.path.join(run_dir, "config.json"), "x", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        if contains_pressure_block(cfg["tests"]):
            self._pq_clear()
        # Pass video recorder to the runner
        runner = PumpTestRunner(cfg, run_dir, self.ui_events, self.video_recorder)
        video_on = [False]

        def start_video():
            # Called between t0 and the first block, so frames.csv covers the whole run
            video_on[0] = self._start_video_recording(runner)

        try:
            runner.start(before_run=start_video)
        except Exception as e:
            log_exception("Start scheduled run: runner.start failed")
            self.run_dir = None
            messagebox.showerror("Start failed", str(e))
            return

        self.active_stream = runner
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_manual_connect.config(state=tk.DISABLED)
        self.btn_manual_disconnect.config(state=tk.DISABLED)
        
        if video_on[0]:
            self.var_status.set(f"Running. Folder: {run_dir} (Video recording on)")
        else:
            self.var_status.set(f"Running. Folder: {run_dir}")

    def manual_connect(self):
        if self.active_stream is not None:
            return
        cfg = self.build_config_from_ui()
        cfg["protocol"] = cfg.get("serial", {}).get("protocol", "freq_only")

        run_dir = None
        out_dir = None
        if bool(self.var_log_manual.get()):
            out_dir = ensure_dir(self.var_out_dir.get().strip())
            run_name = sanitize_run_name(self.var_test_name.get(), "manual")
            run_dir = create_unique_run_dir(out_dir, run_name, "_MANUAL")

        if run_dir:
            cfg.setdefault("meta", {})
            cfg["meta"]["output_dir"] = out_dir
            cfg["meta"]["run_dir"] = run_dir
            cfg["meta"]["clock"] = clock_info()

            with open(os.path.join(run_dir, "meta.json"), "x", encoding="utf-8") as f:
                json.dump(cfg.get("meta", {}), f, indent=2)
            with open(os.path.join(run_dir, "config.json"), "x", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

        self.run_dir = run_dir
        sess = ManualSession(cfg, run_dir, self.ui_events, self.video_recorder)
        try:
            sess.start()
        except Exception as e:
            log_exception("Manual connect: sess.start failed")
            self.run_dir = None
            messagebox.showerror("Manual connect failed", str(e))
            return

        self.active_stream = sess
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_manual_connect.config(state=tk.DISABLED)
        self.btn_manual_disconnect.config(state=tk.NORMAL)
        self.btn_start.config(state=tk.DISABLED)
        self.var_manual_info.set("Manual connected. Use Output ON + Set freq.")
        
        # Start video recording if enabled
        if self._start_video_recording(sess):
            self.var_status.set("Manual connected. (Video recording on)")
        else:
            self.var_status.set("Manual connected.")

    def manual_disconnect(self):
        if isinstance(self.active_stream, ManualSession):
            self.stop_active()

    def stop_active(self):
        if not self.active_stream:
            return
        self.var_status.set("Stopping ...")
        self.btn_stop.config(state=tk.DISABLED)
        threading.Thread(target=self._join_active_bg, daemon=True).start()

    def _join_active_bg(self):
        """Worker thread: stop the session and the video. No Tk calls here (they
        fail off the main thread); the UI is reset by _on_session_ended."""
        run_dir = self.run_dir
        captured_data = None
        stream = self.active_stream
        try:
            # Capture data before stopping the stream
            if stream is not None:
                captured_data = stream.get_plot_data()
                stream.stop()
                stream.join(timeout=5.0)
        except Exception:
            log_exception("Stop: failed while stopping the session")
        finally:
            try:
                if self.video_recorder is not None:
                    self.video_recorder.stop_recording()
            except Exception:
                log_exception("Stop: failed to stop the video recording")
            self.active_stream = None
            self.run_dir = None
            self.ui_events.put(("session_ended", (run_dir, captured_data)))

    def _on_session_ended(self, run_dir: Optional[str], captured_data):
        """Main thread: reset the controls after a session ended."""
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_start.config(state=tk.NORMAL if self.var_mode.get() == "scheduled" else tk.DISABLED)
        self.btn_manual_connect.config(state=tk.NORMAL)
        self.btn_manual_disconnect.config(state=tk.DISABLED)
        self.var_status.set("Idle.")
        self.var_manual_info.set("Manual session not connected.")
        self.var_manual_enabled.set(False)
        self.var_video_rec_status.set("off")
        self._show_operator_banner(None)
        if run_dir and os.path.exists(os.path.join(run_dir, "pq_points.csv")) and self.pq_rows:
            try:
                self.pq_fig.savefig(os.path.join(run_dir, "pq_curve.png"), dpi=150, bbox_inches="tight")
            except Exception:
                log_exception("Session end: failed to save pq_curve.png")
        # Generate summary plot if we have run data
        if run_dir and captured_data and len(captured_data[0]) > 0:
            self.root.after(100, lambda: self._generate_summary_plot(run_dir, captured_data))
    
    def _generate_summary_plot(self, run_dir: str, data: Tuple[List[float], List[float], List[float]]):
        """Generate and save a summary plot when test completes."""
        ts, freqs, flows = data
        
        if not ts or len(ts) == 0:
            return
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Top subplot: Time vs Flow
        ax1.plot(ts, flows, 'b-', lw=1)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Flow rate', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.grid(True, alpha=0.25)
        ax1.set_title('Flow Rate vs Time')
        
        # Add secondary y-axis for frequency on top plot
        ax1_freq = ax1.twinx()
        ax1_freq.plot(ts, freqs, 'r-', lw=1, alpha=0.7)
        ax1_freq.set_ylabel('Frequency (Hz)', color='r')
        ax1_freq.tick_params(axis='y', labelcolor='r')
        
        # Bottom subplot: Time vs Frequency
        ax2.plot(ts, freqs, 'r-', lw=1)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (Hz)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.grid(True, alpha=0.25)
        ax2.set_title('Frequency vs Time')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(run_dir, "summary_plot.png")
        try:
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            self.var_status.set(f"Test complete. Plot saved to: {os.path.basename(plot_path)}")
        except Exception as e:
            log_exception(f"Summary plot: failed to save {plot_path}")
            self.var_status.set(f"Test complete. Failed to save plot: {e}")
        finally:
            plt.close(fig)

    # ---- manual actions ----
    def _manual_toggle_output(self):
        if not isinstance(self.active_stream, ManualSession):
            self.var_manual_enabled.set(False)
            return
        self.active_stream.set_output_enabled(bool(self.var_manual_enabled.get()))

    def _manual_set_freq(self):
        if isinstance(self.active_stream, ManualSession):
            self.active_stream.set_frequency(safe_int(self.var_manual_freq.get(), 0))

    def _manual_set_duty(self):
        if isinstance(self.active_stream, ManualSession):
            self.active_stream.set_duty(safe_float(self.var_manual_duty.get(), 0.0))

    # ---- UI event queue ----
    def _poll_ui_events(self):
        try:
            while True:
                kind, payload = self.ui_events.get_nowait()
                if kind == "status":
                    self.var_status.set(payload)
                elif kind == "voltage_prompt":
                    self._show_voltage_banner(payload)
                elif kind == "operator_prompt":
                    self._show_operator_banner(payload)
                elif kind == "pq_point":
                    self._add_pq_point(payload)
                elif kind == "cal_log":
                    self._cal_log(payload)
                elif kind == "cal_saved":
                    self.var_pr_cal.set(self._cal_summary())
                elif kind == "manual_output_off":
                    self.var_manual_enabled.set(False)
                elif kind == "pip_done":
                    self._on_pip_done(*payload)
                elif kind == "session_ended":
                    self._on_session_ended(*payload)
                elif kind == "error":
                    self.var_status.set(payload)
                    try:
                        messagebox.showerror("Run error", payload)
                    except Exception:
                        log_exception("UI: failed to show error dialog")
        except queue.Empty:
            pass
        self.root.after(120, self._poll_ui_events)

    def _poll_memory(self):
        """Watchdog: flush data and shed load before the process runs out of memory."""
        try:
            avail_mb = psutil.virtual_memory().available / (1024.0 * 1024.0)
            if avail_mb < MEM_CRITICAL_MB:
                # Critical: get every byte to disk, then relieve pressure.
                flush_all_loggers(sync=True)
                if self.video_recorder and self.video_recorder.is_recording():
                    self._stop_video_recording()
                    self.var_status.set(
                        f"LOW MEMORY ({avail_mb:.0f} MB free): data flushed and video recording "
                        f"stopped to protect data. Please stop the run and free memory."
                    )
                else:
                    self.var_status.set(
                        f"LOW MEMORY ({avail_mb:.0f} MB free): data flushed to disk. "
                        f"Please stop the run and free memory."
                    )
                if not self._mem_critical_warned:
                    self._mem_critical_warned = True
                    try:
                        messagebox.showwarning(
                            "Low memory",
                            f"System memory is critically low ({avail_mb:.0f} MB free).\n\n"
                            f"All collected data has been flushed to disk, and video recording "
                            f"(if any) was stopped to reduce memory use.\n\n"
                            f"Please stop the current run and close other applications to avoid a crash."
                        )
                    except Exception:
                        log_exception("Memory watchdog: failed to show warning dialog")
            elif avail_mb < MEM_WARN_MB:
                flush_all_loggers(sync=True)
                log_warning_limited("mem_warn", f"Low memory: {avail_mb:.0f} MB free; flushed data to disk.")
            else:
                self._mem_critical_warned = False
        except Exception:
            log_exception_limited("mem_poll", "Memory watchdog poll failed")
        finally:
            self.root.after(MEM_POLL_MS, self._poll_memory)

    def on_close(self):
        try:
            self.panels.save()
        except Exception:
            log_exception("App close: failed to save the panel layout")
        try:
            if self.active_stream:
                self.active_stream.stop()
                self.active_stream.join(timeout=2.0)
        except Exception:
            log_exception("App close: failed to stop active stream cleanly")
        self.root.after(200, self.root.destroy)

    def run(self):
        self.root.mainloop()


def _install_crash_handlers():
    """Flush all open data logs to disk before the process goes down for any reason."""
    _orig_excepthook = sys.excepthook
    def _hook(exc_type, exc, tb):
        try:
            flush_all_loggers(sync=True)
            LOG.error("Uncaught exception; flushed open data logs before exit.")
        except Exception:
            pass
        _orig_excepthook(exc_type, exc, tb)
    sys.excepthook = _hook

    # Threads (e.g. the daemon sampling thread) that die on an unhandled exception.
    try:
        _orig_threadhook = threading.excepthook
        def _thook(hook_args):
            try:
                flush_all_loggers(sync=True)
                LOG.error("Uncaught exception in worker thread; flushed open data logs.")
            except Exception:
                pass
            _orig_threadhook(hook_args)
        threading.excepthook = _thook
    except Exception:
        log_exception("Crash handler: failed to install threading excepthook")

    atexit.register(lambda: flush_all_loggers(sync=True))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Optional config JSON to load at startup")
    args = ap.parse_args()
    _install_crash_handlers()
    DashboardApp(initial_config_path=args.config).run()

if __name__ == "__main__":
    main()
