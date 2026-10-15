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
from datetime import datetime, timedelta
from collections import deque
from typing import Optional, Dict, Any, List, Tuple

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
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

FLUIGENT_AVAILABLE = False
try:
    from Fluigent.SDK import fgt_init, fgt_close, fgt_get_sensorValue, fgt_set_errorReportMode
    FLUIGENT_AVAILABLE = True
except Exception:
    FLUIGENT_AVAILABLE = False
    log_exception("Optional import failed: Fluigent SDK not available; Fluigent backend disabled.")

NETWORK_FLOW_AVAILABLE = False
try:
    from flow_client import FlowClient, parse_server  # shared flow server client (needs websockets)
    NETWORK_FLOW_AVAILABLE = True
except Exception:
    NETWORK_FLOW_AVAILABLE = False
    log_exception("Optional import failed: flow_client/websockets not available; network flow backend disabled.")

POWER_AVAILABLE = False
try:
    from power_ppk2 import PowerMeter, POWER_DEFAULTS, find_ppk2_ports  # PPK2 power meter (needs numpy)
    POWER_AVAILABLE = True
except Exception:
    POWER_DEFAULTS = {"enabled": False}
    log_exception("Optional import failed: power_ppk2 not available; power meter disabled.")


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
    repeat = max(1, repeat)
    if ttype == "constant":
        return repeat * (safe_float(t.get("duration_s", 0), 0.0) + settle_s)
    steps = max(1, safe_int(t.get("steps", 1), 1))
    if "step_duration_s" in t:
        step_dur = safe_float(t.get("step_duration_s"), 0.0)
    else:
        step_dur = safe_float(t.get("total_duration_s", 0), 0.0) / steps
    return repeat * steps * (step_dur + settle_s)


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

class FluigentFlowSensor(FlowSensorBase):
    def __init__(self, channel: int = 0):
        self.channel = channel
        self._opened = False

    def open(self):
        if not FLUIGENT_AVAILABLE:
            raise RuntimeError("Fluigent SDK not available.")
        fgt_init()
        try:
            fgt_set_errorReportMode("None")
        except Exception:
            log_exception("FluigentFlowSensor.open: failed to set error report mode")
        self._opened = True

    def close(self):
        if self._opened:
            try:
                fgt_close()
            except Exception:
                log_exception("FluigentFlowSensor.close: failed to close SDK")
        self._opened = False

    def read(self):
        if not self._opened:
            return False, None
        try:
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
        self.events_w = csv.writer(self.events_f); self.events_w.writerow(["t_s", "freq_set_hz", "duty_set"] + vcol)
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

    def log_event(self, t_s: float, freq: int, duty: float, v_mv: Optional[float] = None):
        with self._lock:
            if self.events_w:
                self.events_w.writerow([f"{t_s:.6f}", int(freq), f"{float(duty):.6f}"] + self._vcol(v_mv))
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
class LiveSessionBase:
    meter: Optional["PowerMeter"] = None      # PPK2 power meter when enabled

    def get_plot_data(self) -> Tuple[List[float], List[float], List[float]]:
        return [], [], []
    def stop(self): ...
    def join(self, timeout: Optional[float] = None): ...

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

            self.sensor = make_flow_sensor(self.cfg.get("flow_sensor", {}), "Manual")
            self.sensor.open()
        except Exception:
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
            self.logger.log_event(time.monotonic() - self.t0, f if enabled else 0, d)

    def set_frequency(self, hz: int):
        hz = max(0, int(hz))
        with self.lock:
            self.freq_set = hz
            enabled = self.output_enabled
            d = self.duty_set
        if enabled:
            self._set_freq_internal(hz)
        if self.logger and self.t0 is not None:
            self.logger.log_event(time.monotonic() - self.t0, hz if enabled else 0, d)

    def set_duty(self, duty: float):
        duty = clamp(float(duty), 0.0, 1.0)
        with self.lock:
            self.duty_set = duty
            enabled = self.output_enabled
            f = self.freq_set
        if enabled:
            self._set_duty_internal(duty)
        if self.logger and self.t0 is not None:
            self.logger.log_event(time.monotonic() - self.t0, f if enabled else 0, duty)

    def _sampling_loop(self):
        sr = float(self.cfg.get("flow_sensor", {}).get("sample_rate_hz", 25.0))
        sr = 10.0 if sr <= 0 else sr
        dt = 1.0 / sr
        try:
            while not self.stop_flag.is_set():
                ok, val = self.sensor.read()
                t_s = time.monotonic() - self.t0
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

    def _video_loop(self):
        """Background thread for capturing video frames with overlay."""
        video_recorder = None
        
        # Get video recorder from UI if available
        try:
            from pump_test_dashboard import DashboardApp
            # Will be set by UI
            pass
        except ImportError:
            log_warning_limited("video_loop_import", "ManualSession._video_loop: failed to import DashboardApp")
        
        while not self.stop_flag.is_set():
            if self.video_recorder and self.video_recorder.is_recording():
                frame = self.video_recorder.read_frame()
                if frame is not None:
                    t_s = time.monotonic() - self.t0
                    with self.lock:
                        f = self.freq_set if self.output_enabled else 0
                        flow_val = self.buf_flow[-1] if self.buf_flow else 0.0
                    
                    # Add overlay and record
                    overlay_frame = self.video_recorder.add_overlay(frame, t_s, f, flow_val)
                    self.video_recorder.record_frame(overlay_frame)
            
            time.sleep(1.0 / 30.0)  # ~30 fps for video

    def start(self):
        self._setup_hardware()
        self.t0 = time.monotonic()
        self.stop_flag.clear()
        if self.meter:
            self.meter.start(self.t0)
        self._emit("status", "Manual: connected. Sampling started.")
        self.t_sample = threading.Thread(target=self._sampling_loop, daemon=True)
        self.t_sample.start()
        
        # Start video recording if enabled and recorder is available
        if self.video_recorder and self.video_recorder.is_opened():
            self.t_video = threading.Thread(target=self._video_loop, daemon=True)
            self.t_video.start()

    def stop(self):
        self.stop_flag.set()

    def join(self, timeout: Optional[float] = None):
        try:
            if self.t_sample:
                self.t_sample.join(timeout=timeout)
        except Exception:
            log_exception("ManualSession.join: failed while joining sampling thread")
        self._teardown_hardware()
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

    def _nominal_begin(self, t_start: float, dur_s: float):
        with self.lock:
            self._nominal_step = (t_start, dur_s)

    def _nominal_end(self, dur_s: float):
        with self.lock:
            self._nominal_step = None
            self.nominal_done += dur_s

    def get_progress(self) -> Dict[str, Any]:
        """Snapshot for the UI timer (thread-safe)."""
        now = time.monotonic()
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
            self._wait_until(time.monotonic() + (opens - now).total_seconds())
            if self.stop_flag.is_set():
                return

        self._voltage_confirmed.clear()
        asked_wall = datetime.now()
        asked_mono = time.monotonic()
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

        wait_s = time.monotonic() - asked_mono
        with self.lock:
            self.v_set_mv = confirmed
            self.changes_done = index
        if self.meter:
            self.meter.set_voltage(confirmed)
        self.logger.log_voltage_change({
            "index": index, "t_s": f"{time.monotonic() - self.t0:.3f}", "requested_mv": mv,
            "confirmed_mv": confirmed, "asked_at": req["asked_at"],
            "confirmed_at": datetime.now().isoformat(timespec="seconds"),
            "wait_s": f"{wait_s:.1f}", "pump_while_waiting": "hold" if hold else "zero",
        })
        log_event()
        self._emit("status", f"Voltage change #{index} confirmed: {confirmed / 1000:.3f} V after "
                             f"{fmt_duration(wait_s)}" + (f"; settling {settle_s:g} s" if settle_s > 0 else ""))
        if settle_s > 0:
            t_start = time.monotonic()
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

            self.sensor = make_flow_sensor(self.cfg.get("flow_sensor", {}), "Runner")
            self.sensor.open()
        except Exception:
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
        if self.ser:
            with self._ser_lock:
                serial_send_freq(self.ser, hz, self.protocol)

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
                t_s = time.monotonic() - self.t0
                if ok and val is not None:
                    with self.lock:
                        f = self.freq_set
                        d = self.duty_set
                        v = self.v_set_mv
                        self.buf_t.append(t_s)
                        self.buf_freq.append(f)
                        self.buf_flow.append(float(val))
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

    def _video_loop(self):
        """Background thread for capturing video frames with overlay."""
        while not self.stop_flag.is_set():
            if self.video_recorder and self.video_recorder.is_recording():
                frame = self.video_recorder.read_frame()
                if frame is not None:
                    t_s = time.monotonic() - self.t0
                    with self.lock:
                        f = self.freq_set
                        flow_val = self.buf_flow[-1] if self.buf_flow else 0.0
                    
                    # Add overlay and record
                    overlay_frame = self.video_recorder.add_overlay(frame, t_s, f, flow_val)
                    self.video_recorder.record_frame(overlay_frame)
            
            time.sleep(1.0 / 30.0)  # ~30 fps for video

    def _exec_block(self, test, log_event, status_prefix: str = ""):
        """Execute one test block. Supports 'constant', 'sweep', 'voltage' (manual
        supply change, see _exec_voltage) and 'group' (a list of child blocks
        repeated 'repeat' times; repeat <= 0 loops forever until Stop)."""
        ttype = (test.get("type") or "constant").lower()
        settle_ms = max(0, int(test.get("settle_ms", 0)))

        if ttype == "voltage":
            self._exec_voltage(test, log_event)
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
                now = time.monotonic()
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
        now = time.monotonic()
        prev = self._step_deadline
        if prev is not None and 0.0 <= now - prev < 0.05:
            return prev
        return now

    def _wait_until(self, t_end: float):
        """Sleep until the monotonic deadline t_end, or until Stop."""
        while not self.stop_flag.is_set():
            remaining = t_end - time.monotonic()
            if remaining <= 0:
                return
            # Cap each sleep so Stop stays responsive, but never sleep past the
            # deadline: ms-scale group steps would otherwise stretch to 20 ms.
            time.sleep(min(0.02, remaining))

    def _run_tests(self):
        tests = self.cfg.get("tests", [])
        break_between = float(self.cfg.get("break_seconds_between_tests", 0.0))

        def log_event():
            self.logger.log_event(time.monotonic() - self.t0, self.freq_set, self.duty_set, self.v_set_mv)

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
                    t_break = time.monotonic()
                    self._nominal_begin(t_break, break_between)
                    t_end = t_break + break_between
                    while time.monotonic() < t_end and not self.stop_flag.is_set():
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
                self.t_end = time.monotonic()
                self.voltage_request = None
            self.stop_flag.set()

    def start(self):
        self._setup_hardware()
        self.t0 = time.monotonic()
        self.stop_flag.clear()
        if self.meter:
            self.meter.start(self.t0)
        self.t_sample = threading.Thread(target=self._sampling_loop, daemon=True)
        self.t_run = threading.Thread(target=self._run_tests, daemon=True)
        self.t_sample.start()
        self.t_run.start()
        
        # Start video recording if enabled and recorder is available
        if self.video_recorder and self.video_recorder.is_opened():
            self.t_video = threading.Thread(target=self._video_loop, daemon=True)
            self.t_video.start()

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

    def get_plot_data(self):
        with self.lock:
            return list(self.buf_t), list(self.buf_freq), list(self.buf_flow)


# ---------------- video recording ----------------
class VideoRecorder:
    """USB camera video recorder with overlay support and shared frame buffer."""
    
    def __init__(self):
        self.cap = None
        self.writer = None
        self.recording = False
        self.output_path = None
        self.base_output_path = None
        self.frame_width = 1920
        self.frame_height = 1080
        self.fps = 30.0
        self.chunk_duration_s = 60.0
        self.chunk_index = 0
        self.chunk_start_t = 0.0
        self.chunk_paths: List[str] = []
        
        self._writer_lock = threading.Lock()
        self._cap_lock = threading.Lock()
        self._last_frame = None
        self._last_frame_lock = threading.Lock()

    def _release_writer_locked(self):
        if self.writer:
            self.writer.release()
            self.writer = None

    def _build_chunk_path(self, chunk_index: int) -> Optional[str]:
        if not self.base_output_path:
            return None
        stem, ext = os.path.splitext(self.base_output_path)
        if not ext:
            ext = ".mp4"
        return f"{stem}_part{chunk_index:04d}{ext}"

    def _open_next_chunk_locked(self) -> bool:
        self.chunk_index += 1
        chunk_path = self._build_chunk_path(self.chunk_index)
        if not chunk_path:
            return False

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            chunk_path,
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height),
        )
        if not writer.isOpened():
            writer.release()
            log_error(f"VideoRecorder: failed to open video chunk {chunk_path}")
            return False

        self._release_writer_locked()
        self.writer = writer
        self.output_path = chunk_path
        self.chunk_paths.append(chunk_path)
        self.chunk_start_t = time.monotonic()
        return True
        
    def open(self, camera_index: int = 0) -> bool:
        """Open a USB camera."""
        if not CV2_AVAILABLE:
            log_warning("VideoRecorder.open: OpenCV not available; cannot open camera.")
            return False
        with self._cap_lock:
            try:
                self.cap = cv2.VideoCapture(camera_index)
                if not self.cap.isOpened():
                    log_error(f"VideoRecorder.open: failed to open camera index {camera_index}")
                    self.cap = None
                    return False
                # Get actual frame dimensions; fall back to 1080p if unavailable
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if width > 0 and height > 0:
                    self.frame_width = width
                    self.frame_height = height
                else:
                    self.frame_width = 1920
                    self.frame_height = 1080
                    log_warning(
                        f"VideoRecorder.open: camera reported invalid resolution {width}x{height}; "
                        f"using fallback {self.frame_width}x{self.frame_height}"
                    )
                return True
            except Exception:
                log_exception(f"VideoRecorder.open: exception opening camera index {camera_index}")
                self.cap = None
                return False
    
    def close(self):
        """Close the camera and writer."""
        with self._writer_lock:
            self._release_writer_locked()
            self.recording = False
            self.output_path = None
            self.base_output_path = None
            self.chunk_paths = []
            self.chunk_index = 0
            self.chunk_start_t = 0.0
        
        with self._cap_lock:
            if self.cap:
                self.cap.release()
                self.cap = None
        
        with self._last_frame_lock:
            self._last_frame = None
    
    def start_recording(self, output_path: str, chunk_duration_s: float = 60.0) -> bool:
        """Start recording video to chunked files."""
        if not self.is_opened():
            log_error("VideoRecorder.start_recording: camera not opened")
            return False
        try:
            with self._writer_lock:
                self._release_writer_locked()
                self.recording = False
                self.base_output_path = output_path
                self.output_path = None
                self.chunk_duration_s = max(float(chunk_duration_s), 1.0)
                self.chunk_paths = []
                self.chunk_index = 0
                self.chunk_start_t = 0.0
                if not self._open_next_chunk_locked():
                    self.base_output_path = None
                    return False
                self.recording = True
                return True
        except Exception:
            log_exception(f"VideoRecorder.start_recording: exception creating writer for {output_path}")
            with self._writer_lock:
                self._release_writer_locked()
                self.base_output_path = None
                self.output_path = None
                self.recording = False
            return False
    
    def stop_recording(self):
        """Stop recording."""
        with self._writer_lock:
            self.recording = False
            self._release_writer_locked()
            self.output_path = None
            self.base_output_path = None
            self.chunk_paths = []
            self.chunk_index = 0
            self.chunk_start_t = 0.0
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        with self._writer_lock:
            return self.recording and self.writer is not None
    
    def is_opened(self) -> bool:
        """Check if camera is open."""
        with self._cap_lock:
            return self.cap is not None and self.cap.isOpened()
    
    def read_frame(self, fresh: bool = True) -> Optional["np.ndarray"]:
        """
        Read a frame from the camera.
        If fresh is True, grabs a new frame from hardware and caches it.
        If fresh is False, returns the last cached frame.
        """
        if fresh:
            with self._cap_lock:
                if self.cap:
                    ret, frame = self.cap.read()
                    if ret:
                        with self._last_frame_lock:
                            self._last_frame = frame
                        return frame
            return None
        else:
            with self._last_frame_lock:
                return self._last_frame.copy() if self._last_frame is not None else None
    
    def add_overlay(self, frame, t_s: float, freq: float, flow: float) -> "np.ndarray":
        """Add overlay text to frame."""
        try:
            # Add semi-transparent overlay rectangle
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 5), (250, 90), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Add text
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
    
    def record_frame(self, frame) -> bool:
        """Record a frame if recording is active."""
        with self._writer_lock:
            if not self.writer or not self.recording:
                return False

            if self.chunk_start_t and (time.monotonic() - self.chunk_start_t) >= self.chunk_duration_s:
                if not self._open_next_chunk_locked():
                    self.recording = False
                    self._release_writer_locked()
                    return False

            self.writer.write(frame)
            return True


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

        # cache_frame_data=False prevents matplotlib from retaining returned artists
        # for every frame of an unbounded live animation (a known slow memory growth).
        self.ani = animation.FuncAnimation(self.fig, self._update_plot, interval=150,
                                           blit=False, cache_frame_data=False)
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
        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        # Run meta
        meta = ttk.LabelFrame(top, text="Run details & storage")
        meta.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        self.var_test_name = tk.StringVar(value="Pump Test")
        self.var_operator  = tk.StringVar(value="")
        self.var_out_dir   = tk.StringVar(value=os.path.abspath("./pump_runs"))
        self.var_log_manual = tk.BooleanVar(value=False)
        self.txt_desc      = tk.Text(meta, height=4, width=55)
        self._default_test_name = self.var_test_name.get()
        self._default_operator = self.var_operator.get()
        self._default_desc = ""

        ttk.Label(meta, text="Name:").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(meta, textvariable=self.var_test_name, width=40).grid(row=0, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(meta, text="Operator:").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(meta, textvariable=self.var_operator, width=40).grid(row=1, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(meta, text="Output folder:").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(meta, textvariable=self.var_out_dir, width=40).grid(row=2, column=1, sticky="w", padx=4, pady=2)
        ttk.Button(meta, text="Browse...", command=self.pick_out_dir).grid(row=2, column=2, padx=4, pady=2)

        ttk.Label(meta, text="Description:").grid(row=3, column=0, sticky="nw", padx=4, pady=2)
        self.txt_desc.grid(row=3, column=1, columnspan=2, sticky="we", padx=4, pady=2)

        ttk.Checkbutton(meta, text="Log manual sessions to disk", variable=self.var_log_manual)\
            .grid(row=4, column=1, sticky="w", padx=4, pady=(2, 6))

        # Hardware
        hw = ttk.LabelFrame(top, text="Hardware & mode")
        hw.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.var_cfg_file = tk.StringVar(value="")
        self.var_port     = tk.StringVar(value="COM22")
        self.var_baud     = tk.StringVar(value="115200")
        self.var_protocol = tk.StringVar(value="freq_only")

        self.var_sensor_backend = tk.StringVar(value="fluigent" if FLUIGENT_AVAILABLE else "dummy")
        self.var_sensor_channel = tk.StringVar(value="0")
        self.var_flow_server    = tk.StringVar(value="localhost:8765")
        self.var_sample_rate    = tk.StringVar(value="25")
        self.var_break_between  = tk.StringVar(value="0")

        self.var_mode = tk.StringVar(value="scheduled")

        ttk.Label(hw, text="Config file:").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(hw, textvariable=self.var_cfg_file, width=38).grid(row=0, column=1, sticky="w", padx=4, pady=2)
        ttk.Button(hw, text="Load", command=self.pick_and_load_config).grid(row=0, column=2, padx=4, pady=2)
        ttk.Button(hw, text="Save As", command=self.save_config_as).grid(row=0, column=3, padx=4, pady=2)

        ttk.Label(hw, text="Serial port:").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(hw, textvariable=self.var_port, width=12).grid(row=1, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(hw, text="Baud:").grid(row=1, column=2, sticky="e", padx=4, pady=2)
        ttk.Entry(hw, textvariable=self.var_baud, width=10).grid(row=1, column=3, sticky="w", padx=4, pady=2)

        ttk.Label(hw, text="Protocol:").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        proto = ttk.Combobox(hw, state="readonly", width=10, values=["freq_only", "freq_duty"], textvariable=self.var_protocol)
        proto.grid(row=2, column=1, sticky="w", padx=4, pady=2)
        proto.bind("<<ComboboxSelected>>", lambda *_: self._refresh_manual_duty_state())

        ttk.Label(hw, text="Flow backend:").grid(row=3, column=0, sticky="w", padx=4, pady=2)
        ttk.Combobox(hw, state="readonly", width=10, values=["fluigent", "dummy", "network"], textvariable=self.var_sensor_backend)\
            .grid(row=3, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(hw, text="Channel:").grid(row=3, column=2, sticky="e", padx=4, pady=2)
        ttk.Entry(hw, textvariable=self.var_sensor_channel, width=10).grid(row=3, column=3, sticky="w", padx=4, pady=2)

        ttk.Label(hw, text="Sample rate (Hz):").grid(row=4, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(hw, textvariable=self.var_sample_rate, width=10).grid(row=4, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(hw, text="Break between blocks (s):").grid(row=4, column=2, sticky="e", padx=4, pady=2)
        ttk.Entry(hw, textvariable=self.var_break_between, width=10).grid(row=4, column=3, sticky="w", padx=4, pady=2)

        ttk.Label(hw, text="Flow server:").grid(row=5, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(hw, textvariable=self.var_flow_server, width=24).grid(row=5, column=1, sticky="w", padx=4, pady=2)

        mode_row = ttk.Frame(hw)
        mode_row.grid(row=6, column=0, columnspan=4, sticky="w", padx=4, pady=(8, 2))
        ttk.Label(mode_row, text="Mode:").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_row, text="Scheduled tests", variable=self.var_mode, value="scheduled", command=self._update_mode_ui)\
            .pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(mode_row, text="Manual control (testing OFF)", variable=self.var_mode, value="manual", command=self._update_mode_ui)\
            .pack(side=tk.LEFT, padx=6)

        # Buttons row
        btns = ttk.Frame(self.root)
        btns.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self.btn_start = ttk.Button(btns, text="Start Run", command=self.start_scheduled_run)
        self.btn_stop  = ttk.Button(btns, text="Stop", command=self.stop_active, state=tk.DISABLED)
        self.btn_manual_connect = ttk.Button(btns, text="Connect Manual", command=self.manual_connect)
        self.btn_manual_disconnect = ttk.Button(btns, text="Disconnect Manual", command=self.manual_disconnect, state=tk.DISABLED)

        self.btn_start.pack(side=tk.LEFT, padx=4)
        self.btn_stop.pack(side=tk.LEFT, padx=4)
        self.btn_manual_connect.pack(side=tk.LEFT, padx=(18, 4))
        self.btn_manual_disconnect.pack(side=tk.LEFT, padx=4)

        self.var_status = tk.StringVar(value="Idle.")
        ttk.Label(btns, textvariable=self.var_status).pack(side=tk.LEFT, padx=12)

        self._build_progress_row()

        # Main split (responsive, no scrollbars)
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        left = ttk.Frame(main, width=420)
        left.pack_propagate(False)
        main.add(left, weight=0)
        self._build_config_builder(left)

        right = ttk.Frame(main)
        main.add(right, weight=1)

        right_pane = ttk.PanedWindow(right, orient=tk.VERTICAL)
        right_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.plot_frame = ttk.LabelFrame(right_pane, text="Live plots")
        right_pane.add(self.plot_frame, weight=3)

        controls = ttk.Frame(right_pane)
        right_pane.add(controls, weight=1)

        controls.columnconfigure(0, weight=1)
        controls.rowconfigure(3, weight=1)

        self.manual_frame = ttk.LabelFrame(controls, text="Manual control")
        self.manual_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self._build_manual_panel(self.manual_frame)

        self.power_frame = ttk.LabelFrame(controls, text="Power meter (PPK2 as ammeter: bench supply -> PPK2 -> pump driver)")
        self.power_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self._build_power_panel(self.power_frame)

        # Initialize video recorder
        self.video_recorder = None
        self.video_recording = False
        self.var_video_enabled = tk.BooleanVar(value=False)
        self.var_video_camera_index = tk.StringVar(value="0")
        self.var_video_status = tk.StringVar(value="Camera: Not connected")

        # Build video controls and preview
        self._build_video_controls(controls).grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self._build_video_preview(controls).grid(row=3, column=0, sticky="nsew")

        self._update_mode_ui()
        self._refresh_manual_duty_state()

        # Start video preview if camera is connected
        self._init_video()

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
        ttk.Label(r0, text="Bin (ms):").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Entry(r0, textvariable=self.var_pw_bin, width=5).pack(side=tk.LEFT)

        r1 = ttk.Frame(parent); r1.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2)
        ttk.Label(r1, text="Supply voltage at start (V):").pack(side=tk.LEFT)
        ttk.Entry(r1, textvariable=self.var_supply_v, width=7).pack(side=tk.LEFT, padx=4)
        ttk.Label(r1, text="Bench supply limit (mA):").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Entry(r1, textvariable=self.var_supply_limit, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(r1, text="Stop if current ≥ (mA, blank = 95% of limit):").pack(side=tk.LEFT, padx=(12, 2))
        ttk.Entry(r1, textvariable=self.var_pw_watch, width=6).pack(side=tk.LEFT, padx=4)

        r2 = ttk.Frame(parent); r2.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(2, 6))
        self.lbl_pw_live = tk.Label(r2, textvariable=self.var_pw_live, anchor="w")
        self.lbl_pw_live.pack(side=tk.LEFT)
        self._pw_live_fg = self.lbl_pw_live.cget("fg")

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
        self.root.after(1000, self._tick_progress)

    def _update_progress_ui(self):
        now = datetime.now()
        runner = self._active_runner()
        lead_s = 60.0 * safe_float(self._supply_cfg().get("lead_min", 10), 10.0)

        if runner is None:
            self._show_voltage_banner(None)
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
        if req and time.monotonic() - self._last_bell >= 60.0:
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
        waited = time.monotonic() - req["asked_mono"]
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
        video_frame = ttk.LabelFrame(parent, text="USB Camera (optional)")
        self.video_controls_frame = video_frame
        
        row1 = ttk.Frame(video_frame)
        row1.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        
        ttk.Checkbutton(row1, text="Enable video recording", variable=self.var_video_enabled)\
            .pack(side=tk.LEFT, padx=(0, 8))
        
        ttk.Label(row1, text="Camera index:").pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.var_video_camera_index, width=6).pack(side=tk.LEFT, padx=4)
        
        self.btn_video_connect = ttk.Button(row1, text="Connect Camera", command=self._connect_camera)
        self.btn_video_connect.pack(side=tk.LEFT, padx=8)
        
        self.btn_video_disconnect = ttk.Button(row1, text="Disconnect", command=self._disconnect_camera, state=tk.DISABLED)
        self.btn_video_disconnect.pack(side=tk.LEFT, padx=4)
        
        row2 = ttk.Frame(video_frame)
        row2.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 4))
        
        ttk.Label(row2, textvariable=self.var_video_status).pack(side=tk.LEFT)
        ttk.Label(row2, text="  |  Recording: off").pack(side=tk.LEFT)
        
        # Store video label for updating recording status
        self.var_video_rec_status = tk.StringVar(value="off")
        self.lbl_video_rec = ttk.Label(row2, textvariable=self.var_video_rec_status)
        self.lbl_video_rec.pack(side=tk.LEFT, padx=4)
        return video_frame

    def _init_video(self):
        """Initialize video recorder."""
        if not CV2_AVAILABLE:
            self.var_video_status.set("Camera: OpenCV not available (pip install opencv-python)")
            log_warning("Video init: OpenCV not available; camera disabled.")
            return
        
        # Create video recorder instance
        self.video_recorder = VideoRecorder()
        
        # Try to auto-connect to camera 0
        camera_idx = safe_int(self.var_video_camera_index.get(), 0)
        if self.video_recorder.open(camera_idx):
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
        
        camera_idx = safe_int(self.var_video_camera_index.get(), 0)
        if self.video_recorder.open(camera_idx):
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

    def _start_video_recording(self):
        """Start video recording for current run."""
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

        # Start recording in chunked files
        video_path = os.path.join(self.run_dir, "video_recording.mp4")
        if self.video_recorder.start_recording(video_path, chunk_duration_s=60.0):
            self.var_video_rec_status.set("recording")
            return True
        else:
            log_error(f"Video recording: failed to start recording at {video_path}")
            messagebox.showerror("Video", "Failed to start video recording")
            return False

    def _stop_video_recording(self):
        """Stop video recording."""
        if self.video_recorder and self.video_recorder.is_recording():
            self.video_recorder.stop_recording()
            self.var_video_rec_status.set("off")

    def _build_video_preview(self, parent):
        """Build video preview panel to show real-time camera feed."""
        self.video_preview_frame = ttk.LabelFrame(parent, text="Camera Preview")

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
            # Use cached frame from the recorder to avoid thread competition
            # If recording is OFF, we need to grab a fresh frame here
            is_recording = self.video_recorder.is_recording()
            frame = self.video_recorder.read_frame(fresh=not is_recording)
            
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
            w = min(1250, int(sw * 0.95))
            h = min(840, int(sh * 0.90))
            x = max(0, (sw - w) // 2)
            y = max(0, (sh - h) // 2)
            self.root.geometry(f"{w}x{h}+{x}+{y}")
            self.root.minsize(900, 640)
        except Exception:
            log_exception("UI geometry: failed to set default geometry")
            self.root.geometry("1250x840")

    def _build_config_builder(self, parent):
        listf = ttk.LabelFrame(parent, text="Config builder (test blocks)")
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
        ttk.Button(actions, text="Duplicate", command=self.duplicate_selected_block).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions, text="Remove", command=self.remove_selected_block).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions, text="Move Up", command=lambda: self.move_selected_block(-1)).pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text="Move Down", command=lambda: self.move_selected_block(+1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions, text="Clear All", command=self.clear_all_blocks).pack(side=tk.RIGHT, padx=2)

        actions2 = ttk.Frame(listf)
        actions2.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 6))
        ttk.Button(actions2, text="Group Selected", command=self.group_selected_blocks).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions2, text="Ungroup", command=self.ungroup_selected_block).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions2, text="Add Voltage Step", command=self.add_voltage_block).pack(side=tk.LEFT, padx=(14, 2))
        ttk.Label(listf, wraplength=420, justify=tk.LEFT,
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

        ttk.Button(editf, text="Apply changes", command=self.apply_edit_to_selected)\
            .pack(side=tk.TOP, padx=6, pady=(6, 8))

        self.var_edit_type.trace_add("write", lambda *_: self._refresh_editor_visibility())
        self._refresh_editor_visibility()

    def _refresh_editor_visibility(self):
        ttype = self.var_edit_type.get().strip().lower()
        self.edit_constant.pack_forget()
        self.edit_sweep.pack_forget()
        self.edit_group.pack_forget()
        self.edit_voltage.pack_forget()
        if ttype == "constant":
            self.edit_constant.pack(side=tk.TOP, fill=tk.X, padx=6, pady=2)
        elif ttype == "group":
            self.edit_group.pack(side=tk.TOP, fill=tk.X, padx=6, pady=2)
        elif ttype == "voltage":
            self.edit_voltage.pack(side=tk.TOP, fill=tk.X, padx=6, pady=2)
        else:
            self.edit_sweep.pack(side=tk.TOP, fill=tk.X, padx=6, pady=2)

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

        ttk.Label(row, text="Duty (0..1):").pack(side=tk.LEFT, padx=(16, 2))
        self.ent_manual_duty = ttk.Entry(row, textvariable=self.var_manual_duty, width=8)
        self.ent_manual_duty.pack(side=tk.LEFT, padx=6)
        self.btn_set_duty = ttk.Button(row, text="Set duty", command=self._manual_set_duty)
        self.btn_set_duty.pack(side=tk.LEFT, padx=4)

        info = ttk.Frame(parent)
        info.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 8))
        self.var_manual_info = tk.StringVar(value="Manual session not connected.")
        ttk.Label(info, textvariable=self.var_manual_info).pack(side=tk.LEFT)

    def _build_plot(self):
        self.fig = plt.figure(figsize=(8.2, 5.2))
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)

        self.line_freq, = self.ax1.plot([], [], lw=1)
        self.line_flow, = self.ax2.plot([], [], lw=1)
        # Supply current from the PPK2 on a second y-axis of the frequency plot (hidden without a meter)
        self.ax1_cur = self.ax1.twinx()
        self.line_cur, = self.ax1_cur.plot([], [], lw=0.8, color="tab:red")
        self.ax1_cur.set_ylabel("mA", color="tab:red")
        self.ax1_cur.tick_params(axis="y", labelcolor="tab:red")
        self.ax1_cur.set_visible(False)

        self.ax1.set_title("Frequency setpoint (Hz)")
        self.ax1.set_xlabel("t (s)")
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
        
        ttk.Label(zoom_frame, text="  |  ").pack(side=tk.LEFT)
        
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
    def _update_plot(self, _frame):
        if not self.active_stream:
            return self.line_freq, self.line_flow
        ts, freqs, flows = self.active_stream.get_plot_data()
        if not ts:
            return self.line_freq, self.line_flow

        self.line_freq.set_data(ts, freqs)
        self.line_flow.set_data(ts, flows)

        tmax = ts[-1]
        window = self._get_time_window()
        tmin = max(0.0, tmax - window)
        self.ax1.set_xlim(tmin, tmin + window)
        self.ax2.set_xlim(tmin, tmin + window)

        pt, pi = self.active_stream.get_power_plot_data()
        if pt:
            self.line_cur.set_data(pt, pi)
            if not self.ax1_cur.get_visible():
                self.ax1_cur.set_visible(True)
                self.ax1.set_title("Frequency setpoint (Hz) and supply current (mA, red)")
            vis = [i for t, i in zip(pt, pi) if t >= tmin]
            if vis:
                lo, hi = min(vis), max(vis)
                pad = max(0.05 * (hi - lo), 0.1)
                self.ax1_cur.set_ylim(lo - pad, hi + pad)
        elif self.ax1_cur.get_visible():
            self.line_cur.set_data([], [])
            self.ax1_cur.set_visible(False)
            self.ax1.set_title("Frequency setpoint (Hz)")

        # Apply user-defined Y ranges or auto-scale
        if self._user_freq_y_range is not None:
            self.ax1.set_ylim(self._user_freq_y_range)
        else:
            try:
                fmin, fmax = min(freqs), max(freqs)
                if fmin == fmax:
                    fmin -= 1; fmax += 1
                self.ax1.set_ylim(fmin, fmax)
            except Exception:
                log_exception_limited("plot_freq_ylim", "Plot update: failed to auto-scale frequency Y range")

        if self._user_flow_y_range is not None:
            self.ax2.set_ylim(self._user_flow_y_range)
        else:
            try:
                ymin, ymax = min(flows), max(flows)
                if ymin == ymax:
                    ymin -= 0.1; ymax += 0.1
                self.ax2.set_ylim(ymin, ymax)
            except Exception:
                log_exception_limited("plot_flow_ylim", "Plot update: failed to auto-scale flow Y range")

        return self.line_freq, self.line_flow

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

        meta = self.cfg.get("meta", {})
        if not isinstance(meta, dict):
            meta = {}
        self.var_test_name.set(meta.get("name", self._default_test_name))
        self.var_operator.set(meta.get("operator", self._default_operator))
        self.txt_desc.delete("1.0", tk.END)
        self.txt_desc.insert("1.0", meta.get("description", self._default_desc))

        self.cfg.setdefault("tests", [])
        self.refresh_blocks_list()
        self.var_status.set(f"Loaded config: {os.path.basename(path)}")
        self._refresh_manual_duty_state()

    def build_config_from_ui(self) -> Dict[str, Any]:
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
                "created": iso_now(),
                "output_dir": self.var_out_dir.get().strip(),
            },
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
        self.cmb_edit_type.configure(values=["group"] if ttype == "group" else ["constant", "sweep", "voltage"])
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

        with open(os.path.join(run_dir, "meta.json"), "x", encoding="utf-8") as f:
            json.dump(cfg.get("meta", {}), f, indent=2)
        with open(os.path.join(run_dir, "config.json"), "x", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        # Pass video recorder to the runner
        runner = PumpTestRunner(cfg, run_dir, self.ui_events, self.video_recorder)
        try:
            runner.start()
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
        
        # Start video recording if enabled
        if self._start_video_recording():
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
        if self._start_video_recording():
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
        run_dir = self.run_dir
        captured_data = None
        
        try:
            # Capture data before stopping the stream
            if self.active_stream is not None:
                captured_data = self.active_stream.get_plot_data()
            self.active_stream.stop()
            self.active_stream.join(timeout=5.0)
        finally:
            self.active_stream = None
            self.run_dir = None
            self.btn_stop.config(state=tk.DISABLED)
            self.btn_start.config(state=tk.NORMAL if self.var_mode.get() == "scheduled" else tk.DISABLED)
            self.btn_manual_connect.config(state=tk.NORMAL)
            self.btn_manual_disconnect.config(state=tk.DISABLED)
            self.var_status.set("Idle.")
            self.var_manual_info.set("Manual session not connected.")
            self.var_manual_enabled.set(False)
        
        # Stop video recording if active
        self._stop_video_recording()
        
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
                elif kind == "manual_output_off":
                    self.var_manual_enabled.set(False)
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
