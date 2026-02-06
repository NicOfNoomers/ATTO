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
   - Edit selected block
   - Duplicate / Remove
   - Move Up / Move Down (organize order)
   - Save config JSON (compatible with tests[] schema)

Serial protocol:
- freq_only (default): sends "<int_hz>\n"
- freq_duty (optional): sends "F <hz>\n" and "D <duty>\n"

Flow sensor:
- Fluigent SDK backend if available, otherwise dummy generator for UI testing.
"""

from __future__ import annotations

import os
import json
import time
import csv
import math
import threading
import queue
from datetime import datetime
from collections import deque
from typing import Optional, Dict, Any, List, Tuple

# ---- optional deps ----
try:
    import serial  # pyserial
except Exception:
    serial = None

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


# ---------------- util ----------------
def iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def safe_int(s: str, default: int) -> int:
    try:
        return int(float(str(s).strip()))
    except Exception:
        return default

def safe_float(s: str, default: float) -> float:
    try:
        return float(str(s).strip())
    except Exception:
        return default

def linspace_int(a: int, b: int, n: int) -> List[int]:
    if n <= 1:
        return [int(round(a))]
    step = (b - a) / (n - 1)
    return [int(round(a + i * step)) for i in range(n)]


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
        pass
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
            pass
        self._opened = True

    def close(self):
        if self._opened:
            try:
                fgt_close()
            except Exception:
                pass
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


# ---------------- logging ----------------
class DataLogger:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.events_f = None
        self.flow_f = None
        self.merged_f = None
        self.events_w = None
        self.flow_w = None
        self.merged_w = None

    def open(self):
        self.events_f = open(os.path.join(self.run_dir, "events.csv"), "w", newline="", encoding="utf-8")
        self.flow_f   = open(os.path.join(self.run_dir, "flow.csv"), "w", newline="", encoding="utf-8")
        self.merged_f = open(os.path.join(self.run_dir, "merged.csv"), "w", newline="", encoding="utf-8")
        self.events_w = csv.writer(self.events_f); self.events_w.writerow(["t_s", "freq_set_hz", "duty_set"])
        self.flow_w   = csv.writer(self.flow_f);   self.flow_w.writerow(["t_s", "flow"])
        self.merged_w = csv.writer(self.merged_f); self.merged_w.writerow(["t_s", "freq_set_hz", "duty_set", "flow"])

    def close(self):
        for f in (self.events_f, self.flow_f, self.merged_f):
            try:
                if f: f.close()
            except Exception:
                pass

    def log_event(self, t_s: float, freq: int, duty: float):
        if self.events_w:
            self.events_w.writerow([f"{t_s:.6f}", int(freq), f"{float(duty):.6f}"])

    def log_flow(self, t_s: float, flow: float):
        if self.flow_w:
            self.flow_w.writerow([f"{t_s:.6f}", f"{float(flow):.6f}"])

    def log_merged(self, t_s: float, freq: int, duty: float, flow: float):
        if self.merged_w:
            self.merged_w.writerow([f"{t_s:.6f}", int(freq), f"{float(duty):.6f}", f"{float(flow):.6f}"])


# ---------------- runtime sessions ----------------
class LiveSessionBase:
    def get_plot_data(self) -> Tuple[List[float], List[float], List[float]]:
        return [], [], []
    def stop(self): ...
    def join(self, timeout: Optional[float] = None): ...


class ManualSession(LiveSessionBase):
    """Manual real-time control session."""
    def __init__(self, cfg: Dict[str, Any], run_dir: Optional[str], ui_events: "queue.Queue[Tuple[str, Any]]"):
        self.cfg = cfg
        self.run_dir = run_dir
        self.ui_events = ui_events
        self.stop_flag = threading.Event()

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

    def _emit(self, kind: str, msg: str):
        self.ui_events.put((kind, msg))

    def _setup_hardware(self):
        port = self.cfg.get("port", "COM22")
        baud = int(self.cfg.get("baud", 115200))
        timeout_s = float(self.cfg.get("read_timeout_s", 0.2))

        if bool(self.cfg.get("dry_run", False)):
            self.ser = None
            self._emit("status", "Manual: dry_run enabled (no serial).")
        else:
            self._emit("status", f"Manual: opening serial {port} @ {baud} ...")
            self.ser = serial_open(port, baud, timeout_s)

        flow_cfg = self.cfg.get("flow_sensor", {})
        backend = (flow_cfg.get("backend") or ("fluigent" if FLUIGENT_AVAILABLE else "dummy")).lower()
        if backend == "fluigent":
            if not FLUIGENT_AVAILABLE:
                raise RuntimeError("Manual: Fluigent backend requested but SDK not available.")
            self.sensor = FluigentFlowSensor(channel=int(flow_cfg.get("channel", 0)))
        else:
            self.sensor = DummyFlowSensor()
        self.sensor.open()

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
            pass
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass
        try:
            self.sensor.close()
        except Exception:
            pass
        try:
            if self.logger:
                self.logger.close()
        except Exception:
            pass

    def _set_freq_internal(self, hz: int):
        hz = int(hz)
        with self.lock:
            self.freq_set = hz
        if self.ser:
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
            time.sleep(dt)

    def start(self):
        self._setup_hardware()
        self.t0 = time.monotonic()
        self.stop_flag.clear()
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
            pass
        self._teardown_hardware()
        self._emit("status", "Manual: disconnected.")

    def get_plot_data(self):
        with self.lock:
            return list(self.buf_t), list(self.buf_freq), list(self.buf_flow)


class PumpTestRunner(LiveSessionBase):
    """Scheduled runner based on cfg['tests'] blocks (constant + sweep)."""
    def __init__(self, cfg: Dict[str, Any], run_dir: str, ui_events: "queue.Queue[Tuple[str, Any]]"):
        self.cfg = cfg
        self.run_dir = run_dir
        self.ui_events = ui_events
        self.stop_flag = threading.Event()

        self.protocol = (cfg.get("protocol") or cfg.get("serial", {}).get("protocol") or "freq_only")
        self.ser = None
        self.sensor: FlowSensorBase = DummyFlowSensor()

        self.freq_set = 0
        self.duty_set = 0.0

        self.lock = threading.Lock()
        self.buf_t = deque(maxlen=5000)
        self.buf_freq = deque(maxlen=5000)
        self.buf_flow = deque(maxlen=5000)

        self.logger = DataLogger(run_dir)
        self.t0 = None
        self.t_sample = None
        self.t_run = None

    def _emit(self, kind: str, msg: str):
        self.ui_events.put((kind, msg))

    def _setup_hardware(self):
        port = self.cfg.get("port", "COM22")
        baud = int(self.cfg.get("baud", 115200))
        timeout_s = float(self.cfg.get("read_timeout_s", 0.2))

        if bool(self.cfg.get("dry_run", False)):
            self.ser = None
            self._emit("status", "Runner: dry_run enabled (no serial).")
        else:
            self._emit("status", f"Runner: opening serial {port} @ {baud} ...")
            self.ser = serial_open(port, baud, timeout_s)

        flow_cfg = self.cfg.get("flow_sensor", {})
        backend = (flow_cfg.get("backend") or ("fluigent" if FLUIGENT_AVAILABLE else "dummy")).lower()
        if backend == "fluigent":
            if not FLUIGENT_AVAILABLE:
                raise RuntimeError("Runner: Fluigent backend requested but SDK not available.")
            self.sensor = FluigentFlowSensor(channel=int(flow_cfg.get("channel", 0)))
        else:
            self.sensor = DummyFlowSensor()
        self.sensor.open()

        self.logger.open()
        self._set_freq(0)
        self._set_duty(0.0)

    def _teardown_hardware(self):
        try:
            if bool(self.cfg.get("send_zero_on_exit", True)):
                self._set_freq(0)
                self._set_duty(0.0)
        except Exception:
            pass
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass
        try:
            self.sensor.close()
        except Exception:
            pass
        try:
            self.logger.close()
        except Exception:
            pass

    def _set_freq(self, hz: int):
        hz = int(hz)
        with self.lock:
            self.freq_set = hz
        if self.ser:
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
        while not self.stop_flag.is_set():
            ok, val = self.sensor.read()
            t_s = time.monotonic() - self.t0
            if ok and val is not None:
                with self.lock:
                    f = self.freq_set
                    d = self.duty_set
                    self.buf_t.append(t_s)
                    self.buf_freq.append(f)
                    self.buf_flow.append(float(val))
                self.logger.log_flow(t_s, float(val))
                self.logger.log_merged(t_s, f, d, float(val))
            time.sleep(dt)

    def _run_tests(self):
        tests = self.cfg.get("tests", [])
        break_between = float(self.cfg.get("break_seconds_between_tests", 0.0))

        def log_event():
            self.logger.log_event(time.monotonic() - self.t0, self.freq_set, self.duty_set)

        for i, test in enumerate(tests, 1):
            if self.stop_flag.is_set():
                break

            ttype = (test.get("type") or "constant").lower()
            repeat = max(1, int(test.get("repeat", 1)))
            settle_ms = max(0, int(test.get("settle_ms", 0)))

            self._emit("status", f"Running block {i}/{len(tests)}: {ttype} x{repeat}")

            for _ in range(repeat):
                if self.stop_flag.is_set():
                    break

                if ttype == "constant":
                    hz = int(test["frequency_hz"])
                    dur = float(test["duration_s"])
                    self._set_freq(hz)
                    if settle_ms:
                        time.sleep(settle_ms / 1000.0)
                    log_event()
                    t_end = time.monotonic() + dur
                    while time.monotonic() < t_end and not self.stop_flag.is_set():
                        time.sleep(0.02)

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
                        self._set_freq(hz)
                        if settle_ms:
                            time.sleep(settle_ms / 1000.0)
                        log_event()
                        t_end = time.monotonic() + step_dur
                        while time.monotonic() < t_end and not self.stop_flag.is_set():
                            time.sleep(0.02)
                else:
                    raise ValueError(f"Unknown block type: {ttype}")

            if self.stop_flag.is_set():
                break

            if i < len(tests) and break_between > 0:
                self._emit("status", f"Break {break_between:.1f}s ...")
                t_end = time.monotonic() + break_between
                while time.monotonic() < t_end and not self.stop_flag.is_set():
                    time.sleep(0.05)

        self._emit("status", "Test sequence complete.")
        self.stop_flag.set()

    def start(self):
        self._setup_hardware()
        self.t0 = time.monotonic()
        self.stop_flag.clear()
        self.t_sample = threading.Thread(target=self._sampling_loop, daemon=True)
        self.t_run = threading.Thread(target=self._run_tests, daemon=True)
        self.t_sample.start()
        self.t_run.start()

    def stop(self):
        self.stop_flag.set()

    def join(self, timeout: Optional[float] = None):
        for t in (self.t_run, self.t_sample):
            try:
                if t:
                    t.join(timeout=timeout)
            except Exception:
                pass
        self._teardown_hardware()

    def get_plot_data(self):
        with self.lock:
            return list(self.buf_t), list(self.buf_freq), list(self.buf_flow)


# ---------------- UI ----------------
class DashboardApp:
    def __init__(self, initial_config_path: Optional[str] = None):
        self.root = tk.Tk()
        self.root.title("Pump Testing Suite Dashboard (v2)")
        self.root.geometry("1250x840")

        self.ui_events: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
        self.cfg: Dict[str, Any] = {"tests": []}

        self.active_stream: Optional[LiveSessionBase] = None
        self.run_dir: Optional[str] = None
        self.cfg_path: Optional[str] = None

        self._build_layout()
        self._build_plot()

        self.ani = animation.FuncAnimation(self.fig, self._update_plot, interval=150, blit=False)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        if initial_config_path:
            try:
                self.load_config(initial_config_path)
            except Exception as e:
                messagebox.showwarning("Load config", f"Could not load config:\n{e}")

        self._poll_ui_events()

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
        ttk.Combobox(hw, state="readonly", width=10, values=["fluigent", "dummy"], textvariable=self.var_sensor_backend)\
            .grid(row=3, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(hw, text="Channel:").grid(row=3, column=2, sticky="e", padx=4, pady=2)
        ttk.Entry(hw, textvariable=self.var_sensor_channel, width=10).grid(row=3, column=3, sticky="w", padx=4, pady=2)

        ttk.Label(hw, text="Sample rate (Hz):").grid(row=4, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(hw, textvariable=self.var_sample_rate, width=10).grid(row=4, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(hw, text="Break between blocks (s):").grid(row=4, column=2, sticky="e", padx=4, pady=2)
        ttk.Entry(hw, textvariable=self.var_break_between, width=10).grid(row=4, column=3, sticky="w", padx=4, pady=2)

        mode_row = ttk.Frame(hw)
        mode_row.grid(row=5, column=0, columnspan=4, sticky="w", padx=4, pady=(8, 2))
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

        # Main split
        main = ttk.Frame(self.root)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 8))
        self._build_config_builder(left)

        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.plot_frame = ttk.LabelFrame(right, text="Live plots")
        self.plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.manual_frame = ttk.LabelFrame(right, text="Manual control")
        self.manual_frame.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))
        self._build_manual_panel(self.manual_frame)

        self._update_mode_ui()
        self._refresh_manual_duty_state()

    def _build_config_builder(self, parent):
        listf = ttk.LabelFrame(parent, text="Config builder (test blocks)")
        listf.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.lst_blocks = tk.Listbox(listf, width=62, height=16)
        self.lst_blocks.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.lst_blocks.bind("<<ListboxSelect>>", self._on_block_select)

        actions = ttk.Frame(listf)
        actions.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(0, 6))
        ttk.Button(actions, text="Add Constant", command=self.add_constant_block).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions, text="Add Sweep", command=self.add_sweep_block).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions, text="Duplicate", command=self.duplicate_selected_block).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions, text="Remove", command=self.remove_selected_block).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions, text="Move Up", command=lambda: self.move_selected_block(-1)).pack(side=tk.LEFT, padx=8)
        ttk.Button(actions, text="Move Down", command=lambda: self.move_selected_block(+1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(actions, text="Clear All", command=self.clear_all_blocks).pack(side=tk.RIGHT, padx=2)

        editf = ttk.LabelFrame(parent, text="Edit selected block")
        editf.pack(side=tk.TOP, fill=tk.X, pady=(8, 0))

        self.var_edit_type = tk.StringVar(value="constant")
        self.var_edit_repeat = tk.StringVar(value="1")
        self.var_edit_settle_ms = tk.StringVar(value="0")

        row0 = ttk.Frame(editf); row0.pack(side=tk.TOP, fill=tk.X, padx=6, pady=4)
        ttk.Label(row0, text="Type:").pack(side=tk.LEFT)
        ttk.Combobox(row0, state="readonly", width=12, values=["constant", "sweep"], textvariable=self.var_edit_type)\
            .pack(side=tk.LEFT, padx=6)
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

        ttk.Button(editf, text="Apply changes", command=self.apply_edit_to_selected)\
            .pack(side=tk.TOP, padx=6, pady=(6, 8))

        self.var_edit_type.trace_add("write", lambda *_: self._refresh_editor_visibility())
        self._refresh_editor_visibility()

    def _refresh_editor_visibility(self):
        ttype = self.var_edit_type.get().strip().lower()
        if ttype == "constant":
            self.edit_sweep.pack_forget()
            self.edit_constant.pack(side=tk.TOP, fill=tk.X, padx=6, pady=2)
        else:
            self.edit_constant.pack_forget()
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

        self.ax1.set_title("Frequency setpoint (Hz)")
        self.ax1.set_xlabel("t (s)")
        self.ax1.set_ylabel("Hz")
        self.ax2.set_title("Flow-rate (sensor units)")
        self.ax2.set_xlabel("t (s)")
        self.ax2.set_ylabel("Flow")

        self.ax1.grid(True, alpha=0.25)
        self.ax2.grid(True, alpha=0.25)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)

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
        window = 30.0
        tmin = max(0.0, tmax - window)
        self.ax1.set_xlim(tmin, tmin + window)
        self.ax2.set_xlim(tmin, tmin + window)

        try:
            fmin, fmax = min(freqs), max(freqs)
            if fmin == fmax:
                fmin -= 1; fmax += 1
            self.ax1.set_ylim(fmin, fmax)
        except Exception:
            pass

        try:
            ymin, ymax = min(flows), max(flows)
            if ymin == ymax:
                ymin -= 0.1; ymax += 0.1
            self.ax2.set_ylim(ymin, ymax)
        except Exception:
            pass

        self.canvas.draw()
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
            self.var_sample_rate.set(str(flow_cfg.get("sample_rate_hz", self.var_sample_rate.get())))

        meta = self.cfg.get("meta", {})
        if isinstance(meta, dict):
            self.var_test_name.set(meta.get("name", self.var_test_name.get()))
            self.var_operator.set(meta.get("operator", self.var_operator.get()))
            self.txt_desc.delete("1.0", tk.END)
            self.txt_desc.insert("1.0", meta.get("description", ""))

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
            },
            "serial": {"protocol": self.var_protocol.get().strip()},
            "flow_sensor": {
                "backend": self.var_sensor_backend.get().strip(),
                "channel": safe_int(self.var_sensor_channel.get(), 0),
                "sample_rate_hz": safe_float(self.var_sample_rate.get(), 25.0),
            },
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
    def refresh_blocks_list(self):
        self.lst_blocks.delete(0, tk.END)
        for i, t in enumerate(self.cfg.get("tests", []), 1):
            ttype = (t.get("type") or "constant").lower()
            if ttype == "constant":
                s = f"{i:02d}. constant: {t.get('frequency_hz')} Hz for {t.get('duration_s')} s (x{t.get('repeat', 1)})"
            else:
                sdur = t.get("step_duration_s", None)
                tdur = t.get("total_duration_s", None)
                d = f"{sdur}s/step" if sdur is not None else f"{tdur}s total"
                s = f"{i:02d}. sweep: {t.get('start_hz')}→{t.get('end_hz')} Hz, {t.get('steps')} steps, {d} (x{t.get('repeat', 1)})"
            self.lst_blocks.insert(tk.END, s)

    def _on_block_select(self, _evt=None):
        sel = self.lst_blocks.curselection()
        if not sel:
            return
        idx = int(sel[0])
        t = self.cfg["tests"][idx]
        ttype = (t.get("type") or "constant").lower()

        self.var_edit_type.set(ttype)
        self.var_edit_repeat.set(str(t.get("repeat", 1)))
        self.var_edit_settle_ms.set(str(t.get("settle_ms", 0)))

        if ttype == "constant":
            self.var_edit_freq.set(str(t.get("frequency_hz", 500)))
            self.var_edit_dur.set(str(t.get("duration_s", 10)))
        else:
            self.var_edit_start.set(str(t.get("start_hz", 100)))
            self.var_edit_end.set(str(t.get("end_hz", 1000)))
            self.var_edit_steps.set(str(t.get("steps", 10)))
            self.var_edit_step_dur.set(str(t.get("step_duration_s", "")))
            self.var_edit_total_dur.set(str(t.get("total_duration_s", "")))

        self._refresh_editor_visibility()

    def apply_edit_to_selected(self):
        sel = self.lst_blocks.curselection()
        if not sel:
            messagebox.showinfo("Edit", "Select a block to edit.")
            return
        idx = int(sel[0])

        ttype = self.var_edit_type.get().strip().lower()
        repeat = max(1, safe_int(self.var_edit_repeat.get(), 1))
        settle = max(0, safe_int(self.var_edit_settle_ms.get(), 0))

        if ttype == "constant":
            block = {
                "type": "constant",
                "frequency_hz": safe_int(self.var_edit_freq.get(), 500),
                "duration_s": safe_float(self.var_edit_dur.get(), 10.0),
                "repeat": repeat,
                "settle_ms": settle,
            }
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

        self.cfg["tests"][idx] = block
        self.refresh_blocks_list()
        self.lst_blocks.selection_clear(0, tk.END)
        self.lst_blocks.selection_set(idx)
        self.lst_blocks.activate(idx)

    def add_constant_block(self):
        self.cfg.setdefault("tests", [])
        self.cfg["tests"].append({"type": "constant", "frequency_hz": 500, "duration_s": 10, "repeat": 1, "settle_ms": 0})
        self.refresh_blocks_list()

    def add_sweep_block(self):
        self.cfg.setdefault("tests", [])
        self.cfg["tests"].append({"type": "sweep", "start_hz": 100, "end_hz": 1000, "steps": 10,
                                  "step_duration_s": 1.5, "repeat": 1, "settle_ms": 0})
        self.refresh_blocks_list()

    def remove_selected_block(self):
        sel = self.lst_blocks.curselection()
        if not sel:
            return
        self.cfg["tests"].pop(int(sel[0]))
        self.refresh_blocks_list()

    def duplicate_selected_block(self):
        sel = self.lst_blocks.curselection()
        if not sel:
            return
        idx = int(sel[0])
        self.cfg["tests"].insert(idx + 1, dict(self.cfg["tests"][idx]))
        self.refresh_blocks_list()
        self.lst_blocks.selection_clear(0, tk.END)
        self.lst_blocks.selection_set(idx + 1)

    def move_selected_block(self, direction: int):
        sel = self.lst_blocks.curselection()
        if not sel:
            return
        idx = int(sel[0])
        new_idx = idx + int(direction)
        tests = self.cfg.get("tests", [])
        if new_idx < 0 or new_idx >= len(tests):
            return
        tests[idx], tests[new_idx] = tests[new_idx], tests[idx]
        self.refresh_blocks_list()
        self.lst_blocks.selection_clear(0, tk.END)
        self.lst_blocks.selection_set(new_idx)
        self.lst_blocks.activate(new_idx)

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

        out_dir = ensure_dir(self.var_out_dir.get().strip())
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = (self.var_test_name.get().strip().replace(" ", "_")[:40] or "run")
        self.run_dir = ensure_dir(os.path.join(out_dir, f"{run_tag}_{run_name}"))

        with open(os.path.join(self.run_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(cfg.get("meta", {}), f, indent=2)
        with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        runner = PumpTestRunner(cfg, self.run_dir, self.ui_events)
        try:
            runner.start()
        except Exception as e:
            messagebox.showerror("Start failed", str(e))
            return

        self.active_stream = runner
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_manual_connect.config(state=tk.DISABLED)
        self.btn_manual_disconnect.config(state=tk.DISABLED)
        self.var_status.set(f"Running. Folder: {self.run_dir}")

    def manual_connect(self):
        if self.active_stream is not None:
            return
        cfg = self.build_config_from_ui()
        cfg["protocol"] = cfg.get("serial", {}).get("protocol", "freq_only")

        run_dir = None
        if bool(self.var_log_manual.get()):
            out_dir = ensure_dir(self.var_out_dir.get().strip())
            run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = (self.var_test_name.get().strip().replace(" ", "_")[:40] or "manual")
            run_dir = ensure_dir(os.path.join(out_dir, f"{run_tag}_{run_name}_MANUAL"))
            with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(cfg.get("meta", {}), f, indent=2)
            with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

        sess = ManualSession(cfg, run_dir, self.ui_events)
        try:
            sess.start()
        except Exception as e:
            messagebox.showerror("Manual connect failed", str(e))
            return

        self.active_stream = sess
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_manual_connect.config(state=tk.DISABLED)
        self.btn_manual_disconnect.config(state=tk.NORMAL)
        self.btn_start.config(state=tk.DISABLED)
        self.var_manual_info.set("Manual connected. Use Output ON + Set freq.")
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
        try:
            self.active_stream.stop()
            self.active_stream.join(timeout=5.0)
        finally:
            self.active_stream = None
            self.btn_stop.config(state=tk.DISABLED)
            self.btn_start.config(state=tk.NORMAL if self.var_mode.get() == "scheduled" else tk.DISABLED)
            self.btn_manual_connect.config(state=tk.NORMAL)
            self.btn_manual_disconnect.config(state=tk.DISABLED)
            self.var_status.set("Idle.")
            self.var_manual_info.set("Manual session not connected.")
            self.var_manual_enabled.set(False)

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
        except queue.Empty:
            pass
        self.root.after(120, self._poll_ui_events)

    def on_close(self):
        try:
            if self.active_stream:
                self.active_stream.stop()
                self.active_stream.join(timeout=2.0)
        except Exception:
            pass
        self.root.after(200, self.root.destroy)

    def run(self):
        self.root.mainloop()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="Optional config JSON to load at startup")
    args = ap.parse_args()
    DashboardApp(initial_config_path=args.config).run()

if __name__ == "__main__":
    main()
