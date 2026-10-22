"""Nordic PPK2 power meter for the pump dashboard.

Two modes (power.mode):
  * "ampere" (default): the PPK2 sits between the bench supply and the pump
    driver and only measures. The bench supply sets the voltage and its current
    limit protects the pump.
  * "source": the PPK2 itself supplies the driver's VM from VOUT (bench supply
    disconnected) and the dashboard sets the voltage. There is NO hardware
    current limit (the PPK2 can source ~1 A), so the software trip is the only
    protection and cannot be switched off for a real PPK2.
Either way the PPK2 measures the supply current 100,000 times per second and
power = supply voltage x current.

PowerMeter runs one acquisition thread that:
  * reads the PPK2's USB stream and decodes it with numpy (decode_words),
  * writes bins (bin_ms, 1 ms and up: mean/min/max current, power, cumulative
    charge and energy) to <run>/power/bins_*.csv, a new file every rotate_min
    minutes with closed ones gzipped, plus power_1s.csv,
  * saves raw snapshots (last raw_length_s seconds, every raw_every_min
    minutes, and around a trip) to <run>/power/raw/,
  * keeps live values and a plot buffer for the dashboard,
  * trips (on_trip) when
      - the current stays at/above watch_ma for watch_hold_ms (sustained), or
      - any 1 ms mean reaches trip_peak_ma (instant),
    except during blank_ms after the output is switched on or the voltage is
    changed (the driver's decoupling capacitor draws an inrush). On a trip the
    PPK2 output is switched off first, then on_trip runs (pump to 0 Hz),
  * reconnects if the PPK2 drops off USB in ampere mode; in source mode a
    drop is fatal (no automatic re-powering of the pump).

All host times use time.perf_counter (QueryPerformanceCounter): on Python 3.10
time.monotonic only ticks every 15.6 ms on Windows.

Backend "dummy" simulates an H-bridge-driven coil (65 ohm, 20 mH) at real-time
pace, for testing without hardware.

Sample word layout (ppk2-api): bits 0-13 ADC, 14-16 range, 18-23 counter,
24-31 logic inputs. Sample times come from the sample count (100 kS/s),
anchored to PC time (a sample can't arrive before it was taken, so the
fastest-arriving chunk sets the offset); anchors.csv logs both.
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import shutil
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

clock = time.perf_counter     # the dashboard's run clock (sub-microsecond on Windows)

FS = 100_000                  # PPK2 sample rate, samples/s
ADC_MULT = 1.8 / 163840       # same constant as ppk2-api
STALL_S = 5.0                 # no data this long -> treat as disconnected
FSYNC_S = 2.0
PPK2_MIN_MV, PPK2_MAX_MV = 800, 5000     # PPK2 source range (hardware)
DEFAULT_MAX_MV = 2500                    # pump limit; raising it is an explicit config decision

POWER_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "backend": "ppk2",          # "ppk2" | "dummy"
    "mode": "ampere",           # "ampere" (bench supply sets V) | "source" (PPK2 sets V)
    "port": "auto",
    "bin_ms": 10,
    "rotate_min": 60,
    "compress_closed": True,
    "raw_every_min": 10,        # 0 = no periodic snapshots
    "raw_length_s": 2.0,
    "watch_ma": None,           # None -> ampere: 95 % of the bench limit (set by the dashboard); source: 150
    "watch_hold_ms": None,      # None -> ampere: 200; source: 300
    "trip_peak_ma": None,       # None -> ampere: off; source: 300 (on any 1 ms mean)
    "blank_ms": 100,            # no trips this long after output on / voltage change
    "cut_power_on_trip": True,  # switch off the PPK2's output path on a trip
    "restart_on_voltage": False,  # source mode: stop/restart the stream around a voltage change
    "reconnect_timeout_s": 30,
}
SOURCE_WATCH_DEFAULTS = {"watch_ma": 150.0, "watch_hold_ms": 300.0, "trip_peak_ma": 300.0}

BIN_COLS = ["t_s", "i_mean_ma", "i_min_ma", "i_max_ma", "p_mean_mw", "v_mv",
            "charge_mc", "energy_mj", "logic_or", "lost_samples"]
BIN_FMT = ["%.5f", "%.5f", "%.5f", "%.5f", "%.4f", "%.0f", "%.5f", "%.5f", "%d", "%d"]


def resolve_power_cfg(cfg: Optional[Dict[str, Any]], bench_limit_ma: Optional[float] = None) -> Dict[str, Any]:
    """Power config with the mode-dependent trip defaults filled in."""
    p = {**POWER_DEFAULTS, **(cfg or {})}
    mode = str(p.get("mode") or "ampere").lower()
    p["mode"] = "source" if mode == "source" else "ampere"
    if p["mode"] == "source":
        for k, v in SOURCE_WATCH_DEFAULTS.items():
            if p.get(k) in (None, ""):
                p[k] = v
    else:
        if p.get("watch_ma") in (None, ""):
            p["watch_ma"] = 0.95 * float(bench_limit_ma) if bench_limit_ma else 0.0
        if p.get("watch_hold_ms") in (None, ""):
            p["watch_hold_ms"] = 200.0
        if p.get("trip_peak_ma") in (None, ""):
            p["trip_peak_ma"] = 0.0
    for k in ("watch_ma", "watch_hold_ms", "trip_peak_ma", "blank_ms", "bin_ms"):
        p[k] = float(p[k] or 0.0)
    return p


# ---------------- decoding ----------------
def modifier_arrays(mods: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """ppk2-api's per-range calibration dicts -> numpy arrays indexed by range 0..4."""
    return {k: np.array([float(mods[k][str(i)]) for i in range(5)])
            for k in ("R", "GS", "GI", "O", "S", "I", "UG")}


def words_valid(w: np.ndarray) -> np.ndarray:
    return (((w >> 14) & 0x7) <= 4) & (((w >> 17) & 1) == 0)


def best_alignment(data: bytes) -> int:
    """Byte offset (0-3) at which the stream splits into valid sample words
    (range bits <= 4, bit 17 clear, counter stepping by 1 if it is used)."""
    best, best_score = 0, -1.0
    for off in range(4):
        n = (len(data) - off) // 4
        if n < 16:
            continue
        w = np.frombuffer(data[off:off + 4 * n], dtype="<u4")
        cnt = ((w >> 18) & 0x3F).astype(np.int16)
        score = float(np.mean(words_valid(w))) + float(np.mean((np.diff(cnt) % 64) == 1))
        if score > best_score:
            best, best_score = off, score
    return best


def decode_words(w: np.ndarray, marr: Dict[str, np.ndarray], vdd_mv: float):
    """Sample words -> (current uA, range, counter, logic). Same formula as
    ppk2-api, without its spike filter (which smooths ~4 samples after a range
    switch)."""
    adc = (w & 0x3FFF).astype(np.float64) * 4
    rng = np.minimum((w >> 14) & 0x7, 4).astype(np.intp)
    counter = ((w >> 18) & 0x3F).astype(np.uint8)
    logic = (w >> 24).astype(np.uint8)
    rwg = (adc - marr["O"][rng]) * (ADC_MULT / marr["R"][rng])
    amps = marr["UG"][rng] * (rwg * (marr["GS"][rng] * rwg + marr["GI"][rng])
                              + (marr["S"][rng] * (vdd_mv / 1000.0) + marr["I"][rng]))
    return amps * 1e6, rng, counter, logic


def counter_gaps(counter: np.ndarray) -> int:
    """Samples missing according to the 6-bit counter (0 if the counter is unused)."""
    if counter.size < 2 or not np.any(counter):
        return 0
    steps = np.diff(counter.astype(np.int16)) % 64
    return int(np.sum(np.where(steps == 0, 63, steps - 1)))


PPK2_USB_IDS = {(0x1915, 0xC00A)}      # Nordic Semiconductor, PPK2


def find_ppk2_ports() -> List[str]:
    """PPK2 serial ports. ppk2-api matches the Windows name 'nRF Connect USB CDC
    ACM', which needs Nordic's driver; on other PCs the PPK2 shows up as a plain
    'USB Serial Device', so also match its USB ID."""
    found: List[str] = []
    try:
        from ppk2_api.ppk2_api import PPK2_API
        found += [p for p in PPK2_API.list_devices()]
    except Exception:
        pass
    try:
        import serial.tools.list_ports
        for p in serial.tools.list_ports.comports():
            if ((p.vid, p.pid) in PPK2_USB_IDS or (p.product or "") == "PPK2") and p.device not in found:
                found.append(p.device)
    except Exception:
        pass
    return found


def _import_ppk2_api():
    try:
        from ppk2_api.ppk2_api import PPK2_API, PPK2_Command
        return PPK2_API, PPK2_Command
    except ImportError:
        import sys
        raise RuntimeError(f"The ppk2-api package is not installed for the Python running the dashboard "
                           f"({sys.executable}). Click 'Install missing packages' in the Power meter panel, or: "
                           f"\"{sys.executable}\" -m pip install -r requirements.txt") from None


def _drain(ser, quiet_s: float = 0.25, max_s: float = 3.0) -> bool:
    """Discard incoming bytes until the line has been quiet for quiet_s."""
    t_end = clock() + max_s
    last = clock()
    while clock() < t_end:
        n = ser.in_waiting
        if n:
            ser.read(n)
            last = clock()
        elif clock() - last >= quiet_s:
            return True
        time.sleep(0.02)
    return False


def _read_modifiers(ppk, cmd, timeout_s: float = 3.0) -> bool:
    """Ask for the calibration data and parse it. Accumulates the reply until
    'END' and decodes it leniently (ppk2-api decodes every chunk as strict utf-8
    and keeps only the last one)."""
    ppk._write_serial((cmd.GET_META_DATA,))
    buf = b""
    t_end = clock() + timeout_s
    while clock() < t_end:
        n = ppk.ser.in_waiting
        if n:
            buf += ppk.ser.read(n)
            if b"END" in buf:
                break
        time.sleep(0.02)
    if b"END" not in buf:
        return False
    text = buf[:buf.rindex(b"END") + 3].decode("utf-8", errors="ignore")
    return bool(ppk._parse_metadata(text))


def connect_ppk2(port: Optional[str] = None):
    """Open a PPK2, switch its output OFF and read its calibration -> (PPK2_API, port).

    The output is switched off first: a PPK2 left in source mode by a crashed
    session would otherwise keep powering the pump. A PPK2 can also still be
    streaming samples from an earlier session; ppk2-api's get_modifiers() then
    fails with "'utf-8' codec can't decode byte 0x..". So: output off, stop the
    stream, drain the leftover bytes, then ask for the calibration (3 tries).
    """
    PPK2_API, cmd = _import_ppk2_api()
    if not port or str(port).lower() == "auto":
        found = find_ppk2_ports()
        if not found:
            raise RuntimeError("No PPK2 found. Is it plugged in (USB DATA/POWER port) and switched on? "
                               "(Close the nRF Connect Power Profiler app if it is open.)")
        port = found[0]
    ppk = PPK2_API(port, timeout=1, write_timeout=1, exclusive=True)
    try:
        try:
            ppk.ser.set_buffer_size(rx_size=4 * 1024 * 1024)   # ~10 s of data if the reader stalls
        except Exception:
            pass
        ppk.toggle_DUT_power("OFF")
        for _ in range(3):
            ppk.stop_measuring()
            _drain(ppk.ser)
            if _read_modifiers(ppk, cmd):
                return ppk, port
        raise RuntimeError(f"{port}: the PPK2 did not send its calibration data (3 tries). Unplug it and plug "
                           f"it in again; if that doesn't help, update its firmware once with nRF Connect for "
                           f"Desktop -> Power Profiler (then close that app).")
    except Exception:
        try:
            ppk.ser.close()
        except Exception:
            pass
        raise


# ---------------- sources ----------------
class PPK2Source:
    """Decoded sample chunks from a real PPK2 (ampere or source mode).

    Reads happen only in the meter thread; writes (voltage, output on/off) can
    come from any thread and are serialised by _wlock."""
    raw_format = "ppk2_words_uint32le"

    def __init__(self, port: str, vdd_mv: float, mode: str = "ampere", restart_on_voltage: bool = False):
        self.port_cfg = port
        self.port = None
        self.vdd_mv = float(vdd_mv)
        self.mode = mode
        self.restart_on_voltage = restart_on_voltage
        self.ppk = None
        self.modifiers = None
        self.output_on = False
        self._wlock = threading.Lock()

    def open(self):
        ppk, port = connect_ppk2(self.port_cfg)          # output is OFF from here
        try:
            mv = int(round(min(PPK2_MAX_MV, max(PPK2_MIN_MV, self.vdd_mv))))
            if self.mode == "source":
                # Mode and voltage first, output last: never powered before the voltage is set.
                ppk.use_source_meter()
                ppk.set_source_voltage(mv)
                time.sleep(0.1)
            else:
                ppk.use_ampere_meter()
                # Ampere-meter mode powers nothing; ppk2-api needs the input voltage
                # for its calibration and refuses to measure without it.
                ppk.set_source_voltage(mv)
            ppk.toggle_DUT_power("ON")
            self.output_on = True
            ppk.start_measuring()
        except Exception:
            for f in (lambda: ppk.toggle_DUT_power("OFF"), ppk.ser.close):
                try:
                    f()
                except Exception:
                    pass
            self.output_on = False
            raise
        self.ppk, self.port = ppk, port
        self.modifiers = ppk.modifiers
        self._marr = modifier_arrays(ppk.modifiers)
        self._rem = b""
        self._aligned = False
        self._last_counter: Optional[int] = None

    def set_voltage(self, mv: float):
        """Source mode: command the output voltage (already range-checked)."""
        mv = int(round(mv))
        with self._wlock:
            if not self.ppk:
                raise RuntimeError("PPK2 not connected")
            if self.restart_on_voltage:
                self.ppk.stop_measuring()
                self.ppk.set_source_voltage(mv)
                self.ppk.start_measuring()
            else:
                self.ppk.set_source_voltage(mv)
            self.vdd_mv = float(mv)

    def read(self):
        d = self.ppk.get_data()
        if not d:
            return None
        buf = self._rem + d
        if not self._aligned:
            if len(buf) < 4096:
                self._rem = buf
                return None
            buf = buf[best_alignment(buf):]
            self._aligned = True
            self._last_counter = None
        n = len(buf) // 4
        self._rem = buf[4 * n:]
        if n == 0:
            return None
        w = np.frombuffer(buf[:4 * n], dtype="<u4")
        if n >= 1000 and float(np.mean(words_valid(w))) < 0.9:
            self._aligned = False          # lost bytes somewhere: realign on the next read
            self._rem = b""
            return None
        cur, _rng, cnt, logic = decode_words(w, self._marr, self.vdd_mv)
        seq = cnt if self._last_counter is None else np.concatenate(([self._last_counter], cnt))
        gaps = counter_gaps(seq)
        self._last_counter = int(cnt[-1])
        return cur, logic, gaps, w

    def power_cut(self):
        with self._wlock:
            if self.ppk:
                self.ppk.toggle_DUT_power("OFF")
                self.output_on = False

    def close(self):
        with self._wlock:
            ppk, self.ppk = self.ppk, None
            if not ppk:
                return
            # Output off first (in source mode it powers the pump), then stop and close.
            for f in (lambda: ppk.toggle_DUT_power("OFF"), ppk.stop_measuring, ppk.ser.close):
                try:
                    f()
                except Exception:
                    pass
            self.output_on = False


class DummySource:
    """Simulated supply current of an H-bridge driving a 65 ohm / 20 mH coil."""
    raw_format = "current_ua_float32le"
    R_OHM, L_H, IDLE_MA = 65.0, 0.020, 0.05

    def __init__(self, get_state: Callable[[], Tuple[int, Optional[float]]], vdd_mv: float,
                 mode: str = "ampere"):
        self.get_state = get_state
        self.vdd_mv = float(vdd_mv)
        self.mode = mode
        self.port = "dummy"
        self.modifiers = None
        self.extra_ma = 0.0        # tests: add a fault current
        self.fail = False          # tests: simulate a USB drop
        self.output_on = False
        self.voltage_log: List[float] = []     # tests: every commanded voltage

    def open(self):
        if self.fail:
            raise RuntimeError("dummy PPK2 unavailable")
        self._t_last = clock()
        self._n = 0
        self._rs = np.random.default_rng(0)
        self.output_on = True

    def set_voltage(self, mv: float):
        self.vdd_mv = float(mv)
        self.voltage_log.append(float(mv))

    def read(self):
        if self.fail:
            raise IOError("dummy PPK2 disconnected")
        now = clock()
        n = min(int((now - self._t_last) * FS), FS)
        if n < 200:
            return None
        self._t_last += n / FS
        hz, v_mv = self.get_state()
        if self.mode == "source" or v_mv is None:
            v_mv = self.vdd_mv
        v = v_mv / 1000.0
        t = (self._n + np.arange(n)) / FS
        self._n += n
        tau = self.L_H / self.R_OHM
        if not self.output_on:
            i = np.zeros(n)
        elif hz and hz > 0:
            th = np.mod(t, 0.5 / hz)                        # time since the last bridge flip
            k = 2.0 / (1.0 + np.exp(-0.5 / hz / tau))       # steady-state symmetric square-wave response
            i = v / self.R_OHM * np.abs(1.0 - k * np.exp(-th / tau))
        else:
            i = np.zeros(n)
        extra = self.extra_ma if self.output_on else 0.0
        idle = self.IDLE_MA if self.output_on else 0.0
        cur_ua = (i * 1000.0 + idle + extra) * 1000.0 + self._rs.normal(0, 20, n)
        return cur_ua, np.zeros(n, np.uint8), 0, cur_ua.astype("<f4")

    def power_cut(self):
        self.output_on = False

    def close(self):
        self.output_on = False


# ---------------- meter ----------------
class PowerMeter:
    def __init__(self, cfg: Dict[str, Any], run_dir: Optional[str], vdd_mv: float,
                 emit: Callable[[str, str], None],
                 get_state: Callable[[], Tuple[int, Optional[float]]] = lambda: (0, None),
                 on_trip: Optional[Callable[[Dict[str, Any]], None]] = None,
                 on_fatal: Optional[Callable[[str], None]] = None,
                 max_mv: float = DEFAULT_MAX_MV):
        self.cfg = resolve_power_cfg(cfg)
        self.mode = self.cfg["mode"]
        self.dir = os.path.join(run_dir, "power") if run_dir else None
        self.emit = emit
        self.on_trip = on_trip
        self.on_fatal = on_fatal
        self.max_mv = float(min(PPK2_MAX_MV, max_mv or DEFAULT_MAX_MV))
        self.v_mv = float(vdd_mv)
        backend = str(self.cfg.get("backend") or "ppk2").lower()
        self.backend = backend
        if self.mode == "source":
            self.check_voltage(self.v_mv)
            if backend != "dummy" and self.cfg["watch_ma"] <= 0:
                raise RuntimeError("PPK2 source mode without the software current trip is not allowed: "
                                   "set power.watch_ma > 0 (default 150 mA). It is the only protection.")
        if backend == "dummy":
            self.src = DummySource(get_state, vdd_mv, self.mode)
        else:
            self.src = PPK2Source(self.cfg.get("port"), vdd_mv, self.mode, bool(self.cfg.get("restart_on_voltage")))

        self.bin_n = max(1, int(round(FS * float(self.cfg["bin_ms"]) / 1000.0)))
        self.bins_per_s = max(1, int(round(FS / self.bin_n)))
        self.plot_k = max(1, int(round(0.05 * FS / self.bin_n)))           # ~50 ms plot points
        self.watch_ma = self.cfg["watch_ma"]
        self.peak_ma = self.cfg["trip_peak_ma"]
        self.watch_bins = max(1, int(round(self.cfg["watch_hold_ms"] / 1000.0 * FS / self.bin_n)))
        self.blank_s = self.cfg["blank_ms"] / 1000.0

        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.t0: Optional[float] = None
        self.plot_t: deque = deque(maxlen=12000)
        self.plot_i: deque = deque(maxlen=12000)
        self.live: Dict[str, Any] = {"state": "idle", "port": None, "mode": self.mode}
        self.tripped = False
        self.fatal = False
        self.total_lost = 0
        self.charge_mc = 0.0
        self.energy_mj = 0.0
        self._blank_until = 0.0        # run time (s); trips ignored before this

        self._files: Dict[str, Any] = {}
        self._writers: Dict[str, Any] = {}
        self._bins_path = None
        self._ring: deque = deque()
        self._ring_n = 0
        self._ring_max = int(FS * max(5.0, float(self.cfg["raw_length_s"])))

    # ---- public API ----
    def check_voltage(self, mv: float):
        """Source mode range check: [800, max_mv]. Raises ValueError outside it."""
        if not (PPK2_MIN_MV <= float(mv) <= self.max_mv):
            raise ValueError(f"{mv:g} mV is outside the allowed source range "
                             f"{PPK2_MIN_MV}-{self.max_mv:g} mV (supply.max_mv).")

    def open(self):
        """Connect and configure the instrument (raises on failure). In source
        mode this switches the pump's supply on at the start voltage."""
        self.src.open()
        self.live.update(state="connected", port=self.src.port)
        if self.dir:
            os.makedirs(os.path.join(self.dir, "raw"), exist_ok=True)
            self._open_file("events", "power_events.csv", ["t_s", "time", "event", "detail"])
            self._open_file("s1", "power_1s.csv", BIN_COLS)
            self._open_file("anchors", "anchors.csv", ["t_host_s", "sample_index", "t_sample_s", "lag_s", "fs_est"])
            self._open_bins()

    def start(self, t0: float):
        """t0 is the run's start on time.perf_counter."""
        self.t0 = t0
        self._blank_until = (clock() - t0) + self.blank_s
        self._log_event("start", f"backend={self.backend} mode={self.mode} port={self.src.port} "
                                 f"bin_ms={self.cfg['bin_ms']:g} watch_ma={self.watch_ma:g} "
                                 f"hold_ms={self.cfg['watch_hold_ms']:g} peak_ma={self.peak_ma:g} "
                                 f"v_mv={self.v_mv:g} max_mv={self.max_mv:g}")
        self.thread = threading.Thread(target=self._run, name="PowerMeter", daemon=True)
        self.thread.start()

    def set_voltage(self, mv: float) -> float:
        """New supply voltage. Source mode: commands the PPK2 (range-checked,
        raises ValueError outside [800, max_mv]) and blanks the trips briefly.
        Ampere mode: the bench supply was set by hand; only power and the
        calibration Vdd change. Returns the applied voltage."""
        mv = float(mv)
        if self.mode == "source":
            self.check_voltage(mv)
            self.src.set_voltage(mv)
            if self.t0 is not None:
                self._blank_until = (clock() - self.t0) + self.blank_s
        with self.lock:
            self.v_mv = mv
            self.src.vdd_mv = mv
        self._log_event("voltage", f"{mv:g} mV" + (" (commanded)" if self.mode == "source" else ""))
        return mv

    def safe_off(self, reason: str = ""):
        """Switch the PPK2 output off (source mode: removes the pump's supply).
        Safe to call from any thread, any number of times."""
        try:
            self.src.power_cut()
            self._log_event("output_off", reason)
        except Exception as e:
            self._log_event("error", f"output off failed: {e!r}")

    def get_live(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.live)

    def get_plot_data(self) -> Tuple[List[float], List[float]]:
        with self.lock:
            return list(self.plot_t), list(self.plot_i)

    def close(self):
        self.stop_flag.set()
        if self.thread:
            self.thread.join(timeout=3.0)
        try:
            self.src.close()            # output off first, then stop and close
        except Exception:
            pass
        self._log_event("stop", f"energy {self.energy_mj / 1000:.3f} J, charge {self.charge_mc / 1000:.3f} C, "
                                f"lost samples {self.total_lost}")
        if self.dir:
            try:
                with open(os.path.join(self.dir, "summary.json"), "w", encoding="utf-8") as f:
                    json.dump({"energy_j": self.energy_mj / 1000, "charge_c": self.charge_mc / 1000,
                               "lost_samples": self.total_lost, "tripped": self.tripped, "mode": self.mode,
                               "bin_ms": self.cfg["bin_ms"], "port": self.src.port,
                               "last_live": self.get_live()}, f, indent=2, default=str)
            except Exception:
                pass
        for f in list(self._files.values()):
            self._close_quietly(f)
        self._files.clear()
        self._writers.clear()
        with self.lock:
            self.live["state"] = "closed"

    # ---- acquisition thread ----
    def _run(self):
        self._t_next = None            # run time (s) of the next sample
        self._idx = 0                  # device sample index of the next sample
        self._fs = float(FS)
        self._anchor0 = None           # (host t, idx) at the first data
        self._carry = np.zeros(0)
        self._carry_logic = np.zeros(0, np.uint8)
        self._t_carry = 0.0
        self._pending_lost = 0
        self._s1: List[Dict[str, np.ndarray]] = []    # bin arrays waiting for the 1 s summary
        self._s1_n = 0
        self._plot_t: List[np.ndarray] = []
        self._plot_i: List[np.ndarray] = []
        self._over = 0
        self._next_anchor = self._next_flush = self._next_sync = 0.0
        self._next_raw = clock() + 60.0 * float(self.cfg["raw_every_min"] or 0)
        self._rotate_at = clock() + 60.0 * float(self.cfg["rotate_min"] or 1e9)
        self._t_data = clock()
        with self.lock:
            self.live["state"] = "measuring"
        while not self.stop_flag.is_set():
            try:
                got = self.src.read()
            except Exception as e:
                if not self._reconnect(f"read failed: {e}"):
                    return
                continue
            now = clock()
            if got is None:
                if now - self._t_data > STALL_S:
                    if not self._reconnect(f"no data for {STALL_S:g} s"):
                        return
                    continue
                time.sleep(0.003)
                continue
            self._t_data = now
            try:
                self._ingest(*got, now)
                self._housekeeping(now)
            except Exception as e:
                self._log_event("error", f"processing failed: {e!r}")
                self.emit("error", f"Power meter: processing error ({e!r}); measurement continues.")
                time.sleep(0.05)
            time.sleep(0.005)      # let ~500 samples accumulate per read

    def _ingest(self, cur_ua: np.ndarray, logic: np.ndarray, gaps: int, raw, now: float):
        if gaps:
            # Samples the counter says were lost: fill them with the last value so
            # bin times and energy stay continuous; the count is logged per bin.
            self.total_lost += gaps
            self._pending_lost += gaps
            fill = min(gaps, FS)
            last = self._carry[-1] * 1000.0 if self._carry.size else float(cur_ua[0])
            cur_ua = np.concatenate((np.full(fill, last), cur_ua))
            logic = np.concatenate((np.zeros(fill, np.uint8), logic))
        n = cur_ua.size
        t_rel = now - self.t0
        if self._t_next is None:
            self._t_next = t_rel - n / self._fs
            self._t_carry = self._t_next - self._carry.size / self._fs
        self._idx += n
        self._t_next += n / self._fs
        if self._anchor0 is None:
            self._anchor0 = (now, self._idx)
        lag = t_rel - self._t_next
        if abs(lag) > 0.5:
            # Stall, reconnect or bad estimate: resync to the arrival time.
            self._log_event("time_resync", f"lag {lag:+.3f} s")
            self._t_next = t_rel
            self._t_carry = t_rel - (self._carry.size + n) / self._fs
        elif lag < 0:
            # A sample can't arrive before it was taken: the fastest-arriving chunk
            # sets the time offset (bounds the error to the minimum USB latency).
            self._t_next += lag
            self._t_carry += lag

        self._ring.append(raw)
        self._ring_n += raw.size
        while self._ring and self._ring_n - self._ring[0].size >= self._ring_max:
            self._ring_n -= self._ring.popleft().size

        cur_ma = cur_ua / 1000.0
        # Instant trip on 1 ms means of the new samples (independent of bin_ms)
        if self.peak_ma > 0 and not self.tripped and cur_ma.size >= 100:
            m1 = cur_ma[:cur_ma.size // 100 * 100].reshape(-1, 100).mean(1)
            j = int(np.argmax(m1))
            t_j = self._t_next - (cur_ma.size - 100 * j) / self._fs
            if m1[j] >= self.peak_ma and t_j >= self._blank_until:
                self._trip(t_j, float(m1[j]), "peak")

        cur = np.concatenate((self._carry, cur_ma)) if self._carry.size else cur_ma
        lg = np.concatenate((self._carry_logic, logic)) if self._carry_logic.size else logic
        nb = cur.size // self.bin_n
        if nb:
            b = cur[:nb * self.bin_n].reshape(nb, self.bin_n)
            dt_bin = self.bin_n / self._fs
            with self.lock:
                v = self.v_mv
            means = b.mean(1)
            charge = self.charge_mc + np.cumsum(means * dt_bin)              # mC
            energy = self.energy_mj + np.cumsum(means * dt_bin * v / 1000.0)  # mJ
            self.charge_mc, self.energy_mj = float(charge[-1]), float(energy[-1])
            lost = np.zeros(nb, np.int64)
            lost[0] = self._pending_lost
            self._pending_lost = 0
            bins = {
                "t": self._t_carry + np.arange(nb) * dt_bin, "mean": means, "min": b.min(1), "max": b.max(1),
                "p": means * v / 1000.0, "v": np.full(nb, v), "charge": charge, "energy": energy,
                "or": np.bitwise_or.reduce(lg[:nb * self.bin_n].reshape(nb, self.bin_n), axis=1).astype(np.int64),
                "lost": lost,
            }
            self._t_carry += nb * dt_bin
            self._write_bins(bins)
            self._after_bins(bins)
        self._carry = cur[nb * self.bin_n:]
        self._carry_logic = lg[nb * self.bin_n:]

    @staticmethod
    def _cat(chunks: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        return {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}

    def _after_bins(self, bins: Dict[str, np.ndarray]):
        nb = bins["t"].size
        # 1 s summary + live values
        self._s1.append(bins)
        self._s1_n += nb
        while self._s1_n >= self.bins_per_s:
            a = self._cat(self._s1)
            k = self.bins_per_s
            head = {key: val[:k] for key, val in a.items()}
            rest = {key: val[k:] for key, val in a.items()}
            self._s1 = [rest] if rest["t"].size else []
            self._s1_n = rest["t"].size
            s = (head["t"][0], head["mean"].mean(), head["min"].min(), head["max"].max(), head["p"].mean(),
                 head["v"][-1], head["charge"][-1], head["energy"][-1], int(np.bitwise_or.reduce(head["or"])),
                 int(head["lost"].sum()))
            if "s1" in self._writers:
                self._writers["s1"].writerow(self._fmt_row(s))
            with self.lock:
                self.live.update(state="tripped" if self.tripped else "measuring", t_s=s[0], i_ma=s[1],
                                 i_max_ma=s[3], p_mw=s[4], v_mv=s[5], charge_c=s[6] / 1000.0,
                                 energy_j=s[7] / 1000.0, lost=self.total_lost, port=self.src.port,
                                 output_on=bool(getattr(self.src, "output_on", True)))
        # plot points (~50 ms)
        self._plot_t.append(bins["t"])
        self._plot_i.append(bins["mean"])
        pt, pi = np.concatenate(self._plot_t), np.concatenate(self._plot_i)
        m = pt.size // self.plot_k
        if m:
            with self.lock:
                self.plot_t.extend(pt[:m * self.plot_k:self.plot_k].tolist())
                self.plot_i.extend(pi[:m * self.plot_k].reshape(m, self.plot_k).mean(1).tolist())
        self._plot_t, self._plot_i = [pt[m * self.plot_k:]], [pi[m * self.plot_k:]]
        # Sustained trip: watch_ma held for watch_hold_ms (outside the blanking)
        if self.watch_ma > 0 and not self.tripped:
            for j in range(nb):
                if bins["t"][j] < self._blank_until:
                    self._over = 0
                    continue
                self._over = self._over + 1 if bins["mean"][j] >= self.watch_ma else 0
                if self._over >= self.watch_bins:
                    self._trip(float(bins["t"][j]), float(bins["mean"][j]), "sustained")
                    break

    def _trip(self, t_s: float, i_ma: float, kind: str):
        self.tripped = True
        info = {"t_s": t_s, "i_ma": i_ma, "kind": kind, "watch_ma": self.watch_ma,
                "hold_ms": float(self.cfg["watch_hold_ms"]), "peak_ma": self.peak_ma, "mode": self.mode}
        # Output off first: removes the energy (source mode: the pump's supply)
        if self.mode == "source" or bool(self.cfg.get("cut_power_on_trip", True)):
            try:
                self.src.power_cut()
                info["power_cut"] = True
            except Exception as e:
                info["power_cut"] = f"failed: {e!r}"
        self._log_event("trip", json.dumps(info))
        self._snapshot("trip")
        with self.lock:
            self.live["state"] = "tripped"
        if self.on_trip:
            try:
                self.on_trip(info)
            except Exception as e:
                self._log_event("error", f"on_trip handler failed: {e!r}")

    def _reconnect(self, reason: str) -> bool:
        """Reconnect after a USB drop. False (and on_fatal) after reconnect_timeout_s.
        Source mode never reconnects: that would switch the pump's supply back on."""
        self._log_event("disconnect", reason)
        with self.lock:
            self.live["state"] = "reconnecting"
        try:
            self.src.close()
        except Exception:
            pass
        if self.mode == "source":
            return self._fail(f"Power meter (source mode) lost: {reason}. The PPK2 output was switched off "
                              f"if it could still be reached; it is not switched back on automatically.")
        self.emit("status", f"Power meter: {reason}; reconnecting ...")
        deadline = clock() + float(self.cfg["reconnect_timeout_s"])
        while not self.stop_flag.is_set():
            try:
                self.src.open()
                self._log_event("reconnect", f"port {self.src.port}")
                self.emit("status", "Power meter: reconnected.")
                self._t_data = clock()
                self._t_next = None             # restart the time base from PC time
                self._anchor0 = None
                self._carry = np.zeros(0)
                self._carry_logic = np.zeros(0, np.uint8)
                with self.lock:
                    self.live["state"] = "measuring"
                return True
            except Exception as e:
                if clock() > deadline:
                    return self._fail(f"Power meter lost ({reason}) and could not reconnect: {e}")
                self.stop_flag.wait(2.0)
        return False

    def _fail(self, msg: str) -> bool:
        self.fatal = True
        self._log_event("fatal", msg)
        with self.lock:
            self.live["state"] = "lost"
        if self.on_fatal:
            self.on_fatal(msg)
        return False

    # ---- files ----
    def _open_file(self, key, name, header):
        f = open(os.path.join(self.dir, name), "a", newline="", encoding="utf-8")
        w = csv.writer(f, lineterminator="\n")
        if f.tell() == 0:
            w.writerow(header)
        self._files[key], self._writers[key] = f, w

    def _open_bins(self):
        name = f"bins_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self._bins_path = os.path.join(self.dir, name)
        self._open_file("bins", name, BIN_COLS)

    def _rotate_bins(self):
        old_f, old_path = self._files.pop("bins", None), self._bins_path
        self._writers.pop("bins", None)
        self._close_quietly(old_f)
        self._open_bins()
        if old_path and self.cfg.get("compress_closed", True):
            threading.Thread(target=self._gzip, args=(old_path,), daemon=True).start()

    @staticmethod
    def _gzip(path):
        try:
            with open(path, "rb") as fi, gzip.open(path + ".gz", "wb", compresslevel=6) as fo:
                shutil.copyfileobj(fi, fo)
            os.remove(path)
        except Exception:
            pass        # the uncompressed file stays; nothing lost

    @staticmethod
    def _close_quietly(f):
        if not f:
            return
        try:
            f.flush()
            os.fsync(f.fileno())
        except Exception:
            pass
        try:
            f.close()
        except Exception:
            pass

    @staticmethod
    def _fmt_row(r):
        return [f"{r[0]:.5f}", f"{r[1]:.5f}", f"{r[2]:.5f}", f"{r[3]:.5f}", f"{r[4]:.4f}", f"{r[5]:.0f}",
                f"{r[6]:.5f}", f"{r[7]:.5f}", r[8], r[9]]

    def _write_bins(self, bins: Dict[str, np.ndarray]):
        f = self._files.get("bins")
        if f:
            cols = np.column_stack([bins[k] for k in ("t", "mean", "min", "max", "p", "v", "charge", "energy",
                                                       "or", "lost")])
            np.savetxt(f, cols, fmt=BIN_FMT, delimiter=",")

    def _log_event(self, event: str, detail: str = ""):
        t = (clock() - self.t0) if self.t0 is not None else 0.0
        w = self._writers.get("events")
        if w:
            try:
                w.writerow([f"{t:.4f}", datetime.now().isoformat(timespec="milliseconds"), event, detail])
                self._files["events"].flush()
            except Exception:
                pass

    def _snapshot(self, reason: str):
        if not self.dir or not self._ring:
            return
        want = int(FS * float(self.cfg["raw_length_s"])) if reason == "periodic" else self._ring_max
        data = np.concatenate(list(self._ring))[-want:]
        t_end = self._t_carry + self._carry.size / self._fs
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(self.dir, "raw", f"{stamp}_{reason}")
        try:
            data.astype(data.dtype.newbyteorder("<"), copy=False).tofile(base + ".bin")
            with open(base + ".json", "w", encoding="utf-8") as f:
                json.dump({"format": self.src.raw_format, "fs": FS, "samples": int(data.size),
                           "t_start_s": t_end - data.size / self._fs, "vdd_mv": self.v_mv,
                           "reason": reason, "modifiers": self.src.modifiers}, f, indent=2, default=str)
        except Exception as e:
            self._log_event("error", f"snapshot failed: {e!r}")

    def _housekeeping(self, now: float):
        if now >= self._next_anchor and self._anchor0 is not None:
            self._next_anchor = now + 1.0
            t_rel = now - self.t0
            h0, i0 = self._anchor0
            if now - h0 >= 300.0:     # long baseline -> refine the sample rate (clamped to +-0.1 %)
                self._fs = min(FS * 1.001, max(FS * 0.999, (self._idx - i0) / (now - h0)))
            w = self._writers.get("anchors")
            if w:
                w.writerow([f"{t_rel:.5f}", self._idx, f"{self._t_next:.5f}", f"{t_rel - self._t_next:.5f}",
                            f"{self._fs:.3f}"])
        if now >= self._next_flush:
            self._next_flush = now + 1.0
            sync = now >= self._next_sync
            if sync:
                self._next_sync = now + FSYNC_S
            for f in self._files.values():
                try:
                    f.flush()
                    if sync:
                        os.fsync(f.fileno())
                except Exception:
                    pass
        if self.dir and now >= self._rotate_at:
            self._rotate_at = now + 60.0 * float(self.cfg["rotate_min"])
            self._rotate_bins()
        if self.dir and self.cfg["raw_every_min"] and now >= self._next_raw:
            self._next_raw = now + 60.0 * float(self.cfg["raw_every_min"])
            self._snapshot("periodic")
