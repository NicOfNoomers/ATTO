"""Nordic PPK2 power meter for the pump dashboard (ampere-meter mode).

The PPK2 sits between the bench supply and the pump driver and measures the
supply current 100,000 times per second. The bench supply sets the voltage
(and its current limit protects the pump), so power = supply voltage x current.

PowerMeter runs one acquisition thread that:
  * reads the PPK2's USB stream and decodes it with numpy (decode_words),
  * writes bins (default 10 ms: mean/min/max current, power, cumulative
    charge and energy) to <run>/power/bins_*.csv, starting a new file every
    rotate_min minutes and gzipping closed ones, plus power_1s.csv,
  * saves raw snapshots (last raw_length_s seconds, every raw_every_min
    minutes, and around a trip) to <run>/power/raw/,
  * keeps live values and a plot buffer for the dashboard,
  * trips (on_trip) when the current sits at/above watch_ma for
    watch_hold_ms -- with the bench supply limiting at 60 mA, that means the
    supply is clamping, i.e. something is wrong,
  * reconnects if the PPK2 drops off USB (on_fatal after reconnect_timeout_s).

Backend "dummy" simulates an H-bridge-driven coil (65 ohm, 20 mH) at real-time
pace, for testing without hardware.

Sample word layout (ppk2-api): bits 0-13 ADC, 14-16 range, 18-23 counter,
24-31 logic inputs. Sample times come from the sample count (100 kS/s),
corrected against PC time over the run (anchors.csv logs both).
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

FS = 100_000                  # PPK2 sample rate, samples/s
ADC_MULT = 1.8 / 163840       # same constant as ppk2-api
STALL_S = 5.0                 # no data this long -> treat as disconnected
FSYNC_S = 2.0

POWER_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "backend": "ppk2",          # "ppk2" | "dummy"
    "port": "auto",
    "bin_ms": 10,
    "rotate_min": 60,
    "compress_closed": True,
    "raw_every_min": 10,        # 0 = no periodic snapshots
    "raw_length_s": 2.0,
    "watch_ma": None,           # None -> 95 % of the bench supply limit; 0 = off
    "watch_hold_ms": 200,
    "cut_power_on_trip": True,  # also switch off the PPK2's output path on a trip
    "reconnect_timeout_s": 30,
}

BIN_COLS = ["t_s", "i_mean_ma", "i_min_ma", "i_max_ma", "p_mean_mw", "v_mv",
            "charge_mc", "energy_mj", "logic_or", "lost_samples"]


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


# ---------------- sources ----------------
class PPK2Source:
    """Decoded sample chunks from a real PPK2 in ampere-meter mode."""
    raw_format = "ppk2_words_uint32le"

    def __init__(self, port: str, vdd_mv: float):
        self.port_cfg = port
        self.port = None
        self.vdd_mv = float(vdd_mv)
        self.ppk = None
        self.modifiers = None

    def open(self):
        try:
            from ppk2_api.ppk2_api import PPK2_API
        except ImportError:
            import sys
            raise RuntimeError(f"The ppk2-api package is not installed for the Python running the dashboard "
                               f"({sys.executable}). Run install.bat and start with run_dashboard.bat, or: "
                               f"\"{sys.executable}\" -m pip install -r requirements.txt") from None
        port = self.port_cfg
        if not port or str(port).lower() == "auto":
            found = find_ppk2_ports()
            if not found:
                raise RuntimeError("No PPK2 found. Is it plugged in and switched on? "
                                   "(Close the nRF Connect Power Profiler app if it is open.)")
            port = found[0]
        ppk = PPK2_API(port, timeout=1, write_timeout=1, exclusive=True)
        try:
            try:
                ppk.ser.set_buffer_size(rx_size=4 * 1024 * 1024)   # ~10 s if the reader stalls
            except Exception:
                pass
            if not ppk.get_modifiers():
                raise RuntimeError(f"{port}: could not read the PPK2's calibration data.")
            ppk.use_ampere_meter()
            # Ampere-meter mode powers nothing; ppk2-api needs the input voltage
            # for its calibration and refuses to measure without it.
            ppk.set_source_voltage(int(round(min(5000, max(800, self.vdd_mv)))))
            ppk.toggle_DUT_power("ON")
            ppk.start_measuring()
        except Exception:
            try:
                ppk.ser.close()
            except Exception:
                pass
            raise
        self.ppk, self.port = ppk, port
        self.modifiers = ppk.modifiers
        self._marr = modifier_arrays(ppk.modifiers)
        self._rem = b""
        self._aligned = False
        self._last_counter: Optional[int] = None

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
        if self.ppk:
            self.ppk.toggle_DUT_power("OFF")

    def close(self):
        ppk, self.ppk = self.ppk, None
        if not ppk:
            return
        for f in (ppk.stop_measuring, lambda: ppk.toggle_DUT_power("OFF"), ppk.ser.close):
            try:
                f()
            except Exception:
                pass


class DummySource:
    """Simulated supply current of an H-bridge driving a 65 ohm / 20 mH coil."""
    raw_format = "current_ua_float32le"
    R_OHM, L_H, IDLE_MA = 65.0, 0.020, 0.05

    def __init__(self, get_state: Callable[[], Tuple[int, Optional[float]]], vdd_mv: float):
        self.get_state = get_state
        self.vdd_mv = float(vdd_mv)
        self.port = "dummy"
        self.modifiers = None
        self.extra_ma = 0.0        # tests: add a fault current
        self.fail = False          # tests: simulate a USB drop

    def open(self):
        if self.fail:
            raise RuntimeError("dummy PPK2 unavailable")
        self._t_last = time.monotonic()
        self._n = 0
        self._rs = np.random.default_rng(0)

    def read(self):
        if self.fail:
            raise IOError("dummy PPK2 disconnected")
        now = time.monotonic()
        n = min(int((now - self._t_last) * FS), FS)
        if n < 200:
            return None
        self._t_last += n / FS
        hz, v_mv = self.get_state()
        v = (v_mv if v_mv is not None else self.vdd_mv) / 1000.0
        t = (self._n + np.arange(n)) / FS
        self._n += n
        tau = self.L_H / self.R_OHM
        if hz and hz > 0:
            th = np.mod(t, 0.5 / hz)                        # time since the last bridge flip
            k = 2.0 / (1.0 + np.exp(-0.5 / hz / tau))       # steady-state symmetric square-wave response
            i = v / self.R_OHM * np.abs(1.0 - k * np.exp(-th / tau))
        else:
            i = np.zeros(n)
        cur_ua = (i * 1000.0 + self.IDLE_MA + self.extra_ma) * 1000.0 + self._rs.normal(0, 20, n)
        return cur_ua, np.zeros(n, np.uint8), 0, cur_ua.astype("<f4")

    def power_cut(self):
        pass

    def close(self):
        pass


# ---------------- meter ----------------
class PowerMeter:
    def __init__(self, cfg: Dict[str, Any], run_dir: Optional[str], vdd_mv: float,
                 emit: Callable[[str, str], None],
                 get_state: Callable[[], Tuple[int, Optional[float]]] = lambda: (0, None),
                 on_trip: Optional[Callable[[Dict[str, Any]], None]] = None,
                 on_fatal: Optional[Callable[[str], None]] = None):
        self.cfg = {**POWER_DEFAULTS, **(cfg or {})}
        self.dir = os.path.join(run_dir, "power") if run_dir else None
        self.emit = emit
        self.on_trip = on_trip
        self.on_fatal = on_fatal
        self.v_mv = float(vdd_mv)
        backend = str(self.cfg.get("backend") or "ppk2").lower()
        self.src = DummySource(get_state, vdd_mv) if backend == "dummy" else PPK2Source(self.cfg.get("port"), vdd_mv)

        self.bin_n = max(1, int(round(FS * float(self.cfg["bin_ms"]) / 1000.0)))
        self.bins_per_s = max(1, int(round(FS / self.bin_n)))
        self.plot_k = max(1, int(round(0.05 * FS / self.bin_n)))           # ~50 ms plot points
        wm = self.cfg.get("watch_ma")
        self.watch_ma = float(wm) if wm not in (None, "") else 0.0
        self.watch_bins = max(1, int(round(float(self.cfg["watch_hold_ms"]) / 1000.0 * FS / self.bin_n)))

        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.t0: Optional[float] = None
        self.plot_t: deque = deque(maxlen=12000)
        self.plot_i: deque = deque(maxlen=12000)
        self.live: Dict[str, Any] = {"state": "idle", "port": None}
        self.tripped = False
        self.fatal = False
        self.total_lost = 0
        self.charge_mc = 0.0
        self.energy_mj = 0.0

        self._files: Dict[str, Any] = {}
        self._writers: Dict[str, Any] = {}
        self._bins_path = None
        self._ring: deque = deque()
        self._ring_n = 0
        self._ring_max = int(FS * max(5.0, float(self.cfg["raw_length_s"])))

    # ---- public API ----
    def open(self):
        """Connect and configure the instrument (raises on failure)."""
        self.src.open()
        self.live.update(state="connected", port=self.src.port)
        if self.dir:
            os.makedirs(os.path.join(self.dir, "raw"), exist_ok=True)
            self._open_file("events", "power_events.csv", ["t_s", "time", "event", "detail"])
            self._open_file("s1", "power_1s.csv", BIN_COLS)
            self._open_file("anchors", "anchors.csv", ["t_host_s", "sample_index", "t_sample_s", "lag_s", "fs_est"])
            self._open_bins()

    def start(self, t0: float):
        self.t0 = t0
        self._log_event("start", f"backend={self.cfg['backend']} port={self.src.port} bin_ms={self.cfg['bin_ms']} "
                                 f"watch_ma={self.watch_ma:g} v_mv={self.v_mv:g}")
        self.thread = threading.Thread(target=self._run, name="PowerMeter", daemon=True)
        self.thread.start()

    def set_voltage(self, mv: float):
        """Supply voltage now applied (from a confirmed voltage step). Used for
        power and as the PPK2's calibration Vdd (software only, no command)."""
        with self.lock:
            self.v_mv = float(mv)
            self.src.vdd_mv = float(mv)
        self._log_event("voltage", f"{mv:g} mV")

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
            self.src.close()
        except Exception:
            pass
        self._log_event("stop", f"energy {self.energy_mj / 1000:.3f} J, charge {self.charge_mc / 1000:.3f} C, "
                                f"lost samples {self.total_lost}")
        if self.dir:
            try:
                with open(os.path.join(self.dir, "summary.json"), "w", encoding="utf-8") as f:
                    json.dump({"energy_j": self.energy_mj / 1000, "charge_c": self.charge_mc / 1000,
                               "lost_samples": self.total_lost, "tripped": self.tripped,
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
        self._s1: List[tuple] = []     # bins waiting for the 1 s summary
        self._plot_acc: List[tuple] = []
        self._over = 0
        self._next_anchor = self._next_flush = self._next_sync = 0.0
        self._next_raw = time.monotonic() + 60.0 * float(self.cfg["raw_every_min"] or 0)
        self._rotate_at = time.monotonic() + 60.0 * float(self.cfg["rotate_min"] or 1e9)
        self._t_data = time.monotonic()
        with self.lock:
            self.live["state"] = "measuring"
        while not self.stop_flag.is_set():
            try:
                got = self.src.read()
            except Exception as e:
                if not self._reconnect(f"read failed: {e}"):
                    return
                continue
            now = time.monotonic()
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
        # Keep sample times honest: if they drift >0.5 s from PC time (stall,
        # reconnect, bad estimate), resync to the arrival time.
        lag = t_rel - self._t_next
        if abs(lag) > 0.5:
            self._log_event("time_resync", f"lag {lag:+.3f} s")
            self._t_next = t_rel
            self._t_carry = t_rel - (self._carry.size + n) / self._fs

        self._ring.append(raw)
        self._ring_n += raw.size
        while self._ring and self._ring_n - self._ring[0].size >= self._ring_max:
            self._ring_n -= self._ring.popleft().size

        cur = np.concatenate((self._carry, cur_ua / 1000.0)) if self._carry.size else cur_ua / 1000.0
        lg = np.concatenate((self._carry_logic, logic)) if self._carry_logic.size else logic
        nb = cur.size // self.bin_n
        if nb:
            b = cur[:nb * self.bin_n].reshape(nb, self.bin_n)
            means, mins, maxs = b.mean(1), b.min(1), b.max(1)
            ors = np.bitwise_or.reduce(lg[:nb * self.bin_n].reshape(nb, self.bin_n), axis=1)
            dt_bin = self.bin_n / self._fs
            with self.lock:
                v = self.v_mv
            rows = []
            for j in range(nb):
                q = means[j] * dt_bin                       # mC
                self.charge_mc += q
                self.energy_mj += q * v / 1000.0            # mJ
                t = self._t_carry + j * dt_bin
                lost = self._pending_lost if j == 0 else 0
                rows.append((t, means[j], mins[j], maxs[j], means[j] * v / 1000.0, v,
                             self.charge_mc, self.energy_mj, int(ors[j]), lost))
            self._pending_lost = 0
            self._t_carry += nb * dt_bin
            self._write_bins(rows)
            self._after_bins(rows, means)
        self._carry = cur[nb * self.bin_n:]
        self._carry_logic = lg[nb * self.bin_n:]

    def _after_bins(self, rows, means):
        # 1 s summary + live values
        self._s1.extend(rows)
        while len(self._s1) >= self.bins_per_s:
            chunk, self._s1 = self._s1[:self.bins_per_s], self._s1[self.bins_per_s:]
            a = np.array([r[1:6] for r in chunk], dtype=float)
            last = chunk[-1]
            s = (chunk[0][0], a[:, 0].mean(), a[:, 1].min(), a[:, 2].max(), a[:, 3].mean(), last[5],
                 last[6], last[7], int(np.bitwise_or.reduce([r[8] for r in chunk])), sum(r[9] for r in chunk))
            if "s1" in self._writers:
                self._writers["s1"].writerow(self._fmt_row(s))
            with self.lock:
                self.live.update(state="tripped" if self.tripped else "measuring", t_s=s[0], i_ma=s[1],
                                 i_max_ma=s[3], p_mw=s[4], v_mv=s[5], charge_c=s[6] / 1000.0,
                                 energy_j=s[7] / 1000.0, lost=self.total_lost, port=self.src.port)
        # plot points (~50 ms)
        self._plot_acc.extend((r[0], r[1]) for r in rows)
        pts = []
        while len(self._plot_acc) >= self.plot_k:
            grp, self._plot_acc = self._plot_acc[:self.plot_k], self._plot_acc[self.plot_k:]
            pts.append((grp[0][0], sum(x[1] for x in grp) / len(grp)))
        if pts:
            with self.lock:
                for t, i in pts:
                    self.plot_t.append(t)
                    self.plot_i.append(i)
        # current watch: the bench supply is clamping -> trip once
        if self.watch_ma > 0 and not self.tripped:
            for j, m in enumerate(means):
                self._over = self._over + 1 if m >= self.watch_ma else 0
                if self._over >= self.watch_bins:
                    self._trip(rows[j][0], float(m))
                    break

    def _trip(self, t_s: float, i_ma: float):
        self.tripped = True
        info = {"t_s": t_s, "i_ma": i_ma, "watch_ma": self.watch_ma, "hold_ms": float(self.cfg["watch_hold_ms"])}
        cut = bool(self.cfg.get("cut_power_on_trip", True))
        if cut:
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
        """Reconnect after a USB drop. False (and on_fatal) after reconnect_timeout_s."""
        self._log_event("disconnect", reason)
        self.emit("status", f"Power meter: {reason}; reconnecting ...")
        with self.lock:
            self.live["state"] = "reconnecting"
        try:
            self.src.close()
        except Exception:
            pass
        deadline = time.monotonic() + float(self.cfg["reconnect_timeout_s"])
        while not self.stop_flag.is_set():
            try:
                self.src.open()
                self._log_event("reconnect", f"port {self.src.port}")
                self.emit("status", "Power meter: reconnected.")
                self._t_data = time.monotonic()
                self._t_next = None             # restart the time base from PC time
                self._anchor0 = None
                self._carry = np.zeros(0)
                self._carry_logic = np.zeros(0, np.uint8)
                with self.lock:
                    self.live["state"] = "measuring"
                return True
            except Exception as e:
                if time.monotonic() > deadline:
                    self.fatal = True
                    msg = f"Power meter lost ({reason}) and could not reconnect: {e}"
                    self._log_event("fatal", msg)
                    with self.lock:
                        self.live["state"] = "lost"
                    if self.on_fatal:
                        self.on_fatal(msg)
                    return False
                self.stop_flag.wait(2.0)
        return False

    # ---- files ----
    def _open_file(self, key, name, header):
        f = open(os.path.join(self.dir, name), "a", newline="", encoding="utf-8")
        w = csv.writer(f)
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
        return [f"{r[0]:.4f}", f"{r[1]:.5f}", f"{r[2]:.5f}", f"{r[3]:.5f}", f"{r[4]:.4f}", f"{r[5]:.0f}",
                f"{r[6]:.4f}", f"{r[7]:.4f}", r[8], r[9]]

    def _write_bins(self, rows):
        w = self._writers.get("bins")
        if w:
            w.writerows(self._fmt_row(r) for r in rows)

    def _log_event(self, event: str, detail: str = ""):
        t = (time.monotonic() - self.t0) if self.t0 is not None else 0.0
        w = self._writers.get("events")
        if w:
            try:
                w.writerow([f"{t:.3f}", datetime.now().isoformat(timespec="seconds"), event, detail])
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
                w.writerow([f"{t_rel:.4f}", self._idx, f"{self._t_next:.4f}", f"{t_rel - self._t_next:.4f}",
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
