"""Programmable bench supply RS PRO RSD3305P (Korad KA3305P family) for automatic voltage steps.

Protocol (Korad KA series; the RSD3305P is a rebadged KA3305P): 9600 baud 8N1, no flow control. Commands
are plain ASCII WITHOUT a line terminator: the supply takes a command as complete after a short pause,
so commands are spaced >= 60 ms. Answers have no terminator either.
  *IDN?          -> e.g. "RS PRO RSD3305P V2.0 SN:..." (or "KORAD KA3305P ...")
  VSET1:2.50     channel 1 voltage (V, 2 decimals)      VSET1? -> "02.50"
  ISET1:0.060    channel 1 current limit (A, 3 decimals) ISET1? -> "0.060"
  VOUT1? IOUT1?  measured output voltage / current
  OUT1 / OUT0    outputs on / off (all channels)
  STATUS?        one byte: bit 0 = CH1 CV (1) / CC (0), bit 1 = CH2, bit 6 = output on

Address: "COM7" (the supply's USB virtual COM port on this PC), or "host:port" / "socket://host:port"
for a supply plugged into another PC that runs supply_bridge.py (a TCP <-> serial bridge), or
"dummy" (a simulated supply with a resistive load, for tests).

SupplyMonitor polls the supply (V, I, CC/CV) once a second, logs supply/supply.csv and calls on_trip
when the supply stays in current limiting (CC) with the output on: the pump is then being starved,
the same condition the PPK2 watch catches.
"""
from __future__ import annotations

import csv
import os
import random
import re
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, Optional

clock = time.perf_counter


class SupplyError(RuntimeError):
    pass


def parse_address(addr: Any) -> str:
    """'dummy', 'COM7', '/dev/ttyUSB0', 'socket://h:p' unchanged; 'host:port' -> 'socket://host:port'."""
    a = str(addr or "").strip()
    if not a:
        raise ValueError("no supply address (e.g. COM7, or labpc.local:5025 for a supply behind supply_bridge.py)")
    if a.lower() == "dummy" or "://" in a or re.fullmatch(r"(?i)com\d+|/dev/\S+", a):
        return a
    if re.fullmatch(r"[\w.\-]+:\d+", a):
        return "socket://" + a
    raise ValueError(f"supply address {a!r}: use COM7, host:port or socket://host:port")


class RSDSupply:
    """One channel of an RSD3305P. Thread-safe: every command holds the lock and keeps the gap."""

    EOLS = {"": "", "none": "", "lf": "\n", "cr": "\r", "crlf": "\r\n"}

    def __init__(self, address: str, channel: int = 1, max_v: float = 2.5, timeout_s: float = 0.6,
                 gap_s: float = 0.06, eol: str = ""):
        self.address = parse_address(address)
        self.ch = int(channel)
        self.max_v = float(max_v)
        self.timeout_s, self.gap_s = float(timeout_s), float(gap_s)
        self.eol = self.EOLS[str(eol or "").strip().lower()]    # the KA protocol has none; "LF" if a firmware wants one
        self.ser = None
        self.idn = ""
        self._lock = threading.RLock()
        self._last = 0.0

    # ---- connection ----
    def open(self):
        import serial
        # read timeout = the silence that ends an answer (bytes come ~1 ms apart at 9600 baud)
        self.ser = serial.serial_for_url(self.address, baudrate=9600, bytesize=8, parity="N", stopbits=1,
                                         timeout=0.02, write_timeout=1.0)
        time.sleep(0.1)
        self._drain()
        self.idn = self.query("*IDN?")
        if not self.idn:
            self.close()
            raise SupplyError(f"no answer to *IDN? from {self.address} (supply on, cable/bridge connected, "
                              f"right COM port? some firmwares want \"eol\": \"LF\" in supply)")
        return self.idn

    def close(self):
        with self._lock:
            ser, self.ser = self.ser, None
            if ser is None:
                return
            # the supply ends a command by silence: let the last one (e.g. OUT0) finish first
            time.sleep(max(0.0, self.gap_s - (clock() - self._last)))
            try:
                ser.close()
            except Exception:
                pass

    # ---- low level ----
    def _drain(self):
        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass
        t_end = clock() + 0.1
        while clock() < t_end and self.ser.read(256):
            pass

    def _send(self, cmd: str):
        if self.ser is None:
            raise SupplyError("supply not connected")
        wait = self.gap_s - (clock() - self._last)
        if wait > 0:
            time.sleep(wait)
        self.ser.write((cmd + self.eol).encode("ascii"))
        try:
            self.ser.flush()
        except Exception:
            pass
        self._last = clock()

    def write(self, cmd: str):
        with self._lock:
            self._send(cmd)

    def query_raw(self, cmd: str) -> bytes:
        """Send a query, read until the answer stops (no terminator) or the timeout."""
        with self._lock:
            try:
                self.ser.reset_input_buffer()
            except Exception:
                pass
            self._send(cmd)
            data = b""
            t_end = clock() + self.timeout_s
            while clock() < t_end:
                chunk = self.ser.read(64)
                if chunk:
                    data += chunk
                elif data:
                    break                       # quiet for one read timeout after data: done
            self._last = clock()
            return data

    def query(self, cmd: str) -> str:
        return self.query_raw(cmd).decode("ascii", "ignore").strip("\x00\r\n ")

    def _query_float(self, cmd: str) -> float:
        txt = self.query(cmd)
        m = re.search(r"-?\d+(\.\d+)?", txt)
        if not m:
            raise SupplyError(f"{cmd} answered {txt!r}")
        return float(m.group(0))

    # ---- commands ----
    def set_voltage(self, v: float, verify: bool = True) -> float:
        v = float(v)
        if not 0.0 <= v <= self.max_v + 1e-9:
            raise SupplyError(f"{v:.3f} V is outside 0..{self.max_v:.3f} V (supply.max_mv)")
        for _ in range(3):
            self.write(f"VSET{self.ch}:{v:05.2f}")
            if not verify:
                return v
            back = self._query_float(f"VSET{self.ch}?")
            if abs(back - v) < 0.006:
                return back
        raise SupplyError(f"supply kept VSET at {back:.2f} V instead of {v:.2f} V")

    def set_current(self, a: float, verify: bool = True) -> float:
        a = float(a)
        if not 0.0 < a <= 5.0:
            raise SupplyError(f"current limit {a * 1000:.0f} mA is not sensible")
        for _ in range(3):
            self.write(f"ISET{self.ch}:{a:05.3f}")
            if not verify:
                return a
            back = self._query_float(f"ISET{self.ch}?")
            if abs(back - a) < 0.0006:
                return back
        raise SupplyError(f"supply kept ISET at {back * 1000:.0f} mA instead of {a * 1000:.0f} mA")

    def output(self, on: bool):
        self.write("OUT1" if on else "OUT0")

    def read_settings(self) -> Dict[str, float]:
        """Set voltage (V) and current limit (A) of the channel."""
        return {"v_set": self._query_float(f"VSET{self.ch}?"), "i_set": self._query_float(f"ISET{self.ch}?")}

    def status(self) -> int:
        raw = self.query_raw("STATUS?")
        if not raw:
            raise SupplyError("no answer to STATUS?")
        return raw[0]

    def measure(self) -> Dict[str, Any]:
        """Measured V and I of the channel, CV/CC and output state. The output also counts as on when
        it measures a voltage, in case bit 6 means something else on this firmware."""
        v = self._query_float(f"VOUT{self.ch}?")
        i = self._query_float(f"IOUT{self.ch}?")
        st = self.status()
        return {"v": v, "i": i, "cv": bool(st >> (self.ch - 1) & 1), "output": bool(st >> 6 & 1) or v >= 0.02,
                "status": st}

    def wait_for(self, v: float, tol_v: float = 0.05, timeout_s: float = 10.0) -> Dict[str, Any]:
        """Poll until the output reads v (within tol_v); the reading, or SupplyError after timeout_s
        (output off, current limiting, or the output not following)."""
        t_end = clock() + timeout_s
        while True:
            m = self.measure()
            if m["output"] and abs(m["v"] - v) <= tol_v:
                return m
            if clock() >= t_end:
                why = "output is OFF" if not m["output"] else (
                    "current limiting (CC)" if not m["cv"] else "output not following")
                raise SupplyError(f"output reads {m['v']:.3f} V / {m['i'] * 1000:.1f} mA after {timeout_s:g} s, "
                                  f"wanted {v:.3f} V: {why}")
            time.sleep(0.2)


class DummySupply(RSDSupply):
    """Simulated RSD3305P with a resistive load (r_ohm): CC when V/R exceeds the current limit."""

    def __init__(self, address: str = "dummy", channel: int = 1, max_v: float = 2.5, r_ohm: float = 80.0,
                 **_kw):
        self.address, self.ch, self.max_v = "dummy", int(channel), float(max_v)
        self.r_ohm = float(r_ohm)
        self.vset, self.iset, self.on = 0.0, 0.1, False
        self.idn = "DUMMY RSD3305P (simulated)"
        self._lock = threading.RLock()
        self._rng = random.Random(3)
        self.commands = deque(maxlen=1000)

    def open(self):
        return self.idn

    def close(self):
        pass

    def write(self, cmd: str):
        with self._lock:
            self.commands.append(cmd)
            m = re.fullmatch(r"VSET\d:(\d+\.\d+)", cmd)
            if m:
                self.vset = float(m.group(1))
            m = re.fullmatch(r"ISET\d:(\d+\.\d+)", cmd)
            if m:
                self.iset = float(m.group(1))
            if cmd in ("OUT1", "OUT0"):
                self.on = cmd == "OUT1"

    def _operating_point(self):
        if not self.on:
            return 0.0, 0.0, True
        i = self.vset / self.r_ohm
        if i > self.iset:
            return self.iset * self.r_ohm, self.iset, False
        return self.vset, i, True

    def query_raw(self, cmd: str) -> bytes:
        with self._lock:
            self.commands.append(cmd)
            v, i, cv = self._operating_point()
            ans = {"*IDN?": self.idn, f"VSET{self.ch}?": f"{self.vset:05.2f}", f"ISET{self.ch}?": f"{self.iset:05.3f}",
                   f"VOUT{self.ch}?": f"{v:05.2f}", f"IOUT{self.ch}?": f"{i + self._rng.gauss(0, 0.0003):05.3f}"}
            if cmd == "STATUS?":
                return bytes([(1 if cv else 0) | 0b10 | (0x40 if self.on else 0)])
            return ans.get(cmd, "").encode("ascii")


def make_supply(cfg: Dict[str, Any]) -> RSDSupply:
    """Supply driver from the 'supply' config (mode 'rsd3305p'): address, channel, max_mv."""
    addr = str(cfg.get("address") or "").strip()
    max_v = float(cfg.get("max_mv") or 2500) / 1000.0
    ch = int(cfg.get("channel") or 1)
    if addr.lower() == "dummy":
        return DummySupply(channel=ch, max_v=max_v, r_ohm=float(cfg.get("dummy_r_ohm", 80.0)))
    return RSDSupply(addr, channel=ch, max_v=max_v, eol=str(cfg.get("eol") or ""))


def is_limiting(m: Dict[str, Any], v_set: Optional[float], i_set: Optional[float]) -> bool:
    """Supply clamping the current: the CC status bit backed by the readings (current near the limit
    or output below the set voltage), or the readings alone if they clearly say so. The readings
    guard against a status bit with the opposite meaning on some firmware."""
    if not m["output"]:
        return False
    near_limit = i_set is not None and m["i"] >= 0.9 * i_set
    sagging = v_set is not None and m["v"] < v_set - 0.05
    if not m["cv"] and (near_limit or sagging or (i_set is None and v_set is None)):
        return True
    return i_set is not None and v_set is not None and m["i"] >= 0.98 * i_set and m["v"] < v_set - 0.1


class SupplyMonitor:
    """Polls V / I / CC once per period_s, logs supply/supply.csv, trips after cc_trip_n limiting
    readings in a row with the output on (the supply is clamping the pump's current)."""

    COLS = ["t_s", "v_set_v", "v_out_v", "i_out_ma", "mode", "output", "status"]

    def __init__(self, supply: RSDSupply, run_dir: Optional[str], period_s: float = 1.0, cc_trip_n: int = 2,
                 on_trip: Optional[Callable[[Dict[str, Any]], None]] = None,
                 on_lost: Optional[Callable[[str], None]] = None):
        self.supply, self.period_s, self.cc_trip_n = supply, float(period_s), int(cc_trip_n)
        self.on_trip, self.on_lost = on_trip, on_lost
        self.dir = os.path.join(run_dir, "supply") if run_dir else None
        self.live: Dict[str, Any] = {"state": "starting"}
        self.v_set: Optional[float] = None      # V, kept up to date by the session
        self.i_set: Optional[float] = None      # A (current limit)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._f = self._w = None
        self.t0: Optional[float] = None
        self.tripped = False
        self.lock = threading.Lock()

    def start(self, t0: float):
        self.t0 = t0
        if self.dir:
            os.makedirs(self.dir, exist_ok=True)
            self._f = open(os.path.join(self.dir, "supply.csv"), "a", newline="", encoding="utf-8")
            self._w = csv.writer(self._f, lineterminator="\n")
            if self._f.tell() == 0:
                self._w.writerow(self.COLS)
        self._thread = threading.Thread(target=self._loop, name="supply-monitor", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._f:
            try:
                self._f.flush()
                self._f.close()
            except Exception:
                pass
            self._f = None

    def get_live(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.live)

    def _loop(self):
        cc_run, errors = 0, 0
        while not self._stop.is_set():
            t_next = clock() + self.period_s
            try:
                m = self.supply.measure()
                errors = 0
            except Exception as e:
                errors += 1
                with self.lock:
                    self.live = {**self.live, "state": "no answer", "error": str(e)}
                if errors == 5 and self.on_lost:
                    self.on_lost(f"Programmable supply {self.supply.address} stopped answering ({e}).")
                self._stop.wait(max(0.0, t_next - clock()))
                continue
            t_s = clock() - self.t0 if self.t0 is not None else 0.0
            limiting = is_limiting(m, self.v_set, self.i_set)
            mode = "CC" if limiting else "CV"
            with self.lock:
                self.live = {"state": "tripped" if self.tripped else "ok", "v_set": self.v_set, "v": m["v"],
                             "i_ma": m["i"] * 1000.0, "mode": mode, "output": m["output"], "t_s": t_s,
                             "address": self.supply.address}
            if self._w:
                self._w.writerow([f"{t_s:.3f}", "" if self.v_set is None else f"{self.v_set:.3f}", f"{m['v']:.3f}",
                                  f"{m['i'] * 1000:.2f}", mode, int(m["output"]), m["status"]])
                self._f.flush()
            cc_run = cc_run + 1 if limiting else 0
            if cc_run >= self.cc_trip_n and not self.tripped and self.on_trip:
                self.tripped = True
                self.on_trip({"v": m["v"], "i_ma": m["i"] * 1000.0, "n": cc_run, "period_s": self.period_s})
            self._stop.wait(max(0.0, t_next - clock()))


def main(argv=None):
    """Bench check without the dashboard:
        python supply_rsd.py COM7                  identity, settings, output reading (changes nothing)
        python supply_rsd.py labpc:5025 --set 1.5  set 1.5 V (limit --limit-ma, default 60), output on, read
        python supply_rsd.py COM7 --off            output off"""
    import argparse
    ap = argparse.ArgumentParser(description="RSD3305P bench check")
    ap.add_argument("address", help="COM7, host:port (supply_bridge.py) or dummy")
    ap.add_argument("--ch", type=int, default=1)
    ap.add_argument("--set", type=float, default=None, help="volts (at most --max)")
    ap.add_argument("--max", type=float, default=2.5, help="refuse voltages above this (default 2.5 V)")
    ap.add_argument("--limit-ma", type=float, default=60.0)
    ap.add_argument("--off", action="store_true")
    ap.add_argument("--eol", default="", help="LF / CR / CRLF if the firmware wants a terminator")
    a = ap.parse_args(argv)
    cfg = {"address": a.address, "channel": a.ch, "max_mv": a.max * 1000, "eol": a.eol}
    drv = make_supply(cfg)
    print("IDN:", drv.open())
    try:
        if a.off:
            drv.output(False)
        elif a.set is not None:
            print("ISET read back:", drv.set_current(a.limit_ma / 1000.0))
            print("VSET read back:", drv.set_voltage(a.set))
            drv.output(True)
            m = drv.wait_for(a.set, 0.05, 10.0)
            print(f"reached in time: {m['v']:.3f} V")
        st, m = drv.read_settings(), drv.measure()
        print(f"set {st['v_set']:.2f} V / {st['i_set'] * 1000:.0f} mA; output {'ON' if m['output'] else 'off'}, "
              f"reads {m['v']:.3f} V / {m['i'] * 1000:.1f} mA, {'CV' if m['cv'] else 'CC'}, status 0x{m['status']:02X}")
    finally:
        drv.close()


if __name__ == "__main__":
    main()
