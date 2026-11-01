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
                 gap_s: float = 0.06, eol: str = "", baud: int = 9600):
        self.address = parse_address(address)
        self.ch = int(channel)
        self.max_v = float(max_v)
        self.baud = int(baud)
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
        self.ser = serial.serial_for_url(self.address, baudrate=self.baud, bytesize=8, parity="N", stopbits=1,
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
    return RSDSupply(addr, channel=ch, max_v=max_v, eol=str(cfg.get("eol") or ""), baud=int(cfg.get("baud") or 9600))


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
        self._cc_run = 0
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

    def rearm(self):
        """Allow another trip (manual session: output switched on again, limit or voltage changed)."""
        self._cc_run = 0
        self.tripped = False

    def get_live(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.live)

    def _loop(self):
        errors = 0
        self._cc_run = 0
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
            self._cc_run = self._cc_run + 1 if limiting else 0
            if self._cc_run >= self.cc_trip_n and not self.tripped and self.on_trip:
                self.tripped = True
                self.on_trip({"v": m["v"], "i_ma": m["i"] * 1000.0, "n": self._cc_run, "period_s": self.period_s})
            self._stop.wait(max(0.0, t_next - clock()))


def _bits(st: Optional[int]) -> str:
    if st is None:
        return "no answer"
    return (f"0x{st:02X} = {st:08b}: CH1 {'CV' if st & 1 else 'CC'} (bit 0), CH2 {'CV' if st & 2 else 'CC'} (bit 1), "
            f"output {'on' if st & 0x40 else 'off'} (bit 6)")


class _Report:
    """Self-test console + log lines, PASS/FAIL tally, operator prompts (none with --yes)."""

    def __init__(self, interactive: bool):
        self.lines, self.fails, self.interactive = [], [], interactive

    def say(self, s: str = ""):
        print(s, flush=True)
        self.lines.append(s)

    def check(self, ok: bool, what: str) -> bool:
        self.say(("PASS  " if ok else "FAIL  ") + what)
        if not ok:
            self.fails.append(what)
        return ok

    def ask(self, prompt: str) -> str:
        if not self.interactive:
            return ""
        ans = input(prompt).strip()           # Ctrl+C stops the test (settings are restored)
        self.lines.append(prompt + ans)
        return ans

    def ask_number(self, prompt: str) -> Optional[float]:
        while True:
            ans = self.ask(prompt).lower().replace(",", ".").rstrip("vam ").strip()
            if not ans:
                return None
            try:
                return float(ans)
            except ValueError:
                self.say("      type a number like 1.002, or just Enter to skip")

    def save(self, log_dir: str, prefix: str = "supply_selftest"):
        path = os.path.join(log_dir, time.strftime(prefix + "_%Y%m%d_%H%M%S.txt"))
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.lines) + "\n")
            print(f"log: {os.path.abspath(path)}")
        except OSError as e:
            print(f"could not write the log: {e}")


def _list_ports(rep: _Report):
    try:
        from serial.tools import list_ports
        ports = list(list_ports.comports())
    except Exception as e:
        rep.say(f"      (could not list serial ports: {e})")
        return
    rep.say("      serial ports on this PC:" if ports else "      no serial ports on this PC (USB cable, supply on?)")
    for p in ports:
        rep.say(f"        {p.device:8s} {p.description}  [{p.hwid}]")


def _meter_check(drv: RSDSupply, rep: _Report, ch: int, v_set: float):
    """The operator's multimeter on the terminals vs the set voltage and the supply's own reading."""
    v_read = drv.measure()["v"]
    x = rep.ask_number(f"      Multimeter (V DC) on CH{ch} + and -: reading in V? (Enter = skip) ")
    if x is None:
        rep.say(f"      multimeter check at {v_set:.2f} V skipped (supply reads {v_read:.3f} V at its terminals)")
        return
    rep.check(abs(x - v_set) <= 0.05, f"multimeter {x:.3f} V at {v_set:.2f} V set (supply reads {v_read:.3f} V, "
                                      f"meter - supply = {(x - v_read) * 1000:+.0f} mV; pass = within 50 mV of set)")


def _probe_spacing(drv: RSDSupply, rep: _Report, ch: int):
    """The supply ends a command by silence: send set + read-back pairs closer and closer together
    (output off) and see where it stops understanding them."""
    rep.say()
    rep.say("Command spacing (output off): the supply only knows a command ended when the line goes quiet.")
    saved, good = drv.gap_s, []
    try:
        for gap in (0.06, 0.04, 0.03, 0.02, 0.01):
            drv.gap_s = gap
            n_ok = 0
            for k in range(6):
                v = 1.1 + 0.1 * (k % 4)
                drv.write(f"VSET{ch}:{v:05.2f}")
                try:
                    n_ok += abs(drv._query_float(f"VSET{ch}?") - v) < 0.006
                except SupplyError:
                    pass
            rep.say(f"      {gap * 1000:3.0f} ms between commands: {n_ok}/6 set + read-back pairs understood")
            if n_ok == 6:
                good.append(gap)
    finally:
        drv.gap_s = saved
        if getattr(drv, "ser", None) is not None:
            time.sleep(0.3)
            drv._drain()
    rep.check(saved in good, f"commands understood at the driver's {saved * 1000:.0f} ms spacing"
                             + (f" (worked down to {min(good) * 1000:.0f} ms)" if good else ""))
    back = drv.set_voltage(1.0)
    rep.check(bool(drv.query("*IDN?")) and abs(back - 1.0) < 1e-6, "supply answers normally after the fast commands")


def _load_test(drv: RSDSupply, rep: _Report, ch: int, v: float = 2.0):
    """Current limit with a load: step the limit down, log V / I / CC, check that the voltage drops once
    the supply limits and that the dashboard's CC trip (is_limiting) sees it, and doesn't see a load
    that is below the limit."""
    rep.say()
    rep.say(f"Current-limit test: needs a load on CH{ch}: a resistor (47-100 ohm, at least 0.25 W), or the")
    rep.say("multimeter on its mA range across the terminals (that is a short: the supply limits at every step).")
    drv.output(False)
    if rep.interactive and rep.ask("      Output is OFF. Connect the load, then Enter (s = skip) ... ").lower().startswith("s"):
        rep.say("      current-limit test skipped")
        return
    try:
        drv.set_current(0.060)
        drv.set_voltage(v)
        drv.output(True)
        time.sleep(1.0)
        m0 = drv.measure()
        rep.say(f"      {v:.2f} V, limit 60 mA: reads {m0['v']:.3f} V / {m0['i'] * 1000:.1f} mA, status "
                f"{'CV' if m0['cv'] else 'CC'}")
        if m0["i"] < 0.002:
            rep.check(False, "current-limit test: no current flows (is the load connected?)")
            return
        short = m0["v"] < v - 0.05            # the load wants 60 mA or more
        i_cv = None if short else m0["i"]
        rep.say("      load wants >= 60 mA (short or low resistance)" if short else
                f"      load ~{m0['v'] / m0['i']:.0f} ohm, draws {m0['i'] * 1000:.1f} mA at {v:.2f} V")
        rep.say(f"      {'limit':>8}  {'V out':>8}  {'I out':>8}  status bit  dashboard sees")
        rows = []
        for lim_ma in (60, 40, 30, 20, 10, 5):
            drv.set_current(lim_ma / 1000.0)
            time.sleep(0.8)
            m = drv.measure()
            lim = is_limiting(m, v, lim_ma / 1000.0)
            rows.append((lim_ma, m, lim))
            rep.say(f"      {lim_ma:>5} mA  {m['v']:>6.3f} V  {m['i'] * 1000:>5.1f} mA  {'CV' if m['cv'] else 'CC':>10}  "
                    f"{'CC -> would trip' if lim else 'CV, ok'}")
        want_cc = [r for r in rows if i_cv is None or r[0] / 1000.0 < 0.7 * i_cv]
        want_cv = [r for r in rows if i_cv is not None and r[0] / 1000.0 > 1.3 * i_cv]
        if want_cc:
            rep.check(all(m["v"] < v - 0.05 and abs(m["i"] - l / 1000.0) <= max(0.002, 0.1 * l / 1000.0)
                          for l, m, _ in want_cc),
                      f"limited at {', '.join(str(l) for l, _, _ in want_cc)} mA: voltage drops, current = the limit")
            rep.check(all(not m["cv"] for _, m, _ in want_cc), "status bit reads CC while limiting")
            rep.check(all(lim for _, _, lim in want_cc), "the dashboard's CC trip would fire while limiting")
        if want_cv:
            rep.check(not any(lim for _, _, lim in want_cv),
                      f"no false trip at {', '.join(str(l) for l, _, _ in want_cv)} mA (load below the limit)")
        if short and rep.interactive:
            drv.set_current(0.020)
            time.sleep(0.8)
            x = rep.ask_number("      Limit now 20 mA. Multimeter on mA: reading in mA? (Enter = skip) ")
            if x is not None:
                rep.check(abs(x - 20.0) <= 2.0, f"multimeter {x:.1f} mA at a 20 mA limit")
    finally:
        drv.set_current(0.060)
        drv.output(False)
        if rep.interactive:
            rep.ask("      Output is OFF, limit back at 60 mA. Disconnect the load, then Enter ... ")


def probe(address: str, log_dir: str = ".") -> int:
    """No answer to *IDN?: try baud rates x line endings x DTR/RTS with harmless queries only (Korad
    and SCPI style) and print every answer. Stops after the first baud rate that answers."""
    import serial
    queries = ["*IDN?", "VSET1?", "STATUS?"]           # SCPI instruments answer *IDN? too
    eols = [("none", ""), ("LF", "\n"), ("CRLF", "\r\n"), ("CR", "\r")]
    rep = _Report(False)
    rep.say(f"RSD3305P probe {time.strftime('%Y-%m-%d %H:%M:%S')}, address {address}: queries only, nothing is set")
    rep.say("Watch the supply's display: a remote / lock indicator lighting up means it hears us.")
    hits = []
    url = parse_address(address)
    try:
        for dtr in (True, False):
            for baud in (9600, 115200, 19200, 38400, 57600, 4800):
                try:
                    ser = serial.serial_for_url(url, baudrate=baud, timeout=0.05, do_not_open=True)
                    ser.dtr = ser.rts = dtr
                    ser.open()
                except Exception as e:
                    rep.say(f"FAIL  cannot open {url} at {baud}: {e}")
                    _list_ports(rep)
                    rep.save(log_dir, "supply_probe")
                    return 1
                time.sleep(0.2)
                n_before = len(hits)
                try:
                    for eol_name, eol in eols:
                        for q in queries:
                            ser.reset_input_buffer()
                            ser.write((q + eol).encode("ascii"))
                            ser.flush()
                            data, t_end = b"", clock() + 0.4
                            while clock() < t_end:
                                chunk = ser.read(64)
                                if chunk:
                                    data += chunk
                                elif data:
                                    break
                            if data:
                                hits.append((baud, eol_name, dtr, q, data))
                                rep.say(f"ANSWER  {baud} baud, end {eol_name:4s}, DTR/RTS {'on' if dtr else 'off'}: "
                                        f"{q!r} -> {data!r} (hex {data.hex(' ')})")
                            time.sleep(0.12)        # silence: ends the command on a terminator-less supply
                finally:
                    ser.close()
                rep.say(f"      {baud:6d} baud, DTR/RTS {'on' if dtr else 'off'}: {len(hits) - n_before} answers")
                if len(hits) > n_before:
                    break
            if hits:
                break
    except KeyboardInterrupt:
        rep.say("stopped by Ctrl+C")
    rep.say()
    if hits:
        b, e, d, _, _ = hits[0]
        rep.say(f"It answers at {b} baud, line end {e}, DTR/RTS {'on' if d else 'off'}. Send this log to Nic's Claude session.")
        rep.say(f"Then try: python supply_rsd.py {address} --selftest --baud {b}" + (f" --eol {e}" if e != "none" else ""))
    else:
        rep.say("No answer at any setting. Check: USB cable in the supply's USB port (not only power), supply")
        rep.say("switched on, no other program (e.g. the maker's PC software) using the port, and on the supply")
        rep.say("whether remote control has to be enabled or the interface set to USB in a menu.")
    rep.save(log_dir, "supply_probe")
    return 0 if hits else 1


def selftest(address: str, channel: int = 1, eol: str = "", assume_yes: bool = False, log_dir: str = ".",
             load: bool = True, only_load: bool = False, baud: int = 9600) -> int:
    """Everything the dashboard relies on: connection and identity, set + read back, output on/off,
    the voltage at the terminals (supply reading, and the operator's multimeter if typed in), steps up
    to 2.5 V and how fast the output follows, what the STATUS bits mean on this unit, the spacing the
    supply needs between commands, refusal above 2.5 V, repeated reads, and (with a load connected
    when asked) the current limit. Restores the old settings with the output off. Writes
    supply_selftest_<time>.txt. Returns the number of failed checks."""
    rep = _Report(not assume_yes)
    drv = make_supply({"address": address, "channel": channel, "max_mv": 2500, "eol": eol, "baud": baud})
    rep.say(f"RSD3305P self-test {time.strftime('%Y-%m-%d %H:%M:%S')}, address {drv.address}, channel {channel}")
    t = clock()
    try:
        idn = drv.open()
    except Exception as e:
        rep.check(False, f"connect / *IDN?: {e}")
        _list_ports(rep)
        rep.save(log_dir)
        return 1
    ch = channel
    rep.check(True, f"connected: *IDN? -> {idn!r} ({clock() - t:.2f} s incl. opening the port)")
    orig, orig_out = None, False
    try:
        t = clock()
        orig = drv.read_settings()
        rep.say(f"      one query + answer takes ~{(clock() - t) / 2 * 1000:.0f} ms")
        try:
            ch2 = drv._query_float("VSET2?") if ch == 1 else None
        except SupplyError:
            ch2 = None
        raw = drv.query_raw("STATUS?")
        orig_out = bool(raw and raw[0] & 0x40)
        rep.say(f"      now: CH{ch} set {orig['v_set']:.2f} V / {orig['i_set'] * 1000:.0f} mA, output "
                f"{'ON' if orig_out else 'off'}, status {_bits(raw[0] if raw else None)}")
        rep.say()
        rep.say(f"The test switches the outputs ON (all channels together) at up to 2.5 V / 60 mA on CH{ch}"
                + (f"; CH2 is set to {ch2:.2f} V and will be live too" if ch2 is not None else "") + ".")
        rep.say("Only the multimeter (V DC range) on the outputs" + (" for now." if not only_load else
                "; the load test asks for the load when it needs it."))
        if rep.interactive:
            try:
                rep.ask("Press Enter to start, Ctrl+C to stop ... ")
            except (KeyboardInterrupt, EOFError):
                rep.say("stopped, nothing changed")
                drv.close()
                return 1

        if not only_load:
            drv.output(False)
            time.sleep(0.5)
            raw = drv.query_raw("STATUS?")
            st_off = raw[0] if raw else None
            rep.say(f"      output off: status {_bits(st_off)}")
            rep.check(st_off is None or not st_off & 0x40, "status bit 6 reads 0 with the output off")
            if st_off is None:
                rep.say("      (STATUS? is silent with the output off: fine, the dashboard reads it with the output on)")

            _probe_spacing(drv, rep, ch)

            rep.say()
            rep.say("Voltage at the terminals:")
            rep.check(abs(drv.set_current(0.060) - 0.060) < 1e-6, "current limit 60 mA set and read back")
            rep.check(abs(drv.set_voltage(1.0) - 1.0) < 1e-6, "1.00 V set and read back")
            t = clock()
            drv.output(True)
            m = drv.wait_for(1.0, 0.05, 10.0)
            rep.check(True, f"output on: supply reads {m['v']:.3f} V / {m['i'] * 1000:.1f} mA after {clock() - t:.2f} s")
            st_on = drv.status()
            rep.say(f"      output on, no load: status {_bits(st_on)}")
            rep.check(bool(st_on & 0x40), "status bit 6 reads 1 with the output on")
            if not rep.check(bool(st_on >> (ch - 1) & 1), f"CH{ch} status bit reads CV (1) with no load"):
                rep.say("      (the dashboard's CC trip also checks the V/I readings, so this alone doesn't block runs)")
            _meter_check(drv, rep, ch, 1.0)

            rep.say()
            rep.say("Steps (time until the output reads the new voltage, = the wait in each voltage block):")
            for v in (1.5, 2.0, 2.5, 0.5, 1.0):
                t = clock()
                drv.set_voltage(v)
                try:
                    m = drv.wait_for(v, 0.05, 10.0)
                    rep.check(True, f"step to {v:.2f} V: reads {m['v']:.3f} V after {clock() - t:.2f} s")
                except SupplyError as e:
                    rep.check(False, f"step to {v:.2f} V: {e}")
                if v == 2.5:
                    _meter_check(drv, rep, ch, 2.5)
            try:
                drv.set_voltage(3.0)
                rep.check(False, "3.0 V refused (above the 2.5 V maximum)")
            except SupplyError:
                rep.check(True, "3.0 V refused before sending (above the 2.5 V maximum)")

            bad, t = 0, clock()
            for _ in range(20):
                try:
                    bad += abs(drv.measure()["v"] - 1.0) > 0.05
                except SupplyError:
                    bad += 1
            rep.check(bad == 0, f"20 read cycles (V, I, status): {bad} bad, {(clock() - t) / 20 * 1000:.0f} ms per cycle")
            t, ok = clock(), True
            try:
                for k in range(10):
                    drv.set_voltage(1.0 + 0.1 * (k % 5))
            except SupplyError as e:
                ok = False
                rep.say(f"      {e}")
            rep.check(ok, f"10 quick set + read-back cycles ({(clock() - t) / 10 * 1000:.0f} ms each)")

            drv.output(False)
            time.sleep(1.0)
            raw = drv.query_raw("STATUS?")
            v_off = drv._query_float(f"VOUT{ch}?")
            rep.say(f"      output off again: status {_bits(raw[0] if raw else None)}, supply reads {v_off:.3f} V")
            rep.check(not (raw and raw[0] & 0x40), "output switched off")
            x = rep.ask_number("      Multimeter with the output off (should be ~0 V): reading in V? (Enter = skip) ")
            if x is not None:
                rep.check(abs(x) < 0.1, f"multimeter {x:.3f} V with the output off")

        if (load and rep.interactive) or only_load:
            _load_test(drv, rep, ch)
    except KeyboardInterrupt:
        rep.say("stopped by Ctrl+C")
        rep.fails.append("stopped by Ctrl+C")
    except Exception as e:
        rep.check(False, f"stopped by an error: {e}")
    finally:
        try:
            drv.output(False)
            if orig is not None:
                drv.write(f"VSET{ch}:{orig['v_set']:05.2f}")
                drv.write(f"ISET{ch}:{orig['i_set']:05.3f}")
                rep.say(f"      restored CH{ch} {orig['v_set']:.2f} V / {orig['i_set'] * 1000:.0f} mA, output left OFF"
                        + (" (it was ON before)" if orig_out else ""))
        except Exception as e:
            rep.say(f"      could not restore the settings: {e}")
        drv.close()
    rep.say()
    rep.say("ALL PASSED" if not rep.fails else f"{len(rep.fails)} FAILED: " + "; ".join(rep.fails))
    rep.save(log_dir)
    return len(rep.fails)


def main(argv=None):
    """Bench check without the dashboard:
        python supply_rsd.py COM7                  identity, settings, output reading (changes nothing)
        python supply_rsd.py COM7 --selftest       full check: multimeter prompts, current limit with a load
        python supply_rsd.py COM7 --load-test      only the current-limit test (resistor / meter on mA)
        python supply_rsd.py labpc:5025 --set 1.5  set 1.5 V (limit --limit-ma, default 60), output on, read
        python supply_rsd.py COM7 --off            output off
        python supply_rsd.py COM7 --probe          no answer? try baud rates / line endings (queries only)"""
    import argparse
    ap = argparse.ArgumentParser(description="RSD3305P bench check")
    ap.add_argument("address", help="COM7, host:port (supply_bridge.py) or dummy")
    ap.add_argument("--ch", type=int, default=1)
    ap.add_argument("--set", type=float, default=None, help="volts (at most --max)")
    ap.add_argument("--max", type=float, default=2.5, help="refuse voltages above this (default 2.5 V)")
    ap.add_argument("--limit-ma", type=float, default=60.0)
    ap.add_argument("--off", action="store_true")
    ap.add_argument("--eol", default="", help="LF / CR / CRLF if the firmware wants a terminator")
    ap.add_argument("--selftest", action="store_true",
                    help="full check (multimeter prompts, then a current-limit test with a load); writes a log")
    ap.add_argument("--load-test", action="store_true", help="only the current-limit test (load on the output)")
    ap.add_argument("--no-load", action="store_true", help="self-test without the current-limit part")
    ap.add_argument("--yes", action="store_true", help="no prompts (skips the multimeter and load parts)")
    ap.add_argument("--baud", type=int, default=9600)
    ap.add_argument("--probe", action="store_true", help="find baud rate / line ending when *IDN? gets no answer")
    a = ap.parse_args(argv)
    if a.probe:
        return probe(a.address)
    if a.selftest or a.load_test:
        return 1 if selftest(a.address, a.ch, a.eol, a.yes, load=not a.no_load, only_load=a.load_test,
                             baud=a.baud) else 0
    cfg = {"address": a.address, "channel": a.ch, "max_mv": a.max * 1000, "eol": a.eol, "baud": a.baud}
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
    import sys
    sys.exit(main())
