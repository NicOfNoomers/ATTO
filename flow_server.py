#!/usr/bin/env python3
"""
ATTO flow server: share one Fluigent controller's flow sensors with many experiments.

The Fluigent SDK session (fgt_init) can only be owned by one process. This server owns
it, samples every sensor channel as fast as the hardware allows, logs every fresh sample
to CSV (whether or not anyone is connected), and streams the data over WebSocket so
several experiments (dashboards on this PC or others on the network) can each subscribe
to the channels they need, at their own rate.

Usage
-----
    python flow_server.py                              # real hardware, all sensors, ws://0.0.0.0:8765/
    python flow_server.py --channels 0,2,5             # poll only these sensor indices
    python flow_server.py --backend dummy --dummy-channels 4 --dummy-rate 50
    python flow_server.py --poll-hz 0 --no-log         # poll without sleeping, no disk logging
    python flow_server.py --log-dir D:\\flow_logs --rotate-hours 6 --port 9000

Stop with Ctrl+C (Ctrl+Break also works).

Threads
-------
- sampler    : owns EVERY SDK call (fgt_init, reads, fgt_close). Polls all channels on a
               deadline schedule, keeps only fresh samples (device timestamp changed),
               appends them to lock-free per-channel ring buffers and put_nowait()s them
               into the log queue. It never touches the network or disk and never waits
               on a lock held for more than a trivial operation.
- csv-writer : drains the log queue into flow_YYYYmmdd_HHMMSS.csv (rotated).
- main       : asyncio WebSocket server. Every client has its own subscription, ring
               cursors, rate and bounded outgoing backlog (drop-oldest), so a slow client
               only hurts itself.
- logging    : console output goes through a queue, so no thread can stall on a Windows
               console that is in text-selection mode.

Protocol v1 (WebSocket text frames, one JSON object per frame)
-------------------------------------------------------------
server -> client on connect:
    {"type":"hello","protocol":1,"server":"atto-flow-server","backend":"fluigent"|"dummy",
     "channels":[{"ch":0,"sensor_type":"...","serial":"..."|null,"unit":"..."}, ...],
     "server_time":<epoch s>}
client -> server (the server is read-only: no command touches the hardware):
    {"cmd":"subscribe","channels":[0,3]|"all","rate":<Hz>|"max","mode":"latest"|"all"|"mean"}
        -> {"type":"subscribed","channels":[...],"rate":...,"mode":...}
    {"cmd":"unsubscribe"}  -> {"type":"unsubscribed"}
    {"cmd":"info"}         -> {"type":"info","channels":[...],"stats":{...}}
    {"cmd":"ping"}         -> {"type":"pong","server_time":<epoch s>}
    anything invalid       -> {"type":"error","msg":"..."}  (the connection stays open)
data:
    {"type":"data","samples":[{"ch":0,"seq":12,"t":<epoch s>,"dev_t":<ms>|null,
                               "v":<float>|null,"ok":true}, ...]}
    mean mode: one sample per channel per tick with "n","min","max"; "ok" is n > 0.

CSV columns: t_wall_s,ch,seq,dev_t_ms,value,ok  (ok is 1/0; dev_t_ms/value empty when missing)
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import logging
import logging.handlers
import math
import os
import queue
import random
import signal
import socket
import sys
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed

LOG = logging.getLogger("flow_server")

SERVER_NAME = "atto-flow-server"
PROTOCOL_VERSION = 1
MODES = ("latest", "all", "mean")
CSV_HEADER = "t_wall_s,ch,seq,dev_t_ms,value,ok\n"

ERROR_SAMPLE_MIN_INTERVAL_S = 0.1   # at most one error sample per channel per 100 ms
RESYNC_AFTER_S = 0.05               # sampler this far behind schedule -> restart schedule, no burst
LOG_QUEUE_MAX = 200_000             # samples waiting for the CSV writer before we start dropping
LOG_BATCH_MAX = 20_000
LOG_FLUSH_INTERVAL_S = 1.0
LOG_FSYNC_INTERVAL_S = 5.0
RING_MIN, RING_MAX = 1024, 262_144  # per-channel ring buffer bounds (samples)
UNPACED_RATE_GUESS_HZ = 5000.0      # ring sizing when --poll-hz 0
MAX_CLIENT_RATE_HZ = 1000.0
MAX_RATE_MIN_INTERVAL_S = 0.005     # rate "max": at most one frame per 5 ms per client
CLIENT_BACKLOG_FRAMES = 256         # queued data frames per client before drop-oldest
CLIENT_CTRL_BACKLOG = 64
MAX_SAMPLES_PER_FRAME = 5000        # keeps frames well below the common 1 MiB client limit
MAX_INCOMING_BYTES = 64 * 1024

# One sample, as stored in the ring buffers and the log queue:
# (ch, seq, t_wall_s, dev_t_ms | None, value | None, ok)
Sample = Tuple[int, int, float, Optional[int], Optional[float], bool]


# ---------------- logging ----------------
def setup_logging() -> logging.handlers.QueueListener:
    """Send all log records through a queue drained by a listener thread.

    A Windows console in QuickEdit/selection mode blocks every thread that writes to it;
    with the queue, only the listener thread can stall, never the sampler or the server.
    """
    log_q: "queue.SimpleQueue[logging.LogRecord]" = queue.SimpleQueue()
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    listener = logging.handlers.QueueListener(log_q, console)
    root = logging.getLogger()
    root.handlers[:] = [logging.handlers.QueueHandler(log_q)]
    root.setLevel(logging.INFO)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    listener.start()
    return listener


class Throttle:
    """Per-key rate limiter for log messages (use one instance per thread)."""

    def __init__(self) -> None:
        self._last: Dict[Any, float] = {}

    def ready(self, key: Any, interval_s: float) -> bool:
        now = time.monotonic()
        if now - self._last.get(key, -math.inf) >= interval_s:
            self._last[key] = now
            return True
        return False


def raise_timer_resolution() -> Callable[[], None]:
    """Ask Windows for 1 ms timer resolution; returns the function that releases it.

    Without it, asyncio timers (fixed-rate clients, max-rate coalescing) are quantised
    to the default 15.6 ms system tick.
    """
    if sys.platform != "win32":
        return lambda: None
    try:
        winmm = ctypes.WinDLL("winmm")
        if winmm.timeBeginPeriod(1) == 0:
            return lambda: winmm.timeEndPeriod(1)
    except Exception:
        LOG.warning("Could not raise the Windows timer resolution; client timing will be coarser.")
    return lambda: None


# ---------------- channels & backends ----------------
@dataclass
class ChannelInfo:
    ch: int                      # SDK sensor index (what clients subscribe to)
    sensor_type: str
    serial: Optional[str]
    unit: str
    details: Dict[str, Any] = field(default_factory=dict)  # sidecar JSON only

    def public(self) -> Dict[str, Any]:
        return {"ch": self.ch, "sensor_type": self.sensor_type, "serial": self.serial, "unit": self.unit}


class BackendError(RuntimeError):
    """The backend could not be opened. The message is shown to the user as-is."""


class FluigentBackend:
    """Fluigent SDK backend. Every method must be called from the sampler thread."""

    name = "fluigent"

    def __init__(self) -> None:
        self._sdk: Any = None
        self._read_ex: Any = None
        self._error_enum: Any = None

    def open(self) -> List[ChannelInfo]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # the SDK's pkg_resources use warns on import
                from Fluigent import SDK as sdk
                from Fluigent.SDK import low_level
        except Exception as e:
            raise BackendError(
                f"The Fluigent SDK could not be imported ({type(e).__name__}: {e}).\n"
                "Install it (pip install fluigent_sdk), or run with --backend dummy to test without hardware."
            ) from e
        self._sdk = sdk
        try:
            # Failed calls then just return error codes instead of printing warnings.
            sdk.fgt_set_errorReportMode("None")
            init_err = sdk.fgt_init()
            count = int(sdk.fgt_get_sensorChannelCount() or 0)
            if count == 0:
                raise BackendError(
                    f"No Fluigent sensor channels found (fgt_init returned {int(init_err)}: {init_err}).\n"
                    "Check that the Flowboard/FRP is plugged in and powered, and that no other program "
                    "(pump_test_dashboard.py, Fluigent OxyGEN, another flow_server) holds it.\n"
                    "To test without hardware, run with --backend dummy."
                )
            infos, types = sdk.fgt_get_sensorChannelsInfo()
            channels = []
            for i in range(count):
                info = infos[i] if i < len(infos) else None
                sensor_type = str(types[i]) if i < len(types) else "unknown"
                try:
                    unit = str(sdk.fgt_get_sensorUnit(i))
                except Exception:
                    unit = "?"
                details: Dict[str, Any] = {}
                serial = None
                if info is not None:
                    serial = str(info.DeviceSN) if info.DeviceSN else None
                    details = {
                        "controller_sn": info.ControllerSN,
                        "device_sn": info.DeviceSN,
                        "firmware": f"{info.firmware >> 8:x}.{info.firmware & 0xFF:02x}",
                        "position": info.position,
                        "index": info.index,
                        "index_id": info.indexID,
                        "instr_type": str(info.InstrType),
                    }
                channels.append(ChannelInfo(i, sensor_type, serial, unit, details))
        except BaseException:
            self.close()
            raise
        # One DLL call per read. The high-level fgt_get_sensorValue() returns the same
        # (error, value, timestamp) but also calls fgt_get_sensorStatus() after every
        # read, only to print warnings.
        self._read_ex = low_level.fgt_get_sensorValueEx
        self._error_enum = low_level.fgt_ERROR
        return channels

    def read(self, ch: int) -> Tuple[int, float, int]:
        """Return (error_code, value, device_timestamp_ms); the timestamp is a 16-bit ms counter."""
        return self._read_ex(ch)

    def describe_error(self, code: int) -> str:
        try:
            return f"{code} ({self._error_enum(code)})"
        except Exception:
            return str(code)

    def close(self) -> None:
        if self._sdk is not None:
            sdk, self._sdk = self._sdk, None
            sdk.fgt_close()


class DummyBackend:
    """Synthetic sensors that behave like the SDK.

    Each channel holds a cached value that only updates at --dummy-rate Hz and carries
    a 16-bit millisecond device timestamp (which wraps, like the real one), so polling
    faster than the update rate returns repeats and exercises the dedupe path exactly
    as the hardware does. Values are rounded to float32 like the DLL's c_float.
    """

    name = "dummy"
    BASE_FLOWS = (2.0, 5.0, 10.0, 20.0, 1.0, 50.0, 0.5, 100.0, 15.0, 30.0)  # µl/min

    def __init__(self, n_channels: int, rate_hz: float, fail_channels: Optional[List[int]] = None) -> None:
        self.n = n_channels
        self.rate = rate_hz
        self.fail = set(fail_channels or [])
        self._rng = random.Random()
        self._t0 = 0.0

    def open(self) -> List[ChannelInfo]:
        if self.n < 1:
            raise BackendError("--dummy-channels must be at least 1.")
        n = self.n
        self._t0 = time.perf_counter()
        self._base = [self.BASE_FLOWS[ch % len(self.BASE_FLOWS)] for ch in range(n)]
        self._freq = [round(0.05 * (ch + 1), 3) for ch in range(n)]  # Hz, distinct per channel
        self._phase = [0.7 * ch for ch in range(n)]
        self._update_phase = [ch / n for ch in range(n)]          # channels update at staggered times
        self._offset_ms = [60_000 + 1_500 * ch for ch in range(n)]  # timers wrap within the first minute
        self._k = [-1] * n
        self._value = [0.0] * n
        self._dev_t = [0] * n
        return [
            ChannelInfo(ch, "Dummy_flow", f"DUMMY-{ch:02d}", "µl/min",
                        {"simulated": True, "base_ul_min": self._base[ch], "sine_hz": self._freq[ch],
                         "update_hz": self.rate, "fails": ch in self.fail})
            for ch in range(n)
        ]

    def read(self, ch: int) -> Tuple[int, float, int]:
        if ch in self.fail:
            return 3, 0.0, 0  # No_module_at_index, as for an unplugged sensor
        k = int((time.perf_counter() - self._t0) * self.rate + self._update_phase[ch])
        if k != self._k[ch]:
            self._k[ch] = k
            ts = k / self.rate
            base = self._base[ch]
            v = (base + 0.15 * base * math.sin(2 * math.pi * self._freq[ch] * ts + self._phase[ch])
                 + self._rng.gauss(0.0, 0.01 * base))
            self._value[ch] = ctypes.c_float(v).value
            self._dev_t[ch] = (self._offset_ms[ch] + round(ts * 1000.0)) & 0xFFFF
        return 0, self._value[ch], self._dev_t[ch]

    def describe_error(self, code: int) -> str:
        return f"{code} (simulated sensor failure)"

    def close(self) -> None:
        pass


# ---------------- ring buffer ----------------
class SampleRing:
    """Fixed-size ring of one channel's samples, indexed by seq.

    Single writer (sampler thread), lock-free readers (asyncio loop). The writer stores
    the slot first and then publishes `head`; seqs are contiguous, so seq s lives in
    slot s % capacity. A reader that gets lapped sees slots holding newer seqs and drops
    them, which shows up to the client as a seq gap.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._slots: List[Optional[Sample]] = [None] * capacity
        self.head = -1  # seq of the newest published sample

    def append(self, sample: Sample) -> None:
        self._slots[sample[1] % self.capacity] = sample
        self.head = sample[1]

    def latest(self) -> Optional[Sample]:
        head = self.head
        return None if head < 0 else self._slots[head % self.capacity]

    def since(self, cursor: int) -> List[Sample]:
        """Samples with seq > cursor still in the buffer, oldest first."""
        head = self.head
        if head <= cursor:
            return []
        cap = self.capacity
        start = max(cursor + 1, head - cap + 1)
        i0, i1 = start % cap, head % cap
        if i0 <= i1:
            items = self._slots[i0:i1 + 1]
        else:
            items = self._slots[i0:] + self._slots[:i1 + 1]
        return [s for s in items if s is not None and s[1] <= head]


# ---------------- sampler thread ----------------
class Sampler(threading.Thread):
    """Acquisition thread: owns the backend (every SDK call happens here)."""

    def __init__(self, backend: Any, wanted: Optional[List[int]], poll_hz: float, dedupe: bool,
                 log_queue: Optional[queue.Queue], buffer_seconds: float) -> None:
        super().__init__(name="sampler", daemon=True)
        self.backend = backend
        self.wanted = wanted
        self.poll_hz = poll_hz
        self.dedupe = dedupe
        self.log_queue = log_queue
        rate = poll_hz if poll_hz > 0 else UNPACED_RATE_GUESS_HZ
        self.ring_capacity = int(min(RING_MAX, max(RING_MIN, buffer_seconds * rate)))
        self.notify: Optional[Callable[[], None]] = None  # set before go(); called after polls with new samples
        self.channels: List[ChannelInfo] = []
        self.channel_ids: List[int] = []
        self.rings: Dict[int, SampleRing] = {}
        self.open_error: Optional[BackendError] = None
        # Cumulative counters, written only by this thread (index = position in channel_ids).
        self.loops = 0
        self.fresh: List[int] = []
        self.errors: List[int] = []
        self.emitted: List[int] = []
        self.dropped_log = 0
        self._running = True
        self._ready = threading.Event()
        self._go = threading.Event()

    # -- control (called from the main thread) --
    def wait_ready(self) -> List[ChannelInfo]:
        """Block until the backend is open; raise BackendError if it failed."""
        while not self._ready.wait(0.25):
            if not self.is_alive():
                break
        if self.open_error is not None:
            raise self.open_error
        if not self._ready.is_set():
            raise BackendError("The sampler thread exited during startup.")
        return self.channels

    def go(self) -> None:
        self._go.set()

    def stop(self) -> None:
        self._running = False
        self._go.set()

    def counters(self) -> Tuple[int, List[int], List[int]]:
        return self.loops, list(self.fresh), list(self.errors)

    # -- thread body --
    def run(self) -> None:
        try:
            detected = self.backend.open()
            self.channels = self._select(detected)
        except BaseException as e:
            if isinstance(e, BackendError):
                self.open_error = e
            else:
                self.open_error = BackendError(f"The {self.backend.name} backend failed to start: {type(e).__name__}: {e}")
            self._close_backend()
            self._ready.set()
            return
        self.channel_ids = [c.ch for c in self.channels]
        self.rings = {ch: SampleRing(self.ring_capacity) for ch in self.channel_ids}
        n = len(self.channel_ids)
        self.fresh, self.errors, self.emitted = [0] * n, [0] * n, [0] * n
        self._ready.set()
        self._go.wait()
        try:
            if self._running:
                self._poll_loop()
        except Exception:
            LOG.exception("Sampler crashed; acquisition has stopped.")
        finally:
            self._close_backend()

    def _select(self, detected: List[ChannelInfo]) -> List[ChannelInfo]:
        if self.wanted is None:
            return detected
        by_ch = {c.ch: c for c in detected}
        missing = [ch for ch in self.wanted if ch not in by_ch]
        if missing:
            raise BackendError(f"--channels {missing} not available; detected channels: {sorted(by_ch)}.")
        return [by_ch[ch] for ch in sorted(set(self.wanted))]

    def _close_backend(self) -> None:
        try:
            self.backend.close()
        except Exception:
            LOG.exception("Error while closing the %s backend", self.backend.name)

    def _poll_loop(self) -> None:
        ids = self.channel_ids
        n = len(ids)
        rings = [self.rings[ch] for ch in ids]
        read = self.backend.read
        notify = self.notify
        log_put = self.log_queue.put_nowait if self.log_queue is not None else None
        dedupe = self.dedupe
        fresh, errors, emitted = self.fresh, self.errors, self.emitted
        last_dev: List[Optional[int]] = [None] * n
        has_ts = [False] * n            # channel has ever reported a non-zero device timestamp
        last_err_sample = [-math.inf] * n
        in_error = [False] * n
        throttle = Throttle()
        wall, perf, sleep, isfinite = time.time, time.perf_counter, time.sleep, math.isfinite
        period = 1.0 / self.poll_hz if self.poll_hz > 0 else 0.0
        deadline = perf()

        while self._running:
            produced = False
            for i in range(n):
                ch = ids[i]
                try:
                    err, v, dev_t = read(ch)
                except Exception:
                    err, v, dev_t = -1, None, None
                    if throttle.ready(("exc", ch), 10.0):
                        LOG.exception("ch %d: read raised an exception", ch)
                t = wall()
                if err == 0 and v is not None and isfinite(v):
                    if dedupe:
                        if dev_t:
                            has_ts[i] = True
                            if dev_t == last_dev[i]:
                                continue
                        elif has_ts[i] and last_dev[i] == 0:
                            continue  # timer wrapped to exactly 0 and is still there
                        last_dev[i] = dev_t
                    if in_error[i]:
                        in_error[i] = False
                        if throttle.ready(("recover", ch), 5.0):
                            LOG.info("ch %d: reading again", ch)
                    sample: Sample = (ch, emitted[i], t, dev_t if (dev_t or has_ts[i]) else None, v, True)
                    fresh[i] += 1
                else:
                    errors[i] += 1
                    if not in_error[i]:
                        in_error[i] = True
                        if throttle.ready(("fail", ch), 5.0):
                            reason = "exception" if err == -1 else self.backend.describe_error(err)
                            LOG.warning("ch %d: read failed (error %s); sending error samples (max %d/s) until it recovers",
                                        ch, reason, round(1 / ERROR_SAMPLE_MIN_INTERVAL_S))
                    now = perf()
                    if now - last_err_sample[i] < ERROR_SAMPLE_MIN_INTERVAL_S:
                        continue
                    last_err_sample[i] = now
                    sample = (ch, emitted[i], t, None, None, False)
                emitted[i] += 1
                rings[i].append(sample)
                produced = True
                if log_put is not None:
                    try:
                        log_put(sample)
                    except queue.Full:
                        self.dropped_log += 1
                        if throttle.ready("log_full", 5.0):
                            LOG.warning("CSV log queue full: %d samples dropped from the log so far (disk too slow?)",
                                        self.dropped_log)
            self.loops += 1
            if produced and notify is not None:
                notify()
            if period:
                deadline += period
                delay = deadline - perf()
                if delay > 0:
                    sleep(delay)
                elif delay < -RESYNC_AFTER_S:
                    deadline = perf()


# ---------------- CSV log writer thread ----------------
def _log_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class CsvLogWriter(threading.Thread):
    """Drains the sampler's log queue into long-format CSV files, rotated by time."""

    def __init__(self, q: queue.Queue, log_dir: str, rotate_s: float, stamp: str) -> None:
        super().__init__(name="csv-writer", daemon=True)
        self.q = q
        self.log_dir = log_dir
        self.rotate_s = rotate_s
        self.rows_written = 0
        self.rows_failed = 0
        self.files: List[str] = []
        self._f: Any = None
        self._opened_at = 0.0
        self._stop_evt = threading.Event()
        self._throttle = Throttle()
        os.makedirs(log_dir, exist_ok=True)
        self._f = self._open(stamp)  # on the main thread, so a bad --log-dir fails at startup

    @property
    def path(self) -> Optional[str]:
        return self.files[-1] if self.files else None

    def stop(self) -> None:
        self._stop_evt.set()

    def _open(self, stamp: str) -> Any:
        base = os.path.join(self.log_dir, f"flow_{stamp}")
        path, n = base + ".csv", 1
        while True:
            try:
                f = open(path, "x", encoding="utf-8", newline="")
                break
            except FileExistsError:
                path, n = f"{base}_{n}.csv", n + 1
        f.write(CSV_HEADER)
        f.flush()
        self._opened_at = time.monotonic()
        self.files.append(os.path.abspath(path))
        return f

    def _close_file(self, f: Any) -> None:
        try:
            f.flush()
            os.fsync(f.fileno())
        except OSError:
            LOG.exception("CSV log: final flush failed for %s", getattr(f, "name", "?"))
        finally:
            f.close()

    def _rotate(self) -> None:
        try:
            new_f = self._open(_log_stamp())
        except OSError:
            LOG.exception("CSV log: could not open a new file; continuing in %s", self.files[-1])
            self._opened_at = time.monotonic()
            return
        old_f, self._f = self._f, new_f
        self._close_file(old_f)
        LOG.info("CSV log rotated -> %s", self.path)

    def _take_batch(self) -> List[Sample]:
        try:
            batch = [self.q.get(timeout=0.25)]
        except queue.Empty:
            return []
        get = self.q.get_nowait
        try:
            while len(batch) < LOG_BATCH_MAX:
                batch.append(get())
        except queue.Empty:
            pass
        return batch

    def _write(self, batch: List[Sample]) -> None:
        if self.rotate_s > 0 and time.monotonic() - self._opened_at >= self.rotate_s:
            self._rotate()
        text = "".join(
            f"{t:.6f},{ch},{seq},{'' if d is None else d},{'' if v is None else format(v, '.9g')},{1 if ok else 0}\n"
            for ch, seq, t, d, v, ok in batch
        )
        try:
            self._f.write(text)
            self.rows_written += len(batch)
        except OSError:
            self.rows_failed += len(batch)
            if self._throttle.ready("write", 10.0):
                LOG.exception("CSV log: write failed; %d rows lost so far", self.rows_failed)

    def run(self) -> None:
        last_flush = last_sync = time.monotonic()
        try:
            while True:
                batch = self._take_batch()
                if batch:
                    self._write(batch)
                now = time.monotonic()
                if now - last_flush >= LOG_FLUSH_INTERVAL_S:
                    last_flush = now
                    try:
                        self._f.flush()
                        if now - last_sync >= LOG_FSYNC_INTERVAL_S:
                            last_sync = now
                            os.fsync(self._f.fileno())
                    except OSError:
                        if self._throttle.ready("flush", 10.0):
                            LOG.exception("CSV log: flush failed")
                if not batch and self._stop_evt.is_set():
                    break
        finally:
            self._close_file(self._f)


def write_sidecar(log_dir: str, stamp: str, backend_name: str, channels: List[ChannelInfo],
                  args: argparse.Namespace, csv_path: Optional[str]) -> str:
    """Write channels_<stamp>.json describing this server run."""
    path = os.path.join(log_dir, f"channels_{stamp}.json")
    doc = {
        "server": SERVER_NAME,
        "protocol": PROTOCOL_VERSION,
        "backend": backend_name,
        "started": datetime.now().astimezone().isoformat(timespec="seconds"),
        "started_epoch_s": time.time(),
        "host": socket.gethostname(),
        "first_csv": csv_path,
        "csv_columns": CSV_HEADER.strip().split(","),
        "csv_notes": "Long format, one row per sample. t_wall_s: server epoch seconds at acquisition. "
                     "seq: per-channel counter from server start (gaps = samples lost from the log). "
                     "dev_t_ms: 16-bit device timer in ms (wraps every 65.536 s; empty if unavailable). "
                     "ok: 1 = valid reading, 0 = read error (value empty).",
        "args": vars(args),
        "channels": [dict(c.public(), **c.details) for c in channels],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    return os.path.abspath(path)


# ---------------- WebSocket side (asyncio) ----------------
def dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))


def sample_dict(s: Sample) -> Dict[str, Any]:
    return {"ch": s[0], "seq": s[1], "t": s[2], "dev_t": s[3], "v": s[4], "ok": s[5]}


def mean_dict(ch: int, samples: List[Sample]) -> Dict[str, Any]:
    vals = [s[4] for s in samples if s[5]]
    n = len(vals)
    last = samples[-1]
    return {"ch": ch, "seq": last[1], "t": last[2], "dev_t": None,
            "v": sum(vals) / n if n else None, "ok": n > 0, "n": n,
            "min": min(vals) if n else None, "max": max(vals) if n else None}


@dataclass(frozen=True)
class Subscription:
    channels: Tuple[int, ...]
    rate_hz: Optional[float]  # None = "max" (event-driven)
    rate_given: Any           # echoed back in the "subscribed" reply
    mode: str


class RateMeter:
    """Turns the sampler's cumulative counters into rates. Used on the asyncio loop only."""

    Snapshot = Tuple[float, int, List[int], List[int]]

    def __init__(self, sampler: Sampler, window_s: float = 5.0) -> None:
        self.sampler = sampler
        self.window_s = window_s
        self._hist: Deque[RateMeter.Snapshot] = deque()
        self.record()

    def snapshot(self) -> "RateMeter.Snapshot":
        loops, fresh, errors = self.sampler.counters()
        return time.monotonic(), loops, fresh, errors

    def record(self) -> "RateMeter.Snapshot":
        snap = self.snapshot()
        self._hist.append(snap)
        while len(self._hist) > 2 and snap[0] - self._hist[1][0] >= self.window_s:
            self._hist.popleft()
        return snap

    def rates(self, old: "RateMeter.Snapshot", new: "RateMeter.Snapshot"
              ) -> Tuple[float, Dict[int, float], Dict[int, float]]:
        """(poll_hz, fresh_hz per channel, error reads/s per channel) between two snapshots."""
        dt = max(new[0] - old[0], 1e-9)
        ids = self.sampler.channel_ids
        return ((new[1] - old[1]) / dt,
                {ch: (b - a) / dt for ch, a, b in zip(ids, old[2], new[2])},
                {ch: (b - a) / dt for ch, a, b in zip(ids, old[3], new[3])})

    def recent(self) -> Tuple[float, Dict[int, float], Dict[int, float]]:
        """Rates over roughly the last window_s seconds."""
        return self.rates(self._hist[0], self.snapshot())


class Hub:
    """asyncio side: client registry, coalesced wakeups from the sampler, info/stats."""

    def __init__(self, sampler: Sampler, backend_name: str, writer: Optional[CsvLogWriter],
                 stats_interval: float) -> None:
        self.sampler = sampler
        self.rings = sampler.rings
        self.channel_ids = list(sampler.channel_ids)
        self.channels_public = [c.public() for c in sampler.channels]
        self.backend_name = backend_name
        self.writer = writer
        self.stats_interval = stats_interval
        self.meter = RateMeter(sampler)
        self.clients: Set[ClientSession] = set()
        self.dropped_frames = 0
        self.sampler_failed = False
        self._max_clients: Set[ClientSession] = set()
        self._n_max = 0
        self._wake_pending = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.stop_event: Optional[asyncio.Event] = None

    def attach(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self.stop_event = asyncio.Event()

    def detach(self) -> None:
        self._loop = None

    def request_stop(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()

    # -- sampler -> loop hand-off --
    def notify_threadsafe(self) -> None:
        """Called on the sampler thread after a poll produced samples.

        At most one wakeup callback is in flight at a time, and none at all unless some
        client subscribed with rate "max" (fixed-rate clients poll on their own timers).
        """
        if self._wake_pending or not self._n_max:
            return
        loop = self._loop
        if loop is None:
            return
        self._wake_pending = True
        try:
            loop.call_soon_threadsafe(self._wake_max_clients)
        except RuntimeError:  # loop closed during shutdown
            self._wake_pending = False

    def _wake_max_clients(self) -> None:
        self._wake_pending = False  # cleared before clients read, so no sample can be missed
        for client in self._max_clients:
            client.wake.set()

    def set_max_rate(self, client: "ClientSession", enabled: bool) -> None:
        if enabled:
            self._max_clients.add(client)
            client.wake.set()
        else:
            self._max_clients.discard(client)
        self._n_max = len(self._max_clients)

    # -- protocol helpers --
    def hello(self) -> Dict[str, Any]:
        return {"type": "hello", "protocol": PROTOCOL_VERSION, "server": SERVER_NAME,
                "backend": self.backend_name, "channels": self.channels_public, "server_time": time.time()}

    def info(self) -> Dict[str, Any]:
        poll_hz, fresh_hz, _ = self.meter.recent()
        return {"type": "info", "channels": self.channels_public, "stats": {
            "poll_hz": round(poll_hz, 2),
            "fresh_hz": {str(ch): round(hz, 2) for ch, hz in fresh_hz.items()},
            "clients": len(self.clients),
            "log_file": self.writer.path if self.writer is not None else None,
            "dropped_log": self.sampler.dropped_log,
        }}

    def parse_subscription(self, obj: Dict[str, Any]) -> Subscription:
        """Validate a subscribe command; raise ValueError with a client-facing message."""
        chans = obj.get("channels", "all")
        if chans == "all":
            channels = tuple(self.channel_ids)
        elif (isinstance(chans, list) and chans
              and all(isinstance(c, int) and not isinstance(c, bool) for c in chans)):
            unknown = sorted({c for c in chans if c not in self.rings})
            if unknown:
                raise ValueError(f"unknown channel(s) {unknown}; available: {self.channel_ids}")
            channels = tuple(sorted(set(chans)))
        else:
            raise ValueError('"channels" must be "all" or a non-empty list of channel numbers')
        rate = obj.get("rate", "max")
        if rate == "max":
            rate_hz = None
        elif (isinstance(rate, (int, float)) and not isinstance(rate, bool)
              and 0 < rate <= MAX_CLIENT_RATE_HZ):
            rate_hz = float(rate)
        else:
            raise ValueError(f'"rate" must be "max" or a number of Hz in (0, {MAX_CLIENT_RATE_HZ:g}]')
        mode = obj.get("mode", "latest")
        if mode not in MODES:
            raise ValueError(f'"mode" must be one of {list(MODES)}')
        return Subscription(channels, rate_hz, rate, mode)

    # -- connections --
    async def handler(self, ws: ServerConnection) -> None:
        client = ClientSession(self, ws)
        self.clients.add(client)
        LOG.info("Client %s connected (%d connected)", client.peer, len(self.clients))
        try:
            await client.run()
        finally:
            self.clients.discard(client)
            self.set_max_rate(client, False)
            extra = f"; {client.dropped_frames} data frames dropped (slow client)" if client.dropped_frames else ""
            LOG.info("Client %s disconnected (%d connected)%s", client.peer, len(self.clients), extra)

    async def close_clients(self, timeout_s: float = 2.0) -> None:
        """Close every connection (1001 going away), aborting those that don't finish in time.

        websockets' close() waits for the send buffer to drain without a timeout, so a
        client that stopped reading would otherwise hold up shutdown until its next
        keepalive ping (up to 20 s).
        """
        conns = [c.ws for c in self.clients]
        if not conns:
            return
        tasks = [asyncio.create_task(ws.close(1001, "server shutting down")) for ws in conns]
        await asyncio.wait(tasks, timeout=timeout_s)
        for ws, task in zip(conns, tasks):
            if not task.done():
                ws.transport.abort()
        await asyncio.wait(tasks, timeout=1.0)

    async def stats_loop(self) -> None:
        """Every second: record counters, watch the sampler; print stats every stats_interval."""
        last_print = self.meter.snapshot()
        while True:
            await asyncio.sleep(1.0)
            snap = self.meter.record()
            if not self.sampler.is_alive():
                LOG.error("The sampler thread has stopped; shutting the server down.")
                self.sampler_failed = True
                self.request_stop()
                return
            if self.stats_interval > 0 and snap[0] - last_print[0] >= self.stats_interval - 0.5:
                self._print_stats(last_print, snap)
                last_print = snap

    def _print_stats(self, old: RateMeter.Snapshot, new: RateMeter.Snapshot) -> None:
        poll_hz, fresh_hz, err_hz = self.meter.rates(old, new)
        fresh = " ".join(f"{ch}:{hz:.1f}" for ch, hz in fresh_hz.items())
        errs = " ".join(f"{ch}:{hz:.0f}" for ch, hz in err_hz.items() if hz > 0)
        line = f"poll {poll_hz:.1f} Hz | fresh Hz {fresh}"
        if errs:
            line += f" | failed reads/s {errs}"
        line += f" | clients {len(self.clients)}"
        if self.dropped_frames:
            line += f" (dropped frames {self.dropped_frames})"
        if self.writer is not None:
            line += (f" | log rows {self.writer.rows_written} queue {self.writer.q.qsize()}"
                     f" dropped {self.sampler.dropped_log}")
        LOG.info("%s", line)


class ClientSession:
    """One WebSocket connection: subscription, ring cursors and outgoing backlog."""

    def __init__(self, hub: Hub, ws: ServerConnection) -> None:
        self.hub = hub
        self.ws = ws
        addr = ws.remote_address
        self.peer = f"{addr[0]}:{addr[1]}" if isinstance(addr, tuple) and len(addr) >= 2 else str(addr)
        self.sub: Optional[Subscription] = None
        self.cursors: Dict[int, int] = {}  # ch -> last seq consumed/sent
        self.wake = asyncio.Event()        # set by the hub when a rate-"max" client has new data
        self.dropped_frames = 0
        self._ctrl: Deque[str] = deque(maxlen=CLIENT_CTRL_BACKLOG)
        self._data: Deque[str] = deque(maxlen=CLIENT_BACKLOG_FRAMES)
        self._has_output = asyncio.Event()
        self._sender: Optional[asyncio.Task] = None
        self._throttle = Throttle()

    async def run(self) -> None:
        writer = asyncio.create_task(self._writer())
        self._send_ctrl(self.hub.hello())
        try:
            async for message in self.ws:
                self._handle(message)
        except ConnectionClosed:
            pass
        finally:
            self._stop_sender()
            writer.cancel()

    # -- outgoing --
    def _send_ctrl(self, obj: Dict[str, Any]) -> None:
        self._ctrl.append(dumps(obj))
        self._has_output.set()

    def _error(self, msg: str) -> None:
        self._send_ctrl({"type": "error", "msg": msg})

    def _push_data(self, text: str) -> None:
        if len(self._data) == CLIENT_BACKLOG_FRAMES:
            self.dropped_frames += 1  # deque drops the oldest frame on append
            self.hub.dropped_frames += 1
            if self._throttle.ready("drop", 10.0):
                LOG.warning("Client %s is not keeping up: %d data frames dropped so far",
                            self.peer, self.dropped_frames)
        self._data.append(text)
        self._has_output.set()

    async def _writer(self) -> None:
        """The only coroutine that sends on this socket; control replies go first."""
        try:
            while True:
                await self._has_output.wait()
                self._has_output.clear()
                while self._ctrl or self._data:
                    msg = self._ctrl.popleft() if self._ctrl else self._data.popleft()
                    await self.ws.send(msg)
        except ConnectionClosed:
            pass

    # -- commands --
    def _handle(self, message: Any) -> None:
        if not isinstance(message, str):
            self._error("binary frames are not supported; send JSON text frames")
            return
        try:
            obj = json.loads(message)
        except ValueError as e:
            self._error(f"malformed JSON: {e}")
            return
        if not isinstance(obj, dict):
            self._error("expected a JSON object")
            return
        cmd = obj.get("cmd")
        if cmd == "subscribe":
            try:
                sub = self.hub.parse_subscription(obj)
            except ValueError as e:
                self._error(str(e))
                return
            self._subscribe(sub)
        elif cmd == "unsubscribe":
            self._unsubscribe()
            self.cursors = {}
            self._send_ctrl({"type": "unsubscribed"})
        elif cmd == "info":
            self._send_ctrl(self.hub.info())
        elif cmd == "ping":
            self._send_ctrl({"type": "pong", "server_time": time.time()})
        else:
            self._error(f"unknown cmd {cmd!r}; expected subscribe, unsubscribe, info or ping")

    def _subscribe(self, sub: Subscription) -> None:
        self._unsubscribe()
        # When a subscription replaces another, channels that stay subscribed keep their
        # cursor (no artificial gap). Others start now: all/mean from the next sample,
        # latest with the current newest sample.
        cursors = {}
        for ch in sub.channels:
            if ch in self.cursors:
                cursors[ch] = self.cursors[ch]
            else:
                head = self.hub.rings[ch].head
                cursors[ch] = head - 1 if sub.mode == "latest" else head
        self.cursors = cursors
        self.sub = sub
        self._send_ctrl({"type": "subscribed", "channels": list(sub.channels),
                         "rate": sub.rate_given, "mode": sub.mode})
        self._sender = asyncio.create_task(self._send_loop(sub))
        self.hub.set_max_rate(self, sub.rate_hz is None)
        LOG.info("Client %s subscribed: channels=%s rate=%s mode=%s",
                 self.peer, list(sub.channels), sub.rate_given, sub.mode)

    def _unsubscribe(self) -> None:
        self._stop_sender()
        self.hub.set_max_rate(self, False)
        self._data.clear()  # frames of the old subscription must not follow the new reply
        self.sub = None

    def _stop_sender(self) -> None:
        if self._sender is not None:
            self._sender.cancel()
            self._sender = None

    # -- data --
    async def _send_loop(self, sub: Subscription) -> None:
        try:
            if sub.rate_hz is None:
                while True:
                    await self.wake.wait()
                    self.wake.clear()
                    self._emit(sub)
                    await asyncio.sleep(MAX_RATE_MIN_INTERVAL_S)
            else:
                loop = asyncio.get_running_loop()
                period = 1.0 / sub.rate_hz
                next_t = loop.time() + period
                while True:
                    await asyncio.sleep(max(0.0, next_t - loop.time()))
                    self._emit(sub)
                    next_t += period
                    if loop.time() - next_t > period:  # fell behind: skip ticks instead of bursting
                        next_t = loop.time() + period
        except asyncio.CancelledError:
            raise
        except Exception:
            LOG.exception("Client %s: data loop failed", self.peer)

    def _emit(self, sub: Subscription) -> None:
        samples: List[Dict[str, Any]] = []
        for ch in sub.channels:
            ring = self.hub.rings[ch]
            cursor = self.cursors[ch]
            if sub.mode == "latest":
                s = ring.latest()
                if s is not None and s[1] > cursor:
                    self.cursors[ch] = s[1]
                    samples.append(sample_dict(s))
                continue
            new = ring.since(cursor)
            if not new:
                continue
            self.cursors[ch] = new[-1][1]
            if sub.mode == "all":
                samples.extend(map(sample_dict, new))
            else:
                samples.append(mean_dict(ch, new))
        for i in range(0, len(samples), MAX_SAMPLES_PER_FRAME):
            self._push_data(dumps({"type": "data", "samples": samples[i:i + MAX_SAMPLES_PER_FRAME]}))


def _local_addresses() -> List[str]:
    try:
        infos = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
        return sorted({ai[4][0] for ai in infos})
    except OSError:
        return []


async def serve_clients(args: argparse.Namespace, hub: Hub) -> None:
    """Run the WebSocket server until Ctrl+C (task cancelled), Ctrl+Break or sampler failure."""
    loop = asyncio.get_running_loop()
    hub.attach(loop)
    prev_break = None
    if hasattr(signal, "SIGBREAK"):
        prev_break = signal.signal(signal.SIGBREAK, lambda *_: loop.call_soon_threadsafe(hub.request_stop))
    stats_task: Optional[asyncio.Task] = None
    try:
        async with serve(hub.handler, args.host, args.port, compression=None,
                         max_size=MAX_INCOMING_BYTES, close_timeout=2) as server:
            ports = sorted({s.getsockname()[1] for s in server.sockets})
            port = ports[0] if ports else args.port
            if args.host in ("0.0.0.0", "", "::"):
                addrs = ", ".join(f"ws://{a}:{port}/" for a in _local_addresses()) or "?"
                LOG.info("Listening on ws://%s:%d/ (all interfaces; this PC: %s)", socket.gethostname(), port, addrs)
            else:
                LOG.info("Listening on ws://%s:%d/", args.host, port)
            stats_task = asyncio.create_task(hub.stats_loop())
            try:
                await hub.stop_event.wait()
            finally:  # also runs when Ctrl+C cancels this task
                LOG.info("Stopping WebSocket server")
                stats_task.cancel()
                await hub.close_clients()
    finally:
        if stats_task is not None:
            stats_task.cancel()
        hub.detach()
        if prev_break is not None:
            signal.signal(signal.SIGBREAK, prev_break)


# ---------------- main ----------------
def _parse_channel_list(text: str) -> List[int]:
    try:
        chans = [int(p) for p in text.replace(" ", "").split(",") if p != ""]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated integers, got {text!r}")
    if not chans or any(c < 0 for c in chans):
        raise argparse.ArgumentTypeError(f"expected non-negative channel numbers, got {text!r}")
    return chans


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Share one Fluigent controller's flow sensors over WebSocket and log them to CSV.")
    p.add_argument("--backend", choices=("fluigent", "dummy"), default="fluigent",
                   help="sensor source; 'dummy' generates synthetic data for testing (default: %(default)s)")
    p.add_argument("--host", default="0.0.0.0", help="interface to listen on (default: %(default)s)")
    p.add_argument("--port", type=int, default=8765, help="WebSocket port (default: %(default)s)")
    p.add_argument("--channels", type=_parse_channel_list, default=None,
                   help="comma-separated sensor indices to poll, e.g. 0,2,5 (default: all detected)")
    p.add_argument("--poll-hz", type=float, default=1000.0,
                   help="acquisition loop rate; 0 = poll continuously without sleeping (default: %(default)g)")
    p.add_argument("--no-dedupe", action="store_true",
                   help="treat every read as a fresh sample even if the device timestamp did not change")
    p.add_argument("--buffer-seconds", type=float, default=30.0,
                   help="per-channel ring buffer length at --poll-hz; clients read from it with their own "
                        "cursors (default: %(default)g)")
    p.add_argument("--log-dir", default="flow_server_logs",
                   help="directory for CSV logs and the channels sidecar (default: %(default)s)")
    p.add_argument("--rotate-hours", type=float, default=1.0,
                   help="start a new CSV file this often; 0 = never (default: %(default)g)")
    p.add_argument("--no-log", action="store_true", help="disable CSV logging")
    p.add_argument("--stats-interval", type=float, default=10.0,
                   help="seconds between console stats lines; 0 = off (default: %(default)g)")
    g = p.add_argument_group("dummy backend")
    g.add_argument("--dummy-channels", type=int, default=4, help="number of simulated sensors (default: %(default)s)")
    g.add_argument("--dummy-rate", type=float, default=50.0,
                   help="simulated sensor update rate in Hz (default: %(default)g)")
    g.add_argument("--dummy-fail-channels", type=_parse_channel_list, default=None,
                   help="simulate unplugged sensors on these channels, e.g. 3")
    args = p.parse_args(argv)
    if args.poll_hz < 0:
        p.error("--poll-hz must be >= 0")
    if not 0 <= args.port <= 65535:
        p.error("--port must be in 0..65535")
    if args.buffer_seconds <= 0:
        p.error("--buffer-seconds must be > 0")
    if args.rotate_hours < 0:
        p.error("--rotate-hours must be >= 0")
    if args.dummy_rate <= 0:
        p.error("--dummy-rate must be > 0")
    return args


def make_backend(args: argparse.Namespace) -> Any:
    if args.backend == "dummy":
        return DummyBackend(args.dummy_channels, args.dummy_rate, args.dummy_fail_channels)
    return FluigentBackend()


def run(args: argparse.Namespace) -> int:
    backend = make_backend(args)
    log_queue: Optional[queue.Queue] = None if args.no_log else queue.Queue(maxsize=LOG_QUEUE_MAX)
    sampler = Sampler(backend, args.channels, args.poll_hz, not args.no_dedupe, log_queue, args.buffer_seconds)
    LOG.info("Starting %s backend%s", backend.name,
             " (fgt_init may take a few seconds)" if backend.name == "fluigent" else "")
    sampler.start()
    try:
        channels = sampler.wait_ready()
    except BackendError as e:
        LOG.error("%s", e)
        sampler.join(timeout=5)
        return 2
    except KeyboardInterrupt:
        LOG.info("Interrupted during startup")
        sampler.stop()
        return 130

    writer: Optional[CsvLogWriter] = None
    rc = 0
    try:
        LOG.info("Backend: %s | %d channel(s) | poll %s | dedupe %s | ring %d samples/channel",
                 backend.name, len(channels), f"{args.poll_hz:g} Hz" if args.poll_hz > 0 else "unpaced",
                 "off" if args.no_dedupe else "on", sampler.ring_capacity)
        for c in channels:
            LOG.info("  ch %d: %s, serial %s, unit %s", c.ch, c.sensor_type, c.serial or "n/a", c.unit)
        if log_queue is not None:
            stamp = _log_stamp()
            try:
                writer = CsvLogWriter(log_queue, args.log_dir, args.rotate_hours * 3600.0, stamp)
                sidecar = write_sidecar(args.log_dir, stamp, backend.name, channels, args, writer.path)
            except OSError as e:
                LOG.error("Cannot write logs to %s: %s (use --log-dir or --no-log)", os.path.abspath(args.log_dir), e)
                return 1
            writer.start()
            LOG.info("Logging to %s (rotate %s); channel info in %s", writer.path,
                     f"every {args.rotate_hours:g} h" if args.rotate_hours > 0 else "never", sidecar)
        else:
            LOG.info("CSV logging disabled (--no-log)")

        hub = Hub(sampler, backend.name, writer, args.stats_interval)
        sampler.notify = hub.notify_threadsafe
        sampler.go()
        try:
            asyncio.run(serve_clients(args, hub))
            rc = 1 if hub.sampler_failed else 0
        except KeyboardInterrupt:
            LOG.info("Ctrl+C received, shutting down")
        except OSError as e:
            LOG.error("WebSocket server error on %s:%d: %s", args.host, args.port, e)
            rc = 1
    finally:
        sampler.stop()
        sampler.join(timeout=10)
        if sampler.is_alive():
            LOG.error("Sampler thread did not stop within 10 s")
        if writer is not None:
            writer.stop()
            writer.join(timeout=60)
        if sampler.channel_ids:
            per_ch = ", ".join(f"{ch}:{n}" for ch, n in zip(sampler.channel_ids, sampler.emitted))
            LOG.info("Stopped. Samples per channel {%s}, total %d, poll loops %d",
                     per_ch, sum(sampler.emitted), sampler.loops)
        if writer is not None:
            LOG.info("CSV: %d rows written in %d file(s), %d dropped (queue full), %d failed writes; last file %s",
                     writer.rows_written, len(writer.files), sampler.dropped_log, writer.rows_failed, writer.path)
    return rc


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    listener = setup_logging()
    release_timer = raise_timer_resolution()
    # Let the sampler get the GIL back within ~1 ms of waking (default switch interval is 5 ms).
    sys.setswitchinterval(0.001)
    try:
        return run(args)
    finally:
        release_timer()
        listener.stop()


if __name__ == "__main__":
    sys.exit(main())
