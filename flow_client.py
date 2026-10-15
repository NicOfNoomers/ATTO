#!/usr/bin/env python3
"""
Client for the shared ATTO flow server (flow_server.py), protocol v1.

The flow server owns the Fluigent SDK session and streams every flow sensor
over a WebSocket, so several experiments (dashboards, loggers, scripts) can
each read their own sensor at the same time, on this PC or across the LAN.

Library use
-----------
    from flow_client import FlowClient, parse_server

    host, port = parse_server("192.168.1.20:8765")
    client = FlowClient(host, port, channels=[1], rate="max", mode="latest")
    client.start()
    if not client.wait_connected(3.0):
        raise RuntimeError("flow server not reachable")
    sample = client.latest(1)   # {"ch", "seq", "t", "dev_t", "v", "ok", "rx_mono"} or None
    client.close()

    # Or receive every pushed sample in a callback (runs in the client thread):
    client = FlowClient(channels="all", rate="max", mode="all", on_samples=print)

The client runs in a daemon thread, reconnects with backoff (0.5 s -> 5 s) and
re-subscribes after every reconnect. "rx_mono" is the local time.monotonic()
at which the sample arrived; "t" is the server's epoch time at acquisition.
"localhost" is dialled as 127.0.0.1 because on Windows a refused IPv6 (::1)
attempt costs ~2 s before the IPv4 fallback.

Command line (test client and small standalone logger)
------------------------------------------------------
    python flow_client.py                                    # all channels, every new sample
    python flow_client.py --server 192.168.1.20:8765 --channels 0,3 --rate 5 --mode mean
    python flow_client.py --channels 1 --mode all --csv flow_ch1.csv --duration 600
    python flow_client.py --info                             # server channels and stats

Modes (per client, at its own --rate pushes per second, or "max" = as soon as available):
    latest  newest sample per channel, only if it is new (no repeats)
    all     every raw sample since the last push (a gap in seq means lost data)
    mean    per-channel mean since the last push, with n/min/max
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import logging
import queue
import sys
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

try:
    from websockets.sync.client import connect as _ws_connect
except ImportError as exc:
    raise ImportError("flow_client needs the 'websockets' package: pip install websockets") from exc

LOG = logging.getLogger("flow_client")

PROTOCOL_VERSION = 1
DEFAULT_PORT = 8765
MODES = ("latest", "all", "mean")

Channels = Union[str, Sequence[int]]
Rate = Union[str, float]
Sample = Dict[str, Any]

_CONNECT_KW: Dict[str, Any] = {
    "open_timeout": 3.0,
    "close_timeout": 1.0,
    "ping_interval": 5.0,
    "ping_timeout": 5.0,
    "compression": None,
}
if "proxy" in inspect.signature(_ws_connect).parameters:
    # websockets >= 15 honours HTTP(S)_PROXY by default; lab traffic is LAN-only.
    _CONNECT_KW["proxy"] = None

_RECV_POLL_S = 0.25
_BACKOFF_MIN_S = 0.5
_BACKOFF_MAX_S = 5.0


def parse_server(text: Optional[str], default_port: int = DEFAULT_PORT) -> Tuple[str, int]:
    """Parse 'host:port', 'host', ':port', 'ws://host:port/' or '[ipv6]:port' into (host, port)."""
    s = str(text or "").strip()
    if "://" in s:
        s = s.split("://", 1)[1]
    s = s.split("/", 1)[0]

    if s.startswith("["):
        host, _, rest = s[1:].partition("]")
        port_s = rest[1:] if rest.startswith(":") else ""
    elif s.count(":") == 1:
        host, port_s = s.split(":")
    else:
        host, port_s = s, ""  # plain host name, or a bare IPv6 address

    host = host.strip() or "localhost"
    if port_s.strip():
        try:
            port = int(port_s)
        except ValueError:
            raise ValueError(f"Bad port in flow server address {text!r}") from None
    else:
        port = default_port
    if not 0 < port < 65536:
        raise ValueError(f"Port out of range in flow server address {text!r}")
    return host, port


def _normalize_channels(channels: Channels) -> Union[str, List[int]]:
    if isinstance(channels, str):
        if channels.strip().lower() == "all":
            return "all"
        raise ValueError(f"channels must be 'all' or a list of ints, got {channels!r}")
    return [int(c) for c in channels]


def _normalize_rate(rate: Rate) -> Union[str, float]:
    if isinstance(rate, str):
        if rate.strip().lower() == "max":
            return "max"
        rate = float(rate)
    rate = float(rate)
    if rate <= 0:
        raise ValueError(f"rate must be > 0 or 'max', got {rate!r}")
    return rate


class FlowClient:
    """Background-thread WebSocket client for the shared flow server."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = DEFAULT_PORT,
        channels: Channels = "all",
        rate: Rate = "max",
        mode: str = "latest",
        on_samples: Optional[Callable[[List[Sample]], None]] = None,
        reconnect: bool = True,
    ):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.host = str(host)
        self.port = int(port)
        self.channels = _normalize_channels(channels)
        self.rate = _normalize_rate(rate)
        self.mode = mode
        self.on_samples = on_samples
        self.reconnect = bool(reconnect)

        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ws = None

        # All of the following are guarded by self._cond.
        self._connected = False
        self._worker_done = False
        self._hello: Optional[Dict[str, Any]] = None
        self._subscription: Optional[Dict[str, Any]] = None
        self._subscribe_error: Optional[str] = None
        self._last_error: Optional[str] = None
        self._info_reply: Optional[Dict[str, Any]] = None
        self._latest: Dict[int, Sample] = {}
        self._last_seq: Dict[int, int] = {}
        self._n_samples = 0
        self._n_gaps = 0
        self._n_connects = 0
        self._fail_streak = 0

    # ---- public API ----
    @property
    def uri(self) -> str:
        host = "127.0.0.1" if self.host.lower() == "localhost" else self.host
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"ws://{host}:{self.port}/"

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def connected(self) -> bool:
        """True while the connection is up and the subscription is confirmed."""
        with self._cond:
            return self._connected

    @property
    def channels_info(self) -> Optional[List[Dict[str, Any]]]:
        """Channel list from the server's hello, or None if no hello was received yet."""
        with self._cond:
            if self._hello is None:
                return None
            return [dict(c) for c in self._hello.get("channels", []) if isinstance(c, dict)]

    @property
    def server_info(self) -> Optional[Dict[str, Any]]:
        """The full hello message of the current/last connection."""
        with self._cond:
            return dict(self._hello) if self._hello is not None else None

    @property
    def subscription(self) -> Optional[Dict[str, Any]]:
        """The server's 'subscribed' reply for the current connection."""
        with self._cond:
            return dict(self._subscription) if self._subscription is not None else None

    @property
    def last_error(self) -> Optional[str]:
        with self._cond:
            return self._last_error

    def stats(self) -> Dict[str, Any]:
        """Counters since start(): samples received, seq gaps (mode 'all' only), connections."""
        with self._cond:
            return {
                "connected": self._connected,
                "samples": self._n_samples,
                "seq_gaps": self._n_gaps,
                "connects": self._n_connects,
                "last_error": self._last_error,
            }

    def start(self) -> "FlowClient":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(target=self._run, name=f"FlowClient-{self.address}", daemon=True)
        self._thread.start()
        return self

    def wait_connected(self, timeout: Optional[float] = None) -> bool:
        """Block until hello + subscribed were received. Returns early (False) if the
        server rejected the subscription or the client stopped."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while not self._connected and self._subscribe_error is None and not self._worker_done:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    break
                self._cond.wait(remaining)
            return self._connected

    def latest(self, ch: int) -> Optional[Sample]:
        """Last sample received on this connection for channel `ch` (copy), or None."""
        with self._cond:
            s = self._latest.get(int(ch))
            return dict(s) if s is not None else None

    def info(self, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
        """Ask the server for {'channels', 'stats'}; None if not connected or no reply in time."""
        ws = self._ws
        if ws is None or not self.connected:
            return None
        with self._cond:
            self._info_reply = None
        try:
            ws.send(json.dumps({"cmd": "info"}))
        except Exception as exc:
            LOG.debug("info request failed: %s", exc)
            return None
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._info_reply is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return self._info_reply

    def close(self, timeout: float = 1.0) -> None:
        """Stop the background thread, waiting at most `timeout` for it. A connect attempt
        still in flight is not waited for; the daemon thread exits when it finishes."""
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout)

    def __enter__(self) -> "FlowClient":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ---- background thread ----
    def _subscribe_msg(self) -> Dict[str, Any]:
        return {"cmd": "subscribe", "channels": self.channels, "rate": self.rate, "mode": self.mode}

    def _run(self) -> None:
        backoff = _BACKOFF_MIN_S
        try:
            while not self._stop.is_set():
                connects_before = self._n_connects
                error: Optional[str] = None
                try:
                    self._session()
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
                self._on_disconnect(error)
                if self._stop.is_set() or not self.reconnect:
                    break
                if self._n_connects > connects_before:
                    backoff = _BACKOFF_MIN_S
                self._stop.wait(backoff)
                backoff = min(backoff * 2, _BACKOFF_MAX_S)
        finally:
            with self._cond:
                self._connected = False
                self._worker_done = True
                self._cond.notify_all()

    def _session(self) -> None:
        with _ws_connect(self.uri, **_CONNECT_KW) as ws:
            self._ws = ws
            try:
                if self._stop.is_set():
                    return
                hello = json.loads(ws.recv(timeout=5.0))
                if not isinstance(hello, dict) or hello.get("type") != "hello":
                    raise ConnectionError(f"expected hello from server, got {str(hello)[:200]}")
                if hello.get("protocol") != PROTOCOL_VERSION:
                    LOG.warning("Flow server %s speaks protocol %r, client expects %d",
                                self.address, hello.get("protocol"), PROTOCOL_VERSION)
                with self._cond:
                    self._hello = hello
                    self._subscription = None
                    self._subscribe_error = None
                    self._latest.clear()
                    self._last_seq.clear()
                ws.send(json.dumps(self._subscribe_msg()))

                while not self._stop.is_set():
                    try:
                        raw = ws.recv(timeout=_RECV_POLL_S)
                    except TimeoutError:
                        continue
                    try:
                        msg = json.loads(raw)
                    except ValueError:
                        LOG.warning("Flow server %s sent a non-JSON frame; ignored", self.address)
                        continue
                    if isinstance(msg, dict):
                        self._dispatch(msg)
            finally:
                self._ws = None

    def _dispatch(self, msg: Dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind == "data":
            self._handle_samples(msg.get("samples"))
        elif kind == "subscribed":
            with self._cond:
                was_connected = self._connected
                self._connected = True
                self._subscription = msg
                self._subscribe_error = None
                if not was_connected:
                    self._n_connects += 1
                    self._fail_streak = 0
                n = self._n_connects
                self._cond.notify_all()
            if not was_connected:
                LOG.info("%s to flow server %s: channels=%s rate=%s mode=%s",
                         "Connected" if n == 1 else "Reconnected", self.address,
                         msg.get("channels"), msg.get("rate"), msg.get("mode"))
        elif kind == "error":
            text = str(msg.get("msg", "unknown error"))
            with self._cond:
                self._last_error = f"server: {text}"
                if not self._connected:
                    self._subscribe_error = text
                self._cond.notify_all()
            LOG.warning("Flow server %s error: %s", self.address, text)
        elif kind == "info":
            with self._cond:
                self._info_reply = msg
                self._cond.notify_all()
        # hello (again), pong, unsubscribed and unknown types need no action.

    def _handle_samples(self, samples: Any) -> None:
        if not isinstance(samples, list):
            return
        rx = time.monotonic()
        good: List[Sample] = []
        with self._cond:
            for s in samples:
                if not isinstance(s, dict) or not isinstance(s.get("ch"), int):
                    continue
                s["rx_mono"] = rx
                ch = s["ch"]
                seq = s.get("seq")
                if isinstance(seq, int):
                    prev = self._last_seq.get(ch)
                    if prev is not None and self.mode == "all" and seq > prev + 1:
                        self._n_gaps += seq - prev - 1
                    if prev is None or seq > prev:
                        self._last_seq[ch] = seq
                    cur = self._latest.get(ch)
                    if cur is None or not isinstance(cur.get("seq"), int) or seq >= cur["seq"]:
                        self._latest[ch] = s
                else:
                    self._latest[ch] = s
                good.append(s)
            self._n_samples += len(good)
        if good and self.on_samples is not None:
            try:
                self.on_samples(good)
            except Exception:
                LOG.exception("FlowClient on_samples callback failed")

    def _on_disconnect(self, error: Optional[str]) -> None:
        with self._cond:
            was_connected = self._connected
            self._connected = False
            self._subscription = None
            if error:
                self._last_error = error
                self._fail_streak += 1
            streak = self._fail_streak
            self._cond.notify_all()
        if self._stop.is_set():
            return
        if was_connected:
            LOG.warning("Lost connection to flow server %s (%s); reconnecting",
                        self.address, error or "closed by server")
        elif error and streak == 1:
            LOG.warning("Cannot connect to flow server %s (%s); retrying", self.address, error)
        elif error:
            LOG.debug("Flow server %s still unreachable (%s)", self.address, error)


# ---------------- command line ----------------
def _parse_channels_arg(text: str) -> Union[str, List[int]]:
    t = text.strip().lower()
    if t in ("", "all", "*"):
        return "all"
    try:
        return [int(x) for x in t.replace(" ", "").split(",") if x]
    except ValueError:
        raise argparse.ArgumentTypeError(f"channels must be 'all' or e.g. 0,3 (got {text!r})") from None


def _parse_rate_arg(text: str) -> Union[str, float]:
    try:
        return _normalize_rate(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"rate must be a positive number or 'max' (got {text!r})") from None


def _fmt_time(epoch: Any) -> str:
    if not isinstance(epoch, (int, float)):
        return "-"
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _fmt_num(x: Any, fmt: str = "{:.5g}") -> str:
    return fmt.format(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else "None"


def _print_hello(client: FlowClient) -> None:
    hello = client.server_info or {}
    print(f"Connected to {client.uri} (server={hello.get('server')}, backend={hello.get('backend')}, "
          f"protocol={hello.get('protocol')})", file=sys.stderr)
    print(f"  {'ch':>3}  {'sensor_type':<20} {'serial':<16} unit", file=sys.stderr)
    for c in client.channels_info or []:
        print(f"  {c.get('ch', '?'):>3}  {str(c.get('sensor_type')):<20} {str(c.get('serial')):<16} {c.get('unit')}",
              file=sys.stderr)
    print(f"  subscribed: {client.subscription}", file=sys.stderr)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Test client and small logger for the shared flow server (flow_server.py).",
        epilog="examples:\n"
               "  python flow_client.py --channels 0 --rate 2 --mode mean\n"
               "  python flow_client.py --server 192.168.1.20:8765 --channels 1,2 --mode all --csv out.csv --duration 60",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--server", metavar="HOST:PORT", help="server address (overrides --host/--port)")
    ap.add_argument("--host", default="localhost", help="server host (default: localhost)")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"server port (default: {DEFAULT_PORT})")
    ap.add_argument("--channels", type=_parse_channels_arg, default="all", help="e.g. 0,3 or all (default: all)")
    ap.add_argument("--rate", type=_parse_rate_arg, default="max", help="pushes per second, or max (default: max)")
    ap.add_argument("--mode", choices=MODES, default="latest", help="latest | all | mean (default: latest)")
    ap.add_argument("--csv", metavar="PATH", help="write samples to CSV instead of printing them")
    ap.add_argument("--duration", type=float, metavar="S", help="stop after S seconds (default: run until Ctrl+C)")
    ap.add_argument("--info", action="store_true", help="print the server's channel list and stats, then exit")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # units like "µl/min" on a cp1252 pipe
        except Exception:
            pass
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(message)s")
    if not args.verbose:
        logging.getLogger("websockets").setLevel(logging.WARNING)

    host, port = parse_server(args.server) if args.server else (args.host, args.port)
    batches: "queue.Queue[List[Sample]]" = queue.Queue()
    client = FlowClient(host, port, channels=args.channels, rate=args.rate, mode=args.mode,
                        on_samples=None if args.info else batches.put)

    t_start = time.monotonic()
    deadline = None if args.duration is None else t_start + args.duration
    counts: Dict[int, int] = {}
    csv_file = None
    writer = None
    printed_connects = 0
    try:
        client.start()
        print(f"Connecting to {client.uri} ...", file=sys.stderr)
        if args.info:
            if not client.wait_connected(5.0 if deadline is None else max(0.1, deadline - time.monotonic())):
                print(f"Flow server at {host}:{port} not reachable ({client.last_error}).", file=sys.stderr)
                return 1
            _print_hello(client)
            print(json.dumps(client.info() or {"error": "no info reply"}, indent=2))
            return 0

        if args.csv:
            csv_file = open(args.csv, "w", newline="", encoding="utf-8")
            writer = csv.writer(csv_file)
            writer.writerow(["server_time", "server_iso", "ch", "seq", "dev_t", "value", "ok", "n", "min", "max"])
        units = {}
        last_flush = last_report = time.monotonic()

        while deadline is None or time.monotonic() < deadline:
            try:
                batch = batches.get(timeout=0.2)
            except queue.Empty:
                batch = []
            n_connects = client.stats()["connects"]
            if n_connects > printed_connects:
                printed_connects = n_connects
                _print_hello(client)
                units = {c.get("ch"): c.get("unit", "") for c in client.channels_info or []}
            for s in batch:
                ch = s["ch"]
                counts[ch] = counts.get(ch, 0) + 1
                if writer is not None:
                    writer.writerow([s.get("t"), _fmt_time(s.get("t")), ch, s.get("seq"), s.get("dev_t"),
                                     s.get("v"), int(bool(s.get("ok"))), s.get("n", ""), s.get("min", ""),
                                     s.get("max", "")])
                else:
                    line = (f"{_fmt_time(s.get('t'))}  ch={ch:<2} seq={s.get('seq')!s:<8} "
                            f"v={_fmt_num(s.get('v'), '{:>10.5g}')} {units.get(ch, '')}  ok={bool(s.get('ok'))}")
                    if "n" in s:
                        line += f"  n={s.get('n')} min={_fmt_num(s.get('min'))} max={_fmt_num(s.get('max'))}"
                    print(line, flush=False)
            now = time.monotonic()
            if writer is not None and now - last_report >= 2.0:
                last_report = now
                per_ch = ", ".join(f"ch{c}: {n}" for c, n in sorted(counts.items()))
                print(f"[{now - t_start:6.1f} s] {sum(counts.values())} samples ({per_ch})", file=sys.stderr)
            if now - last_flush >= 1.0:
                last_flush = now
                (csv_file or sys.stdout).flush()
    except KeyboardInterrupt:
        pass
    finally:
        client.close()
        if csv_file is not None:
            csv_file.close()

    elapsed = time.monotonic() - t_start
    st = client.stats()
    print(f"\nSummary after {elapsed:.1f} s: {sum(counts.values())} samples, connects={st['connects']}, "
          f"seq_gaps={st['seq_gaps']} (counted in mode 'all' only), last_error={st['last_error']}", file=sys.stderr)
    for ch, n in sorted(counts.items()):
        print(f"  ch{ch}: {n} samples, {n / elapsed if elapsed > 0 else 0:.2f} /s", file=sys.stderr)
    if args.csv:
        print(f"CSV written to {args.csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
