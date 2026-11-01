"""TCP <-> serial bridge: run this on the PC the RSD3305P is plugged into, so the dashboard on the rig PC
can drive the supply over the network.

    python supply_bridge.py --port COM5 --allow 10.0.0.23
    python supply_bridge.py --list                       # show COM ports

Then in the dashboard: Supply control = RSD3305P, address = <this PC's name or IP>:5025.

Bytes are forwarded unchanged in both directions (the supply's protocol has no terminators, so nothing
is buffered or reframed). One client at a time: a second connection is refused while one is active, so
two dashboards can never fight over the supply. --allow restricts which IPs may connect (recommended);
Windows Firewall must allow inbound TCP on the port (the first run usually asks).
Needs only pyserial (pip install pyserial).
"""
from __future__ import annotations

import argparse
import socket
import sys
import threading
import time


def log(msg: str):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def serve(ser, listen_host: str, listen_port: int, allow=None, idle_s: float = 0.0):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((listen_host, listen_port))
    srv.listen(2)
    log(f"listening on {listen_host}:{listen_port} -> {ser.port or ser.name} @ {ser.baudrate} baud"
        + (f", allowed: {', '.join(sorted(allow))}" if allow else ", any client (use --allow to restrict)"))
    busy = threading.Lock()

    def handle(conn, peer):
        stop = threading.Event()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.settimeout(0.2)
        last = [time.monotonic()]

        def serial_to_net():
            while not stop.is_set():
                try:
                    data = ser.read(ser.in_waiting or 1)
                except Exception as e:
                    log(f"serial read failed: {e}")
                    stop.set()
                    break
                if data:
                    try:
                        conn.sendall(data)
                    except OSError:
                        stop.set()

        t = threading.Thread(target=serial_to_net, daemon=True)
        t.start()
        try:
            while not stop.is_set():
                try:
                    data = conn.recv(1024)
                except socket.timeout:
                    if idle_s and time.monotonic() - last[0] > idle_s:
                        log(f"{peer[0]} idle for {idle_s:.0f} s, closing")
                        break
                    continue
                except OSError:
                    break
                if not data:
                    break
                last[0] = time.monotonic()
                ser.write(data)
        finally:
            stop.set()
            t.join(timeout=1.0)
            try:
                conn.close()
            except OSError:
                pass
            busy.release()
            log(f"{peer[0]}:{peer[1]} disconnected")

    while True:
        conn, peer = srv.accept()
        if allow and peer[0] not in allow:
            log(f"refused {peer[0]} (not in --allow)")
            conn.close()
            continue
        if not busy.acquire(blocking=False):
            log(f"refused {peer[0]}: another client is connected")
            conn.close()
            continue
        log(f"{peer[0]}:{peer[1]} connected")
        try:
            ser.reset_input_buffer()
        except Exception:
            pass
        threading.Thread(target=handle, args=(conn, peer), daemon=True).start()


def main(argv=None):
    ap = argparse.ArgumentParser(description="TCP <-> serial bridge for the RSD3305P bench supply")
    ap.add_argument("--port", help="serial port of the supply, e.g. COM5 (any pyserial URL works)")
    ap.add_argument("--baud", type=int, default=9600)
    ap.add_argument("--listen", default="0.0.0.0:5025", help="host:port to listen on (default 0.0.0.0:5025)")
    ap.add_argument("--allow", nargs="*", default=None, help="client IPs allowed to connect")
    ap.add_argument("--idle", type=float, default=0.0, help="drop a client silent for this many s (0 = never)")
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    a = ap.parse_args(argv)
    import serial
    from serial.tools import list_ports
    if a.list or not a.port:
        for p in list_ports.comports():
            print(f"{p.device:8s} {p.description}  [{p.hwid}]")
        if not a.port:
            print("\nPick the supply's port and run again with --port COMx")
        return 0
    host, _, port = a.listen.rpartition(":")
    ser = serial.serial_for_url(a.port, baudrate=a.baud, bytesize=8, parity="N", stopbits=1, timeout=0.02)
    try:
        serve(ser, host or "0.0.0.0", int(port), set(a.allow) if a.allow else None, a.idle)
    except KeyboardInterrupt:
        log("stopped")
    finally:
        ser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
