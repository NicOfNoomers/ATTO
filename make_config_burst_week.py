"""Multi-day burst (long-term delivery) test, any pump (NM via atto_test_configs, 5 Nov 2026).

Bursts of --f Hz for --on-s every --period-s, for --days, with a check panel (400 / 600 / 1300 Hz, 10 s each,
10 s zeros around) at the start, every --check-every-h hours and at the end. Built like make_config_long_hold_100hz.py:
one group block per check interval (repeat = bursts in that interval), so a 5-day config stays small.
RSD3305P COM7 at --v (2.5 V), PPK2 on (trip 57 mA / 300 ms) with --bin-ms bins (100 ms default: 10 ms bins are
~27 MB/h uncompressed), raw snapshots hourly, no video (continuous video is ~6 GB/h). Outlet vented, Pump ID required.

The burst pattern is Nic's call: --on-s and --period-s have no defaults. The script prints the reservoir volume.

usage: python make_config_burst_week.py --pump 261030-1 --on-s 10 --period-s 600 [--f 600] [--v 2500] [--days 5]
                                        [--check-every-h 6] [--q-est 200] [--bin-ms 100] [--address COM7]
    -> config_burst_week_<f>Hz_<on>s_per_<period>s.json
Runner (no video; --max-h must cover the run, the default 12 h would stop it):
    py -3.10 scripts\\run_headless.py <config> --name burst_week --pump-id <id> --max-h 130 --show
"""
import argparse
import json
import math

ap = argparse.ArgumentParser(description="Multi-day burst test with periodic check panels")
ap.add_argument("--pump", required=True, help="pump ID on the rig (ask Nic)")
ap.add_argument("--on-s", type=float, required=True, help="burst length (s)")
ap.add_argument("--period-s", type=float, required=True, help="burst period, start to start (s)")
ap.add_argument("--f", type=int, default=600, help="burst frequency (Hz)")
ap.add_argument("--v", type=int, default=2500, help="supply voltage in mV (0 < v <= 2500)")
ap.add_argument("--days", type=float, default=5.0)
ap.add_argument("--check-every-h", type=float, default=6.0, help="check panel interval in hours (0 = only start/end)")
ap.add_argument("--checks", default="400,600,1300", help="check panel frequencies (Hz)")
ap.add_argument("--q-est", type=float, default=200.0, help="flow at --f for the volume estimate (uL/min)")
ap.add_argument("--bin-ms", type=float, default=100.0, help="PPK2 bin width (ms)")
ap.add_argument("--address", default="COM7")
ap.add_argument("--auto-prime", action="store_true",
                help="if the check at --monitor-hz falls more than --drop-pct below the first check, prime and go on")
ap.add_argument("--monitor-hz", type=int, default=None, help="check frequency to watch (default: the burst --f)")
ap.add_argument("--drop-pct", type=float, default=15.0, help="priming trigger: drop below the first check, %%")
ap.add_argument("--prime-f", type=int, default=1300, help="priming frequency (the pump's peak band)")
ap.add_argument("--prime-rounds", type=int, default=3, help="prime_at rounds per priming (70 s each)")
ap.add_argument("--max-primes", type=int, default=3)
args = ap.parse_args()

if not 0 < args.v <= 2500:
    raise SystemExit("--v must be between 1 and 2500 mV (the pump's limit)")
if not 0 < args.on_s < args.period_s:
    raise SystemExit("need 0 < --on-s < --period-s")
if not 1 <= args.f <= 3000:
    raise SystemExit("--f must be within 1..3000 Hz")
CHECKS = [int(x) for x in args.checks.split(",") if x.strip()]
CHECK_S, CHECK_ZERO_S = 10.0, 10.0
MONITOR = args.monitor_hz or args.f
if args.auto_prime and MONITOR not in CHECKS:
    CHECKS = [MONITOR] + CHECKS               # the watched frequency must be in every check panel


def zero(s, **m):
    return {"type": "constant", "frequency_hz": 0, "duration_s": float(s), "repeat": 1, "settle_ms": 0, **m}


def drive(hz, s, **m):
    return {"type": "constant", "frequency_hz": int(hz), "duration_s": float(s), "repeat": 1, "settle_ms": 0, **m}


def panel(name):
    out = [zero(CHECK_ZERO_S)]
    for f in CHECKS:
        out += [drive(f, CHECK_S, section="check", panel=name, level_mv=args.v), zero(CHECK_ZERO_S)]
    return out


def bursts(n, k):
    return {"type": "group", "repeat": int(n), "section": "bursts", "interval": k,
            "blocks": [drive(args.f, args.on_s, section="burst", level_mv=args.v),
                       zero(round(args.period_s - args.on_s, 3))]}


total_bursts = int(round(args.days * 86400 / args.period_s))
if args.check_every_h > 0:
    per = max(1, int(round(args.check_every_h * 3600 / args.period_s)))
    chunks = [per] * (total_bursts // per) + ([total_bursts % per] if total_bursts % per else [])
else:
    chunks = [total_bursts]

supply = {"mode": "rsd3305p", "channel": 1, "current_limit_ma": 60, "max_mv": 2500, "initial_mv": args.v,
          "tolerance_mv": 50, "step_timeout_s": 10, "off_on_exit": True, "eol": "LF"}
if args.address:
    supply["address"] = args.address
tests = [zero(60.0), {"type": "voltage", "voltage_mv": args.v, "while_waiting": "zero", "settle_s": 5,
                      "note": f"{args.v / 1000:.2f} V by the supply"}]
tests += panel("C0")
for k, n in enumerate(chunks, start=1):
    tests.append(bursts(n, k))
    tests += panel(f"C{k}")
tests.append(zero(60.0))

name = f"config_burst_week_{args.f}Hz_{args.on_s:g}s_per_{args.period_s:g}s.json"
cfg = {
    "port": "COM3", "baud": 115200, "read_timeout_s": 0.2,
    "start_time": None, "end_time": None,
    "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
    "require_pump_id": True,
    "tests": tests,
    "meta": {"name": f"Burst week {args.f} Hz {args.on_s:g} s / {args.period_s:g} s", "operator": "",
             "pump_id": args.pump, "burst_hz": args.f, "on_s": args.on_s, "period_s": args.period_s,
             "days": args.days, "check_every_h": args.check_every_h, "check_hz": CHECKS,
             "description": (f"BURST WEEK: {args.f} Hz for {args.on_s:g} s every {args.period_s:g} s for "
                             f"{args.days:g} days at {args.v / 1000:.2f} V (RSD3305P), PPK2 on ({args.bin_ms:g} ms bins), "
                             f"check panel {', '.join(map(str, CHECKS))} Hz at the start, every {args.check_every_h:g} h "
                             f"and at the end. Outlet vented; refill / check the reservoir; no video.")},
    "serial": {"protocol": "freq_only"},
    "protocol": "freq_only",
    "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
    "supply": supply,
    "power": {"enabled": True, "backend": "ppk2", "port": "auto", "bin_ms": args.bin_ms, "watch_ma": 57,
              "watch_hold_ms": 300, "raw_every_min": 60, "raw_length_s": 2.0, "post_run_tail_s": 10.0},
}
if args.auto_prime:
    pf = args.prime_f
    rnd = [drive(pf, 30.0, section="auto_prime"), zero(5.0), drive(round(0.8 * pf), 10.0, section="auto_prime"),
           zero(5.0), drive(round(1.2 * pf), 10.0, section="auto_prime"), zero(10.0)]
    cfg["auto_prime"] = {"monitor_hz": MONITOR, "monitor_section": "check", "drop_pct": args.drop_pct,
                         "reference": "first", "max_primes": args.max_primes,
                         "prime": [b for _ in range(args.prime_rounds) for b in rnd] + [zero(20.0)]}
    cfg["meta"]["auto_prime"] = (f"if the {MONITOR} Hz check falls more than {args.drop_pct:g} % below the first "
                                 f"check: {args.prime_rounds} x prime_at {pf} Hz rounds, then the run continues "
                                 f"(at most {args.max_primes} times); see auto_prime.csv")
with open(name, "w", encoding="utf-8") as fh:
    json.dump(cfg, fh, indent=2)

dur = 120 + sum(c * args.period_s for c in chunks) + (len(chunks) + 1) * (CHECK_ZERO_S + len(CHECKS) * (CHECK_S + CHECK_ZERO_S))
burst_ul = args.q_est * args.on_s / 60.0
per_day = 86400 / args.period_s
vol_bursts = total_bursts * burst_ul / 1000.0
vol_checks = (len(chunks) + 1) * len(CHECKS) * args.q_est * CHECK_S / 60.0 / 1000.0
bins_mb_h = 27.0 * 10.0 / args.bin_ms
print(f"{name}: {len(tests)} blocks ({len(chunks)} burst groups), {dur / 3600:.1f} h = {dur / 86400:.2f} days")
print(f"  {per_day:.0f} bursts/day, {total_bursts} in total; {burst_ul:.2f} uL per burst at ~{args.q_est:g} uL/min "
      f"= {burst_ul * 3600 / args.period_s:.1f} uL/h (steady-state flow; short bursts deliver less, see the start transient)")
print(f"  liquid: bursts {vol_bursts:.1f} mL ({vol_bursts / args.days:.1f} mL/day) + checks {vol_checks:.1f} mL "
      f"= {vol_bursts + vol_checks:.1f} mL over {args.days:g} days (upper bound at {args.q_est:g} uL/min)")
print(f"  PPK2 bins ~{bins_mb_h:.0f} MB/h uncompressed (~{bins_mb_h * 0.21 * dur / 3600 / 1000:.1f} GB gzipped over the run)")
