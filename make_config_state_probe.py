"""A 4-minute 'where are we' probe.

Answers three questions the duty-cycle runs leave open:

  1. What priming state is the pump in now? -> burst peak, directly comparable
     to the whole history of this device (160-270 since 21 Sep; 621-717 on
     17 Sep, never repeated).

  2. Is it stable right now, or decaying? -> the burst is run before AND after,
     so the ratio says whether state is moving. The duty-cycle config has only
     an opening burst and therefore cannot detect the decay that in fact ran
     through both of its runs (-49% on 5 Oct, -31% on 7 Oct).

  3. How does it compare to the duty-cycle runs' own baseline? -> a 30 s
     continuous 100 Hz block, the same condition those runs used as their
     reference. Theirs started at 75 uL/min (5 Oct) and 110 (7 Oct) and
     decayed from there.

Gate, same as the bracketed protocol: post/pre burst ratio outside [0.5, 2.0]
means state moved too much during the probe for the numbers to mean anything.
"""
import json

zero = lambda s: {"type": "constant", "frequency_hz": 0, "duration_s": float(s),
                  "repeat": 1, "settle_ms": 0}
burst = {"type": "sweep", "start_hz": 1, "end_hz": 1000, "steps": 18,
         "total_duration_s": 1.0, "repeat": 60, "settle_ms": 0}
cont = {"type": "constant", "frequency_hz": 100, "duration_s": 30.0,
        "repeat": 1, "settle_ms": 0}

tests = [zero(30.0), dict(burst), zero(15.0), dict(cont),
         zero(15.0), dict(burst), zero(15.0)]

cfg = {"port": "COM17", "baud": 115200, "read_timeout_s": 0.2,
       "start_time": None, "end_time": None,
       "break_seconds_between_tests": 0.0, "send_zero_on_exit": True,
       "tests": tests,
       "meta": {"name": "State probe",
                "description": ("4 min. Bracketed priming burst either side of a 30 s "
                                "continuous 100 Hz reference. Tells you the current state, "
                                "whether it is moving, and how it compares to the duty-cycle "
                                "runs' own baseline."),
                "operator": "", "pump_number": "20260827-1"},
       "serial": {"protocol": "freq_only"},
       "flow_sensor": {"backend": "fluigent", "channel": 0, "sample_rate_hz": 25.0},
       "protocol": "freq_only"}

json.dump(cfg, open("config_state_probe.json", "w"), indent=2)
total = sum(t["duration_s"] if t["type"] == "constant" else 72.0 for t in tests)
print("config_state_probe.json")
print(f"  {len(tests)} blocks | {total:.0f} s = {total/60:.1f} min")
print("  zero 30 | BURST | zero 15 | 100 Hz 30 s | zero 15 | BURST | zero 15")
print("\n  how to read the result:")
print("    burst 160-270   -> unchanged since 21 Sep; the 5/7 Oct duty data is current")
print("    burst < 50      -> lost prime; nothing is comparable until re-primed")
print("    burst > 400     -> something improved (refill or re-prime); duty data is stale")
print("    post/pre > 1.3  -> still priming up, wait before running a long protocol")
print("    post/pre < 0.7  -> decaying now, as it did through both duty runs")
print("    100 Hz block    -> compare against 75 (5 Oct) and 110 (7 Oct) uL/min")
