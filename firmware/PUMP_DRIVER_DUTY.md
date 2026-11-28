# Duty-cycle (carrier-PWM) pump-driver firmware: flashing and checks

Why: capsule-driver note 12 (`PhD/PCB/atto_capsule_driver/notes/12_lab_driver_duty_cycle_brief_2026-11-28.md`). The
capsule sets the coil amplitude with a PWM carrier inside each half-cycle: slow decay, the coil sees ±D × VM.

| Sketch | Board | Notes |
|---|---|---|
| `pump_driver_duty_samd21/` (UF2 in `build/`) | XIAO SAMD21: the rig, USB 2886:802F | **Needs a wire from D9 to the D0 / IN1 net first.** On the SAMD21, D0 = PA02 has no timer output. The firmware keeps D0 as an input. Self-test command `T`. |
| `pump_driver_duty_nrf52/` (UF2 in `build/`) | XIAO nRF52840 | The board's footprint. No wire needed. No self-test. |

Commands, the same on both:
- `<hz>` or `F <hz>`: the frequency (0 = stop).
- `D`: ignored (it is the dashboard's legacy on/off duty).
- `A <0..1>`: carrier duty.
- `C <hz>`: carrier frequency.
- `R <s>`: ramp.
- `P <0|1>`: polarity flip.
- `?`: status.

With no `A` command the drive equals today's plain ±VM square wave. In the dashboard, use protocol `freq_amp` with the
block keys `carrier_duty` / `carrier_hz`. The runner refuses those keys under any other protocol.

## Procedure (each step needs NM's OK; the rig must be idle)
1. **Back up the old firmware. It is the only copy:**
   - double-tap the driver XIAO's reset (RST pads), and a USB drive appears;
   - copy `CURRENT.UF2` to `firmware/driver_backup/CURRENT_<date>.UF2`;
   - the family ID at bytes 28-31 of the first 512-byte block tells the chip: `0x68ed2b88` = SAMD21, `0xada52840` = nRF52840.
2. **SAMD21 only:** solder the wire D9 → D0 / IN1 net. Do this before flashing.
3. **Flash:** double-tap again, then copy the matching `build/*.uf2` to the drive. The board reboots.
4. **Check** with the dashboard closed: `py -3.10 scripts\driver_check.py --port COM3`. It must print `DRIVER CHECK PASSED`.
5. **Undo, if needed:** double-tap and copy the backed-up `CURRENT_<date>.UF2` back.

Then: `make_config_duty_tests.py --test A|B|C` (test A's reference arm needs the RSD3305P), and analyse with
`scripts/duty_blocks.py <run>`.
