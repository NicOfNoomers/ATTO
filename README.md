# ATTO
ATTO TEAM repo

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- USB camera (optional, for video recording)
- OpenCV Python package (optional, installed via requirements.txt)

### Install Dependencies

#### Option 1: Automated Installation Script

**Windows:**
```cmd
install.bat
install.bat C:\Python313\python.exe   (to pick a specific Python)
```
`install.bat` picks Python **3.10** first (the test PC has to run 3.10), then a working `.venv`, `py -3`, `python`, `python3` (see `find_python.bat`). It installs `requirements.txt` into that Python, installs the Fluigent SDK from `fluigent_sdk-23.0.0/` if it isn't installed yet, and runs `scripts\check_setup.py`, which lists missing packages and the serial ports (PPK2 and Arduino marked). Start the dashboard with `run_dashboard.bat`, which uses the same Python.

Packages are per Python: if you start the dashboard some other way (e.g. `python3.10 pump_test_dashboard.py`) and it reports a missing package such as `ppk2_api`, click **Install missing packages** in the Power meter panel. It installs into exactly the Python the dashboard is running on.

**Linux/Mac:**
```bash
chmod +x install.sh
./install.sh
```

The Linux/Mac script will:
1. Check for Python installation
2. Detect or create a virtual environment (`.venv/`)
3. Prompt to use existing virtual environment if found
4. Install all dependencies from requirements.txt
5. Install the Fluigent SDK if available

#### Option 2: Manual Installation

```bash
# Create virtual environment (recommended)
python3 -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Fluigent SDK (if available)
pip install ./fluigent_sdk-23.0.0
```

### Running the Dashboard

```cmd
python pump_test_dashboard.py
```

For help with command-line options:
```cmd
python pump_test_dashboard.py --help
```

### Fluigent SDK

The Fluigent SDK (version 23.0.0) is included in the repository under `fluigent_sdk-23.0.0/`. It will be automatically installed by the install script if the directory exists.

### Virtual Environment

Using a virtual environment is recommended to avoid conflicts with other Python projects. The install scripts will:
- Create a `.venv` virtual environment if none exists
- Prompt to use an existing virtual environment if found at `.venv/` or `venv/`

## Power Meter (Nordic PPK2) and Voltage Steps

The dashboard can log the pump's supply current, power and energy with a Nordic Power Profiler Kit II, used as an **ammeter**: the bench supply sets the voltage and its current limit protects the pump; the PPK2 only measures.

### Wiring
- bench supply + -> PPK2 **VIN**, PPK2 **VOUT** -> pump driver **VM**, all grounds connected
- PPK2 **USB DATA/POWER** port -> the test PC, power switch on
- bench supply current limit set (60 mA for the pump), voltage at most 2.5 V

### First time on a PC
1. `install.bat` (installs `ppk2-api`; the setup check should show the PPK2 as `<- PPK2`, USB ID 1915:c00a)
2. With the **bench supply off**, check that the PPK2 doesn't put a voltage on the line in ammeter mode:
   `python scripts\ppk2_probe.py --idle-check` and measure VOUT-GND with a multimeter (must be ~0 V)
3. Optional bench test (current per frequency, decoder check, lost samples):
   `python scripts\ppk2_probe.py --supply-mv 2500 --arduino COM17`

Close the dashboard while the probe runs (only one program can use the PPK2), and close the nRF Connect Power Profiler app.

### Use
In the **Power meter (PPK2)** panel tick *Measure power* (port `auto` or *Detect*), set the supply voltage at start and the bench supply's current limit. The PPK2 then starts and stops with every scheduled run or manual session. The live line shows current, power, voltage and energy; the supply current is drawn in red on the frequency plot. If the current stays at or above the trip level (default 95 % of the bench supply limit) for 200 ms, the supply is clamping: the run stops, the pump goes to 0 Hz and the PPK2's output is switched off.

**Voltage steps** (*Add Voltage Step* in the config builder) pause the run until you have set the bench supply and confirmed the voltage in the yellow banner. Before the run starts the dashboard lists when each change is due (also saved as `schedule.txt`; *Voltage schedule...* shows the live list) so you can set alarms.

### Files (in each run folder)
- `power/bins_*.csv(.gz)`: every 10 ms: mean/min/max current (mA), power (mW), supply voltage, cumulative charge (mC) and energy (mJ), logic inputs, lost samples. A new file every hour; closed files are gzipped
- `power/power_1s.csv` (1 s summary), `power/raw/` (2 s of raw 100 kS/s data every 10 min, and around a trip), `power/power_events.csv`, `power/anchors.csv`, `power/summary.json`
- `voltage_changes.csv` (requested/confirmed voltage, how long the confirmation took), `schedule.txt`, `status.json` (progress heartbeat every 10 s)

## Pressure Control and PQ Curves (stepper syringe)

A 1 mL liquid-filled syringe on the pump **outlet**, driven by a stepper on an Arduino Uno, holds the outlet pressure while the flow is measured. The pressure sensor (a Fluigent channel) sits between the outlet and the syringe. Code: `pressure_control.py`; firmware: the old `syringe_simple_controls_sketch` (v1) works, `firmware/syringe_v2/syringe_v2.ino` is better (stops mid-move, velocity mode, holds torque, keeps its position over the Uno's reset).

### Set up once
1. **Pressure control (syringe)** panel: the syringe Uno's port, *Find* to pick the Fluigent pressure channel (the one in mbar), **Microsteps** = the driver's setting (A4988 with MS1-MS3 high = 16). Ranges in the config stay in full steps (syringe 0..2470); the software multiplies them. The flow and pressure sensors share one Fluigent session.
2. Tick *Control pressure*, *Connect Manual* (pump output off), zero the sensor with the line vented, empty the syringe and press *Syringe emptied: pos 0*. That is enough: holds measure the mbar per step themselves with small probe moves.
3. Optional, *Calibrate...*: give the syringe room to push (Output ON ~400 Hz, Hold 0 mbar for 20-30 s, Release, Output OFF, Stop), then the gain + backlash sweep (pump off, line liquid-filled; it takes up the slack 1-4 steps at a time, pushes up from the start pressure and back, never below it, because sucking on the outlet draws liquid through the stopped pump; clamp the inlet of a valveless pump), optionally the leak test and step response. *Save calibration* keeps it in `%LOCALAPPDATA%\ATTO\pressure_cal.json` (rescaled if Microsteps changes).

Lab, 29 Oct: at full steps one step moved the liquid-filled line 10-25 mbar (the 345 mbar sensor range was only ~15-30 steps) with ~20 steps of backlash; a hold then saw-tooths +-half a step. Microstepping (x16: ~1 mbar per count) or a small air pocket in the line fixes that; the panel warns when the steps are coarse.

The controller learns while it runs (mbar per step for pushing and pulling, backlash, the pressure drift from the pump flow), starting from the calibration (one saved before microsteps were recorded is ignored except for the zero). Jogs are capped at 5 full steps until the gain is known, then by the predicted pressure. Limits: *Min/Max* in the panel; the max is also capped at 95 % of the sensor range. Above them, on a saturated or silent sensor, a stuck syringe, or a full syringe, the controller faults: the syringe stops, overpressure is let down, the pump goes to 0 Hz and the run stops. With v2 firmware the syringe runs in velocity mode (continuous motion); with v1 it makes stop-start moves (`"control"`: `auto` / `steps` / `velocity`). While the pump runs the syringe only **pulls**: the pump raises the pressure, and every direction change costs the plunger/coupling slack (~350 counts at x16 in the lab, 29 Oct). A pressure the pump can't reach then plateaus below the target and the PQ runner logs it as `above_shutoff`. Holding below 0 mbar draws liquid in through the pump and degasses the line (Min defaults to -30 mbar).

### Run
`python make_config_pq.py` writes `config_pq_smoke.json` and `config_pq.json`. A **PQ block** (*Add PQ Curve*) holds each pressure in turn at each frequency, waits until the 1 s mean is within 2 mbar for 3 s, then averages flow and pressure over `measure_s`. A **Pressure block** holds one pressure (or releases). The pump flow fills the syringe: before a point that would not fit, the run sets the pump to 0 Hz, lets the pressure down and shows an orange banner. Empty the syringe **by hand** to its empty end, reconnect, close the line and press *Done - continue*. The position becomes 0 and the point is repeated. A pressure the pump cannot reach is logged as `above_shutoff` (x on the plot) and the higher pressures at that frequency are skipped.

### Files (in each run folder)
- `pq_points.csv`: one row per point: target, mean/SD pressure and flow, syringe steps/s (and µL/min from `ul_per_step`), mean supply current (PPK2), settled flag, settling time, syringe clears, note. `pq_curve.png` at the end; the *PQ curve* tab can reopen any `pq_points.csv`
- `pressure/pressure.csv` (25 Hz: pressure, target, syringe position, mode, learned gains and drift), `pressure/events.json` (targets, clears, faults, learned values), `pressure/calibration_used.json`

## USB Camera Video Recording

The dashboard supports recording video from a USB camera during tests. Video recording is **disabled by default** and must be enabled in the UI.

### Setup
1. Connect your USB camera before starting the dashboard
2. Note the camera index (default is 0 for the first camera; `dummy` gives a synthetic test camera)
3. Click "Connect Camera" in the "USB Camera (optional)" section
4. Once connected, enable video recording with the checkbox

### Usage
- During a test, video will automatically record if enabled and camera is connected
- A capture thread reads the camera as fast as it delivers frames; a separate writer thread encodes them, so encoding never slows capture. The status line shows the camera's real frame rate, frames written and frames dropped
- **Overlay** (time, frequency, flow in the top-left corner) is a checkbox; turn it off when the image itself is measured
- **Codec:** `mjpg` (default, quality 95), `ffv1` (lossless, much larger) or `mp4v` (the old default)
- Files in the run folder:
  - `video_partNNNN.avi`: a new file every 60 s (`video.chunk_s` in the config), with the measured frame rate in the header
  - `frames.csv`: one row per written frame: `frame_index`, `t_s` (same clock as events.csv), `t_perf_s`, `cam_pos_msec`, `dropped_before`, `video_file`, `video_frame` (frame number inside that file)
  - `video_summary.json`: frames written and dropped, header and measured fps, largest frame gap
- Optional in the config: `"video": {"crop": [x, y, w, h]}` records only that strip of the image
- Recording automatically stops when the test ends

### Run timing
All run times (events.csv, merged.csv, power, frames.csv) use `time.perf_counter`. On Python 3.10 `time.monotonic` only ticks every 15.6 ms on Windows and a 1 ms sleep takes ~15 ms, so steps used to run up to ~16 ms off; during a run the dashboard now asks Windows for a 1 ms timer and waits the last few ms precisely. `meta.json` records both clocks (`clock`). During runs the GIL switch interval is 0.5 ms (the plot redraw holds the GIL for 50-70 ms and used to delay commands by up to ~15-40 ms), and the process opts out of Windows 11's timer throttling (otherwise the 1 ms timer is ignored while the dashboard window is covered or minimized). Scheduled commands are typically within 0.5 ms of nominal with the window, plots and 1080p video running; a few times a minute a long call on the UI thread or the OS still delays a wake-up by tens of ms, so use `t_cmd_s` for exact command times. `events.csv` has `t_cmd_s` / `t_cmd_done_s`: the run clock just before and after each frequency command's serial write.

## Stroke-volume run (1 Hz voltage ladder, microscope video)

`make_config_stroke_volume.py` writes `config_stroke_volume.json`: at 1 Hz, bench-supply voltage steps 2.5, 2.0, 1.5, 1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1 V; each step is: confirm the voltage (yellow banner) → 20 s rest at 0 Hz → 19.75 s at 1 Hz (20 strokes; the drive stops a quarter period early so the stop never races the edge that would start stroke 21). About 7 min plus the confirmations.
- Wiring: bench supply straight to the driver VM, **PPK2 out of the path** (its ammeter mode needs 0.8-5.0 V); power measurement is off in this config. Bench supply current limit 60 mA.
- The config requires a **Pump ID** (Run details) and sets the overlay off and MJPG quality 95.
- Volume accounting: the driver bridge sleeps at 0 Hz, so each drive goes neutral → +x (half swing) → 39 full swings → −x → neutral (half swing on stop): 40 swing-equivalents = 20 strokes (`edges_commanded` 40 on each drive block in config.json), whether the last half-cycle lasts 0.25 or 0.5 s. With 41 edges it would be 20.5 strokes, hence the quarter-period margin. The analysis takes the net meniscus advance rest-to-rest (end of the rest after the drive minus the end of the rest before) / 20.
- Check the video log afterwards: `python scripts\check_video.py <run folder>`.
- Runs recorded with the old video code (`video_recording_partNNNN.mp4`, no frames.csv) still have their timing in the burned-in overlay: `python scripts\overlay_to_frames.py <run folder>` reads "Time" and "Freq" from every frame and writes `frames_from_overlay.csv`. Run it with OpenCV 4 (`pip install "opencv-python<5"`): OpenCV 5 draws text with a different font.

### Stroke volume against frequency (configs A and B)
`make_config_stroke_volume_freq.py` writes two configs for 1, 100, 200, 400 and 700 Hz with the 0.5 mm observation tube:
- `config_stroke_volume_freq_A_valves.json` (with valves): bursts of N strokes, sized for about 2.5 mm of meniscus travel. Levels 2.5, 1.5, 1.0 and 0.8 V; 2 bursts per cell; two passes (voltage down, then up). Before each cell a yellow banner at the same voltage is the pause for re-centring the stage. About 31 min plus 40 pauses.
- `config_stroke_volume_freq_B_valveless.json` (valves removed): 10 s per frequency, 20 s rests, at 2.5 and 1.0 V, two passes. About 11 min plus 4 pauses.
- Run them from this branch, so frames.csv and `t_cmd_s` are written. The burst stroke count at 400–700 Hz rests on the host's command timing: check it with an oscilloscope across the coil. Notes, expected travel and expected swing: `Paper/analysis_2026-10-22_stroke_volume/NOTES_2026-10-25_frequency_configs.md`.

### Smoke test of this branch
`make_config_stroke_volume_smoke_test.py` writes `config_stroke_volume_smoke_test.json`: one 2.5 V stop, then 5 strokes at 1 Hz, 25-stroke bursts at 100 and 700 Hz, and 10 s at 400 Hz, with 10 s rests (65 s nominal). Afterwards run `python scripts/check_video.py <run folder>` and `python scripts/check_bursts.py <run folder>`. The second compares each drive's commanded length (from `t_cmd_s` in events.csv) with the config and flags anything more than 0.5 ms off.

### Troubleshooting
- If camera connection fails, check USB connection and camera index
- Camera recording will be automatically disabled if camera is not connected when test starts
- Install OpenCV with: `pip install opencv-python`
