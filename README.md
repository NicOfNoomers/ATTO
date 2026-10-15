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

## USB Camera Video Recording

The dashboard supports recording video from a USB camera during tests. Video recording is **disabled by default** and must be enabled in the UI.

### Setup
1. Connect your USB camera before starting the dashboard
2. Note the camera index (default is 0 for the first camera)
3. Click "Connect Camera" in the "USB Camera (optional)" section
4. Once connected, enable video recording with the checkbox

### Usage
- During a test, video will automatically record if enabled and camera is connected
- Each frame displays: Time since test start, current frequency, and current flow rate
- Video is saved as `video_recording.mp4` in the test run folder
- Recording automatically stops when the test ends

### Troubleshooting
- If camera connection fails, check USB connection and camera index
- Camera recording will be automatically disabled if camera is not connected when test starts
- Install OpenCV with: `pip install opencv-python`
