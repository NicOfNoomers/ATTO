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
```

**Linux/Mac:**
```bash
chmod +x install.sh
./install.sh
```

Both scripts will:
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
