@echo off
REM ATTO Pump Test Dashboard - Installation Script (Windows)
REM This script installs all required dependencies for the Pump Testing Suite

echo ============================================
echo ATTO Pump Test Dashboard - Installation
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Check for virtual environment
set USE_VENV=0
set VENV_PATH=.venv

if exist "%VENV_PATH%\Scripts\activate.bat" (
    set USE_VENV=1
    echo Virtual environment found at %VENV_PATH%
) else if exist "venv\Scripts\activate.bat" (
    set VENV_PATH=venv
    set USE_VENV=1
    echo Virtual environment found at %VENV_PATH%
)

if "%USE_VENV%"=="1" (
    echo.
    set /p USE_VENV_CHOICE="Use existing virtual environment? (Y/n): "
    if /i not "!USE_VENV_CHOICE!"=="n" (
        echo Activating virtual environment...
        call "%VENV_PATH%\Scripts\activate.bat"
        goto install_deps
    )
)

echo.
echo Creating new virtual environment...
python -m venv .venv
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1

:install_deps
echo.
echo Installing dependencies from requirements.txt...
echo.
pip install -r requirements.txt
echo.

REM Install Fluigent SDK if available
if exist "fluigent_sdk-23.0.0" (
    echo Installing Fluigent SDK...
    pip install ./fluigent_sdk-23.0.0
    echo.
)

echo ============================================
echo Installation complete!
echo ============================================
echo.
echo Virtual environment: %VENV_NAME%
echo.
echo To run the dashboard:
echo   python pump_test_dashboard.py
echo.
echo For help:
echo   python pump_test_dashboard.py --help
echo.
pause
