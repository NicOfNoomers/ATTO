@echo off
REM ATTO Pump Test Dashboard - installation (Windows)
REM   install.bat                         use the first working Python (see find_python.bat)
REM   install.bat C:\Python313\python.exe use this Python
REM Installs into that Python (no new virtual environment), then checks the setup.
setlocal
cd /d "%~dp0"

echo ============================================
echo ATTO Pump Test Dashboard - Installation
echo ============================================
echo.

call "%~dp0find_python.bat" %1
if not defined PY (
    echo ERROR: no working Python 3.8+ found.
    echo Install Python from https://www.python.org/downloads/ and tick "Add python.exe to PATH",
    echo or give the path:  install.bat C:\Python313\python.exe
    pause
    exit /b 1
)
echo Using Python: %PY%
%PY% --version
echo.

echo Installing dependencies from requirements.txt ...
%PY% -m pip install --upgrade pip
%PY% -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: pip install failed - see the messages above.
    pause
    exit /b 1
)

%PY% -c "import Fluigent.SDK" >nul 2>&1
if errorlevel 1 if exist "fluigent_sdk-23.0.0" (
    echo.
    echo Installing the Fluigent SDK ...
    %PY% -m pip install .\fluigent_sdk-23.0.0
)

echo.
echo ============================================
echo Checking the setup (packages, PPK2, Arduino)
echo ============================================
%PY% scripts\check_setup.py

echo.
echo To start the dashboard: double-click run_dashboard.bat
echo   or run:  %PY% pump_test_dashboard.py
echo.
pause
