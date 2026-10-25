@echo off
REM Start the pump test dashboard with the same Python install.bat uses.
REM Extra arguments go to the dashboard, e.g.  run_dashboard.bat --config my_config.json
setlocal
cd /d "%~dp0"
call "%~dp0find_python.bat"
if not defined PY (
    echo ERROR: no working Python 3.8+ found - run install.bat first.
    pause
    exit /b 1
)
echo Starting the dashboard in %CD%
echo with Python: %PY%
%PY% -c "import sys; print('  ', sys.executable, sys.version.split()[0])"
%PY% pump_test_dashboard.py %*
if errorlevel 1 pause
