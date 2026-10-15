@echo off
REM Sets PY to a working Python 3.8+ for install.bat and run_dashboard.bat.
REM The test PC must use Python 3.10 (dependency issues with newer ones), so 3.10 comes first.
REM Order: argument 1 (a python.exe path), Python 3.10 (py launcher, python3.10), .venv, py -3, python, python3
set "PY="
if not "%~1"=="" (
    "%~1" -c "import sys; sys.exit(sys.version_info < (3, 8))" >nul 2>&1 && set "PY="%~1""
)
if not defined PY for %%P in ("py -3.10" python3.10) do (
    if not defined PY (
        %%~P -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 10) else 1)" >nul 2>&1 && set "PY=%%~P"
    )
)
if not defined PY if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys; sys.exit(sys.version_info < (3, 8))" >nul 2>&1 && set "PY=".venv\Scripts\python.exe""
)
if not defined PY for %%P in ("py -3" python python3) do (
    if not defined PY (
        %%~P -c "import sys; sys.exit(sys.version_info < (3, 8))" >nul 2>&1 && set "PY=%%~P"
    )
)
