@echo off
rem Fly Daggers one-time setup: finds Python, makes .venv, installs packages,
rem runs the tests and times the brain on this computer. Safe to run again.
setlocal
cd /d "%~dp0.."

if not exist "data\2025_Connectivity_783.parquet" (
    echo The connectome file data\2025_Connectivity_783.parquet is missing.
    echo Get the whole repository, including the data folder, then run this again.
    pause & exit /b 1
)

if exist ".venv\Scripts\python.exe" goto install

rem Find a 64-bit Python 3.10-3.13 (the versions numba supports)
set "PY="
for %%v in (3.13 3.12 3.11 3.10) do (
    if not defined PY py -%%v -c "import sys; sys.exit(sys.maxsize <= 2**32)" >nul 2>&1 && set "PY=py -%%v"
)
if not defined PY (
    python -c "import sys; sys.exit(not ((3,10) <= sys.version_info[:2] <= (3,13) and sys.maxsize > 2**32))" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo No suitable Python found. Install 64-bit Python 3.13 from https://www.python.org/downloads/
    echo ^(tick "Add python.exe to PATH" in the installer^), then run this again.
    pause & exit /b 1
)
echo Using %PY%
%PY% -m venv .venv || (echo Could not create .venv & pause & exit /b 1)

:install
echo Installing packages (first time takes a few minutes)...
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -r code\daggers\requirements.txt --quiet || (echo Package install failed & pause & exit /b 1)

echo.
echo Running tests...
".venv\Scripts\python.exe" -m unittest tests.test_daggers || (echo Tests failed & pause & exit /b 1)

echo.
echo Timing the brain on this computer, with the busiest possible input.
echo Real play is lighter: about 0.7x here or more is fine for playing.
".venv\Scripts\python.exe" code\daggers\brain.py

echo.
echo Setup done. Next: daggers\record.bat to record yourself playing.
pause
