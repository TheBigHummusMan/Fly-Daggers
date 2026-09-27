@echo off
rem Train the fly on your recordings, then export a playable policy and run the controls.
rem   train.bat               new run, 8 hours
rem   train.bat 2             new run, 2 hours
rem   train.bat resume        carry on the newest run for 8 hours
rem   train.bat resume 3      ... for 3 hours
rem   train.bat export        only export + controls for the newest run
rem The computer is kept awake while it runs. Close the window (or Ctrl+C) to stop;
rem a stopped run can be resumed.
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (echo Run daggers\setup.bat first. & pause & exit /b 1)
set "PY=%CD%\.venv\Scripts\python.exe"

set "MODE=new"
set "HOURS=8"
if /i "%~1"=="resume" (
    set "MODE=resume"
    if not "%~2"=="" set "HOURS=%~2"
) else if /i "%~1"=="export" (
    set "MODE=export"
) else if not "%~1"=="" (
    set "HOURS=%~1"
)

"%PY%" code\daggers\dataset.py || (pause & exit /b 1)
cd code\daggers
if "%MODE%"=="new"    "%PY%" -u train.py evolve --hours %HOURS% || (pause & exit /b 1)
if "%MODE%"=="resume" "%PY%" -u train.py evolve --resume latest --hours %HOURS% || (pause & exit /b 1)

echo.
echo Exporting the policy...
"%PY%" -u train.py export latest || (pause & exit /b 1)
echo.
echo Controls (the fly should beat blind, shuffled and nobrain)...
"%PY%" -u train.py controls latest --clips 40 --workers 4
echo.
echo Done. Play it with daggers\play.bat
pause
