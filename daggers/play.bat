@echo off
rem The fly plays Devil Daggers with the newest trained policy.
rem   play.bat                the fly plays
rem   play.bat dry            show what it would do, send no input
rem Start the game (borderless) and begin a run. F9 pauses the fly, F10 stops it;
rem touching the mouse or keys gives you control for 2 seconds. After dying it
rem presses R to start the next run.
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (echo Run daggers\setup.bat first. & pause & exit /b 1)
if /i "%~1"=="dry" (
    ".venv\Scripts\python.exe" code\daggers\play.py --dry-run
) else (
    ".venv\Scripts\python.exe" code\daggers\play.py
)
pause
