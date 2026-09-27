@echo off
rem Record yourself playing Devil Daggers (borderless or windowed). F9 pauses, F10 stops.
rem   record.bat              record until F10
rem   record.bat 30           stop after 30 minutes of play
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (echo Run daggers\setup.bat first. & pause & exit /b 1)
if "%~1"=="" (
    ".venv\Scripts\python.exe" code\daggers\record.py
) else (
    ".venv\Scripts\python.exe" code\daggers\record.py --minutes %~1
)
echo.
echo Preparing the recordings for training...
".venv\Scripts\python.exe" code\daggers\dataset.py
pause
