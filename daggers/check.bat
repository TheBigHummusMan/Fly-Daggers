@echo off
rem Run the tests and time the brain with the busiest possible input. Start a
rem Devil Daggers run first to see whether this computer keeps up while the game
rem runs: about 0.7x or more is fine, since real play is lighter than this test.
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (echo Run daggers\setup.bat first. & pause & exit /b 1)
".venv\Scripts\python.exe" -m unittest tests.test_daggers
".venv\Scripts\python.exe" code\daggers\brain.py
pause
