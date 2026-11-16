@echo off
setlocal
cd /d "%~dp0"
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
if not exist ".venv\Scripts\python.exe" (
  python -m venv .venv
  if errorlevel 1 goto :failed
)
".venv\Scripts\python.exe" -c "import numpy, scipy, matplotlib, PySide6" >nul 2>&1
if errorlevel 1 (
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto :failed
)
".venv\Scripts\python.exe" -m plenoptic_studio
if errorlevel 1 goto :failed
exit /b 0
:failed
echo.
echo Setup or launch failed. Python 3.10+ and internet access are required for first setup.
echo See the README for manual installation and troubleshooting.
pause
exit /b 1
