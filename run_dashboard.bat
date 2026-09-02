@echo off
REM Section loads dashboard. Needs Python 3.10+. No admin rights.
REM First run creates .venv and installs packages; later runs just start.
setlocal
cd /d "%~dp0"

set PY=
where py >nul 2>&1 && set PY=py -3
if "%PY%"=="" (where python >nul 2>&1 && set PY=python)
if "%PY%"=="" (
  echo Python was not found on this machine.
  echo Install Python 3.10 or newer from python.org. Tick "Install for me only"
  echo and "Add python.exe to PATH". Neither needs admin rights.
  pause
  exit /b 1
)

%PY% -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)"
if errorlevel 1 (
  echo Python 3.10 or newer is required. Found:
  %PY% -V
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating .venv ...
  %PY% -m venv .venv
  if errorlevel 1 (
    echo Could not create a virtual environment. Is the venv module available?
    pause
    exit /b 1
  )
  if exist "wheels\" (
    echo Installing from the wheels folder ...
    ".venv\Scripts\python.exe" -m pip install --no-index --find-links wheels -r requirements.txt --quiet
  ) else (
    echo Installing packages from the internet, a few minutes the first time ...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt --quiet
  )
  if errorlevel 1 (
    echo.
    echo Install failed. If this machine cannot reach the internet, ask for the
    echo wheels folder, drop it next to this file, delete .venv and run again.
    pause
    exit /b 1
  )
)

echo Starting. Close this window to stop the dashboard.
".venv\Scripts\python.exe" -m streamlit run scripts\dashboard.py
pause