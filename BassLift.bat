@echo off
title BassLift
cd /d "%~dp0"
set "PY=python"
where py >nul 2>nul && set "PY=py -3"
%PY% run.py %*
if errorlevel 1 (
  echo.
  echo === BassLift zakonczyl sie bledem ===
  pause
)
