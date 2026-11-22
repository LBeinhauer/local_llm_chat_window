@echo off
title Local Ollama Chat
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 echo Python was not found. Install it from python.org and tick "Add to PATH". & pause & exit /b 1

rem First run only: create a private environment and install dependencies
if not exist ".venv\Scripts\python.exe" echo First run: setting things up... & python -m venv .venv & ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt

rem Open the browser after a short delay so the server is up by then.
rem To force a fresh window, replace the line below with e.g.:
rem   start "" msedge --new-window http://127.0.0.1:5000
msedge --app=http://127.0.0.1:5000

rem Run the server. Closing this window stops the chat tool.
".venv\Scripts\python.exe" app.py
pause
