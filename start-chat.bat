@echo off
title Local Ollama Chat
cd /d "%~dp0"

rem ---- Settings -------------------------------------------------------
set "URL=http://127.0.0.1:5000"
rem Browser for the chat window: msedge or chrome opens a standalone
rem app-style window (no tabs / address bar). Leave empty ("set BROWSER=")
rem to use your default browser instead.
set "BROWSER=msedge"
rem Optional: full path to the python.exe you want to use. Leave empty to
rem auto-detect. Example (conda):
rem   set "PYTHON_EXE=C:\Users\YourName\miniconda3\python.exe"
set "PYTHON_EXE="
rem ---------------------------------------------------------------------

rem Find a REAL Python (the Microsoft Store placeholder does not count).
rem Order: PYTHON_EXE above, py launcher, python on PATH, then common
rem install folders (python.org, Miniconda, Anaconda, Miniforge).
set "PY="
if defined PYTHON_EXE if exist "%PYTHON_EXE%" set PY="%PYTHON_EXE%"
if not defined PY py -3 --version >nul 2>nul && set "PY=py -3"
if not defined PY python --version >nul 2>nul && set "PY=python"
if not defined PY call :findpython
if not defined PY goto nopython
echo Using Python: %PY%

rem First run only: create a private environment, then install dependencies
if not exist ".venv\Scripts\python.exe" (
  echo First run: creating environment...
  %PY% -m venv .venv
  if errorlevel 1 goto fail
)
if not exist ".venv\deps_ok" (
  echo Installing dependencies...
  ".venv\Scripts\python.exe" -m pip install --default-timeout 60 --retries 5 --disable-pip-version-check -r requirements.txt
  if errorlevel 1 goto fail
  echo ok> ".venv\deps_ok"
)

rem Open the browser after a short delay so the server is up by then
if defined BROWSER (
  start "" /min powershell -NoProfile -Command "Start-Sleep 2; Start-Process %BROWSER% '--app=%URL%'"
) else (
  start "" /min powershell -NoProfile -Command "Start-Sleep 2; Start-Process '%URL%'"
)

rem Run the server. Closing this window stops the chat tool.
".venv\Scripts\python.exe" app.py
pause
exit /b 0

:findpython
rem Python is installed but not on PATH: check the usual install folders
for /d %%D in (
  "%LocalAppData%\Programs\Python\Python3*"
  "%ProgramFiles%\Python3*"
  "%UserProfile%\miniconda3"
  "%UserProfile%\anaconda3"
  "%UserProfile%\miniforge3"
  "%UserProfile%\mambaforge"
  "%LocalAppData%\miniconda3"
  "%LocalAppData%\anaconda3"
  "%ProgramData%\miniconda3"
  "%ProgramData%\anaconda3"
) do (
  if not defined PY if exist "%%~D\python.exe" set PY="%%~D\python.exe"
)
exit /b 0

:nopython
echo.
echo Could not find a working Python installation.
echo.
echo If you use conda: open "Anaconda Prompt", run   where python
echo and copy the path it prints into the PYTHON_EXE line at the top of
echo this file, right after the = sign (the first one, not the comment).
echo.
echo Otherwise install Python from https://www.python.org/downloads/
echo.
pause
exit /b 1

:fail
echo.
echo Setup failed - see the messages above.
echo.
pause
exit /b 1
